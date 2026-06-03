#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RTMPC Gazebo node (ROS1 + MAVROS).

Fixed workflow target:
- iris linear RTMPC (8-state)
- full reference trajectory: takeoff/entry segment + circular tracking
- one RTMPC controller from the beginning; no PD pre-align or controller switch

Main loop:
1) read current UAV state from MAVROS (ENU)
2) convert to RTMPC state (NED)
3) solve RTMPC QP against the full reference trajectory
4) publish attitude+thrust command to UAV
"""

import csv
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, SetMode

# Reuse RTMPC implementation modules from the main workspace.
WORK_PY = Path("/home/zxy/work/py")
if WORK_PY.exists() and str(WORK_PY) not in sys.path:
    sys.path.insert(0, str(WORK_PY))

from gp_residual_model import VelocityResidualGP, residual_shrink_bounds  # noqa: E402
from rtmpc_constants import (  # noqa: E402
    base_input_bounds,
    base_state_bounds,
    disturbance_half_bounds,
    gp_query_state_bounds,
    input_cost_matrix,
    state_cost_matrix,
)
from rtmpc_demo import (  # noqa: E402
    LinearIrisHover,
    apply_tracking_profile_iris,
    build_takeoff_entry_circle_reference,
    compute_infinite_lqr,
    compute_rpi_box,
    solve_rtmc_qp_paper,
    solve_rtmc_qp_with_gp_stagewise,
    tighten_box_bounds_with_auto_scale,
)


class RtmpcGazeboNode:
    def __init__(self) -> None:
        # --------------------- Parameters ---------------------
        self.rate_hz = float(rospy.get_param("~rate_hz", 10.0))
        self.dt = float(rospy.get_param("~dt", 0.1))
        self.horizon = int(rospy.get_param("~horizon", 30))

        self.auto_offboard_arm = bool(rospy.get_param("~auto_offboard_arm", True))
        self.start_delay_sec = float(rospy.get_param("~start_delay_sec", 1.0))
        self.yaw_deg = float(rospy.get_param("~yaw_deg", 90.0))

        # Full reference task, aligned with py/rtmpc_demo.py takeoff_circle mode.
        self.takeoff_start_ned = np.asarray(
            rospy.get_param("~takeoff_start_ned", [0.0, 0.0, 0.0]), dtype=float
        ).reshape(3)
        self.entry_steps = int(rospy.get_param("~entry_steps", 140))
        self.circle_radius = float(rospy.get_param("~circle_radius", 4.0))
        self.reference_altitude_m = float(rospy.get_param("~reference_altitude_m", 2.0))
        # Match the pure-code demo geometry: start at local origin, shift the circle south
        # so the vehicle enters the circle along the tangent direction.
        self.circle_center_ne = np.array(
            [
                self.takeoff_start_ned[0] - self.circle_radius,
                self.takeoff_start_ned[1] - 2.0 * self.circle_radius,
            ],
            dtype=float,
        )
        self.circle_period_steps = int(rospy.get_param("~circle_period_steps", 126))
        self.clockwise = bool(rospy.get_param("~clockwise", True))
        self.tracking_profile = str(rospy.get_param("~tracking_profile", "high_speed_extension"))

        # Softer objective for the takeoff/entry segment. The tube feedback K and
        # tightened bounds still use the nominal RTMPC design; only QP tracking
        # weights are changed before the circular segment.
        self.entry_q_scale = float(rospy.get_param("~entry_q_scale", 0.3))
        self.entry_r_scale = float(rospy.get_param("~entry_r_scale", 3.0))

        # Disturbance/tube config.
        self.disturbance_mode = str(rospy.get_param("~disturbance_mode", "force_only"))
        self.force_bound_mg = float(rospy.get_param("~force_bound_mg", 0.0))
        self.force_d_axis_scale = float(rospy.get_param("~force_d_axis_scale", 0.15))

        # GP options. The RTMPC controller runs throughout the trajectory; GP mean
        # compensation is only applied after the entry segment, where training data is in-distribution.
        self.use_gp = bool(rospy.get_param("~use_gp", False))
        self.gp_model_path = str(
            rospy.get_param("~gp_model_path", "/home/zxy/work/gp_model/iris_linear_residual_gp.npz")
        )
        self.gp_beta_sigma = float(rospy.get_param("~gp_beta_sigma", 1.0))
        self.gp_shrink_mode = str(rospy.get_param("~gp_shrink_mode", "residual"))
        self.gp_stagewise_refine_steps = int(rospy.get_param("~gp_stagewise_refine_steps", 1))
        self.gp_grid_points_per_dim = int(rospy.get_param("~gp_grid_points_per_dim", 9))

        # Vehicle / thrust mapping.
        self.mass_kg = float(rospy.get_param("~mass_kg", 1.5))
        self.g = float(rospy.get_param("~g", 9.81))
        self.hover_thrust_norm = float(rospy.get_param("~hover_thrust_norm", 0.705))
        # If <=0: auto set to mass*g/hover_thrust_norm.
        self.thrust_to_dT_scale = float(rospy.get_param("~thrust_to_dT_scale", -1.0))

        # Output command slew-rate limiting. This protects PX4/Gazebo from impulsive RTMPC commands.
        self.command_slew_enable = bool(rospy.get_param("~command_slew_enable", True))
        self.max_dT_rate_nps = float(rospy.get_param("~max_dT_rate_nps", 8.0))
        self.max_phi_rate_radps = float(rospy.get_param("~max_phi_rate_radps", 0.30))
        self.max_theta_rate_radps = float(rospy.get_param("~max_theta_rate_radps", 0.30))
        self._last_cmd: Optional[np.ndarray] = None
        self._last_cmd_time: Optional[rospy.Time] = None

        # Optional sign tuning for frame convention mismatch.
        self.state_roll_sign = float(rospy.get_param("~state_roll_sign", 1.0))
        self.state_pitch_sign = float(rospy.get_param("~state_pitch_sign", 1.0))
        self.cmd_roll_sign = float(rospy.get_param("~cmd_roll_sign", 1.0))
        self.cmd_pitch_sign = float(rospy.get_param("~cmd_pitch_sign", 1.0))

        # Topics.
        self.pose_topic = str(rospy.get_param("~pose_topic", "mavros/local_position/pose"))
        self.vel_topic = str(rospy.get_param("~vel_topic", "mavros/local_position/velocity_local"))
        self.state_topic = str(rospy.get_param("~state_topic", "mavros/state"))
        self.att_sp_topic = str(rospy.get_param("~att_sp_topic", "mavros/setpoint_raw/attitude"))

        # Diagnostics.
        self.diag_enable = bool(rospy.get_param("~diag_enable", True))
        self.diag_csv_path = str(
            rospy.get_param("~diag_csv_path", "/home/zxy/off_node/src/off_node/diag_outputs/rtmpc_diag.csv")
        )
        self.diag_log_hz = float(rospy.get_param("~diag_log_hz", 10.0))

        # --------------------- ROS io ---------------------
        self.current_state: Optional[State] = None
        self.pose_msg: Optional[PoseStamped] = None
        self.vel_msg: Optional[TwistStamped] = None

        self.state_sub = rospy.Subscriber(self.state_topic, State, self._state_cb, queue_size=20)
        self.pose_sub = rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=50)
        self.vel_sub = rospy.Subscriber(self.vel_topic, TwistStamped, self._vel_cb, queue_size=50)

        self.att_pub = rospy.Publisher(self.att_sp_topic, AttitudeTarget, queue_size=50)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

        # --------------------- RTMPC init ---------------------
        self.dynamics = "iris_linear"
        self.sim = LinearIrisHover(dt=self.dt, mass=self.mass_kg)
        self.A = self.sim.A
        self.B = self.sim.B
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]

        self.Qx = state_cost_matrix(self.dynamics)
        self.Ru = input_cost_matrix(self.dynamics, self.m)
        if self.entry_q_scale <= 0.0 or self.entry_r_scale <= 0.0:
            raise ValueError("entry_q_scale and entry_r_scale must be positive")
        self.Px, self.K = compute_infinite_lqr(self.A, self.B, self.Qx, self.Ru)
        self.Qx_entry = self.Qx * self.entry_q_scale
        self.Ru_entry = self.Ru * self.entry_r_scale
        self.Px_entry, _ = compute_infinite_lqr(self.A, self.B, self.Qx_entry, self.Ru_entry)

        self.x_min_base, self.x_max_base = base_state_bounds(self.dynamics)
        self.u_min_base, self.u_max_base = base_input_bounds(self.dynamics, mass=self.mass_kg, m=self.m)
        self._expand_state_bounds_for_full_reference()

        self.gp_model: Optional[VelocityResidualGP] = None
        self._prepare_tube_and_bounds()
        self._prepare_reference()
        self._init_diag_logger()

        rospy.loginfo("[rtmpc_gz] init done")
        rospy.loginfo(
            "[rtmpc_gz] dt=%.3f, horizon=%d, entry_steps=%d, radius=%.2f, period=%d",
            self.dt,
            self.horizon,
            self.entry_steps,
            self.circle_radius,
            self.circle_period_steps,
        )
        rospy.loginfo(
            "[rtmpc_gz] entry objective scales: Qx*=%.3f, Ru*=%.3f; circle uses nominal weights",
            self.entry_q_scale,
            self.entry_r_scale,
        )

    # --------------------- setup helpers ---------------------
    def _expand_state_bounds_for_full_reference(self) -> None:
        """Expand local task bounds so the configured full reference is feasible."""
        if self.entry_steps < 2:
            raise ValueError("entry_steps must be >= 2")
        if self.circle_period_steps <= 0:
            raise ValueError("circle_period_steps must be positive")

        pos_margin = 1.0
        circle_ne_extent = np.abs(self.circle_center_ne) + float(self.circle_radius)
        max_ne = max(
            float(np.max(np.abs(self.takeoff_start_ned[:2]))) + pos_margin,
            float(np.max(circle_ne_extent)) + pos_margin,
        )
        self.x_min_base = self.x_min_base.copy()
        self.x_max_base = self.x_max_base.copy()
        self.x_min_base[0] = min(float(self.x_min_base[0]), -max_ne)
        self.x_max_base[0] = max(float(self.x_max_base[0]), max_ne)
        self.x_min_base[1] = min(float(self.x_min_base[1]), -max_ne)
        self.x_max_base[1] = max(float(self.x_max_base[1]), max_ne)
        self.x_max_base[4] = max(float(self.x_max_base[4]), float(self.takeoff_start_ned[2]) + 0.2)
        rospy.loginfo(
            "[rtmpc_gz] expanded bounds for full reference: n/e half=%.2f, pd_max=%.2f",
            max_ne,
            float(self.x_max_base[4]),
        )

    def _prepare_tube_and_bounds(self) -> None:
        base_w_half = disturbance_half_bounds(
            self.dynamics,
            dt=self.dt,
            mode=self.disturbance_mode,
            force_bound_mg=self.force_bound_mg,
            force_d_axis_scale=self.force_d_axis_scale,
        )

        gp_unc_half = np.zeros_like(base_w_half)
        gp_comp_half = np.zeros_like(base_w_half)

        if self.use_gp:
            gp_path = Path(self.gp_model_path)
            if not gp_path.exists():
                raise FileNotFoundError(f"GP model not found: {gp_path}")
            self.gp_model = VelocityResidualGP.load(str(gp_path))
            if self.gp_model.dynamics != self.dynamics:
                raise ValueError(f"GP dynamics mismatch: {self.gp_model.dynamics} vs {self.dynamics}")
            if abs(float(self.gp_model.dt) - float(self.dt)) > 1e-9:
                raise ValueError(f"GP dt mismatch: {self.gp_model.dt} vs {self.dt}")
            if int(self.gp_model.state_dim) != int(self.n):
                raise ValueError(f"GP state dim mismatch: {self.gp_model.state_dim} vs {self.n}")

            xq_min, xq_max = gp_query_state_bounds(self.dynamics)
            gp_unc_half = self.gp_model.conservative_uncertainty_bound(
                x_min=xq_min,
                x_max=xq_max,
                beta_sigma=self.gp_beta_sigma,
                grid_points_per_dim=self.gp_grid_points_per_dim,
            )
            gp_comp_half = self.gp_model.conservative_mean_bound(
                x_min=xq_min,
                x_max=xq_max,
                grid_points_per_dim=self.gp_grid_points_per_dim,
            )

        self.w_half = residual_shrink_bounds(
            base_w_half=base_w_half,
            gp_comp_half=gp_comp_half,
            gp_unc_half=gp_unc_half,
            mode=self.gp_shrink_mode,
        )

        A_cl = self.A + self.B @ self.K
        self.z_half = compute_rpi_box(A_cl, self.w_half)
        u_half = np.abs(self.K) @ self.z_half

        self.x_min_t, self.x_max_t, self.gamma_x = tighten_box_bounds_with_auto_scale(
            self.x_min_base, self.x_max_base, self.z_half, name="state"
        )
        self.u_min_t, self.u_max_t, self.gamma_u = tighten_box_bounds_with_auto_scale(
            self.u_min_base, self.u_max_base, u_half, name="input"
        )

        rospy.loginfo("[rtmpc_gz] w_half=%s", np.array2string(self.w_half, precision=6))
        rospy.loginfo("[rtmpc_gz] z_half=%s", np.array2string(self.z_half, precision=6))
        rospy.loginfo("[rtmpc_gz] gamma_x=%.4f, gamma_u=%.4f", self.gamma_x, self.gamma_u)

    def _prepare_reference(self) -> None:
        circle_len = max(self.circle_period_steps, self.horizon + 1)
        total_len = int(self.entry_steps) + int(circle_len)
        x_ref_full = build_takeoff_entry_circle_reference(
            total_len=total_len,
            dt=float(self.dt),
            start_ned=self.takeoff_start_ned,
            radius=float(self.circle_radius),
            period_steps=int(self.circle_period_steps),
            entry_steps=int(self.entry_steps),
            clockwise=bool(self.clockwise),
            circle_center_ne=self.circle_center_ne,
            reference_altitude_m=float(self.reference_altitude_m),
        )
        x_ref_full = apply_tracking_profile_iris(
            x_ref_full,
            dt=self.dt,
            tracking_profile=self.tracking_profile,
            g=self.g,
            phi_bounds=(float(self.x_min_base[6]), float(self.x_max_base[6])),
            theta_bounds=(float(self.x_min_base[7]), float(self.x_max_base[7])),
        )
        self.x_ref_entry = x_ref_full[: self.entry_steps].copy()
        self.x_ref_circle_period = x_ref_full[
            self.entry_steps : self.entry_steps + self.circle_period_steps
        ].copy()
        self.x_ref_full = x_ref_full.copy()
        rospy.loginfo(
            "[rtmpc_gz] full reference: takeoff_start_ned=%s, circle_center_ne=%s, altitude=%.2fm, entry_steps=%d, circle_steps=%d",
            np.array2string(self.takeoff_start_ned, precision=3),
            np.array2string(self.circle_center_ne, precision=3),
            float(self.reference_altitude_m),
            int(self.entry_steps),
            int(self.circle_period_steps),
        )

    # --------------------- callbacks ---------------------
    def _state_cb(self, msg: State) -> None:
        self.current_state = msg

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.pose_msg = msg

    def _vel_cb(self, msg: TwistStamped) -> None:
        self.vel_msg = msg

    # --------------------- diagnostics ---------------------
    def _init_diag_logger(self) -> None:
        self._diag_file = None
        self._diag_writer = None
        self._diag_last_log_sec = None

        if not self.diag_enable:
            return

        diag_path = Path(self.diag_csv_path).expanduser()
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        self._diag_file = diag_path.open("w", newline="")
        self._diag_writer = csv.writer(self._diag_file)
        self._diag_writer.writerow(
            [
                "t",
                "phase",
                "pn", "pe", "pd", "vn", "ve", "vd", "phi", "theta",
                "target_pn", "target_pe", "target_pd",
                "target_vn", "target_ve", "target_vd",
                "pos_err_n", "pos_err_e", "pos_err_d",
                "vel_err_n", "vel_err_e", "vel_err_d",
                "u_dT", "u_phi", "u_theta",
                "u_raw_dT", "u_raw_phi", "u_raw_theta",
                "fail_reason",
            ]
        )
        self._diag_file.flush()
        rospy.on_shutdown(self._close_diag_logger)
        rospy.loginfo("[rtmpc_gz] diag enabled: %s", str(diag_path))

    def _close_diag_logger(self) -> None:
        diag_file = getattr(self, "_diag_file", None)
        if diag_file is not None:
            try:
                diag_file.flush()
            except Exception:
                pass
            try:
                diag_file.close()
            except Exception:
                pass
            self._diag_file = None

    def _diag_log(
        self,
        *,
        now: rospy.Time,
        phase: str,
        x: np.ndarray,
        target_pos_ned: np.ndarray,
        target_vel_ned: np.ndarray,
        u_cmd: Optional[np.ndarray],
        fail_reason: str,
        u_raw: Optional[np.ndarray] = None,
    ) -> None:
        if (not self.diag_enable) or (self._diag_writer is None):
            return

        t_now = float(now.to_sec())
        if self.diag_log_hz > 0.0 and self._diag_last_log_sec is not None:
            if (t_now - float(self._diag_last_log_sec)) < (1.0 / self.diag_log_hz):
                return

        tar_p = np.asarray(target_pos_ned, dtype=float).reshape(3)
        tar_v = np.asarray(target_vel_ned, dtype=float).reshape(3)
        pos_e = tar_p - np.array([x[0], x[1], x[4]], dtype=float)
        vel_e = tar_v - np.array([x[2], x[3], x[5]], dtype=float)

        u = np.array([np.nan, np.nan, np.nan], dtype=float)
        if u_cmd is not None:
            u = np.asarray(u_cmd, dtype=float).reshape(3)
        raw = u.copy()
        if u_raw is not None:
            raw = np.asarray(u_raw, dtype=float).reshape(3)

        self._diag_writer.writerow(
            [
                t_now,
                phase,
                float(x[0]), float(x[1]), float(x[4]),
                float(x[2]), float(x[3]), float(x[5]),
                float(x[6]), float(x[7]),
                float(tar_p[0]), float(tar_p[1]), float(tar_p[2]),
                float(tar_v[0]), float(tar_v[1]), float(tar_v[2]),
                float(pos_e[0]), float(pos_e[1]), float(pos_e[2]),
                float(vel_e[0]), float(vel_e[1]), float(vel_e[2]),
                float(u[0]), float(u[1]), float(u[2]),
                float(raw[0]), float(raw[1]), float(raw[2]),
                fail_reason,
            ]
        )
        self._diag_file.flush()
        self._diag_last_log_sec = t_now

    # --------------------- conversion & command ---------------------
    def _enu_to_ned_state(self) -> np.ndarray:
        if self.pose_msg is None or self.vel_msg is None:
            raise RuntimeError("pose/velocity not ready")

        p = self.pose_msg.pose.position
        v = self.vel_msg.twist.linear
        q = self.pose_msg.pose.orientation

        pn = float(p.y)
        pe = float(p.x)
        pd = float(-p.z)
        vn = float(v.y)
        ve = float(v.x)
        vd = float(-v.z)

        roll_enu, pitch_enu, _ = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        phi = self.state_roll_sign * float(roll_enu)
        theta = self.state_pitch_sign * float(pitch_enu)

        return np.array([pn, pe, vn, ve, pd, vd, phi, theta], dtype=float)

    def _dT_to_thrust_norm(self, dT: float) -> float:
        scale = self.thrust_to_dT_scale
        if scale <= 0.0:
            scale = (self.mass_kg * self.g) / max(self.hover_thrust_norm, 1e-6)
        thrust = self.hover_thrust_norm + float(dT) / float(scale)
        return float(np.clip(thrust, 0.0, 1.0))

    def _apply_command_slew(self, u_cmd: np.ndarray, now: rospy.Time) -> np.ndarray:
        u_cmd = np.asarray(u_cmd, dtype=float).reshape(3)
        if not self.command_slew_enable:
            self._last_cmd = u_cmd.copy()
            self._last_cmd_time = now
            return u_cmd

        if self._last_cmd is None or self._last_cmd_time is None:
            self._last_cmd = np.zeros(3, dtype=float)
            self._last_cmd_time = now

        dt_sec = max(1.0 / max(self.rate_hz, 1e-6), float((now - self._last_cmd_time).to_sec()))
        max_delta = np.array(
            [
                max(0.0, self.max_dT_rate_nps) * dt_sec,
                max(0.0, self.max_phi_rate_radps) * dt_sec,
                max(0.0, self.max_theta_rate_radps) * dt_sec,
            ],
            dtype=float,
        )
        delta = np.clip(u_cmd - self._last_cmd, -max_delta, max_delta)
        u_limited = self._last_cmd + delta
        self._last_cmd = u_limited.copy()
        self._last_cmd_time = now
        return u_limited

    def _publish_attitude_cmd(self, dT: float, phi_cmd: float, theta_cmd: float, yaw_cmd: float) -> np.ndarray:
        now = rospy.Time.now()
        u_sent = self._apply_command_slew(np.array([dT, phi_cmd, theta_cmd], dtype=float), now)

        msg = AttitudeTarget()
        msg.header.stamp = now
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )

        roll = self.cmd_roll_sign * float(u_sent[1])
        pitch = self.cmd_pitch_sign * float(u_sent[2])
        yaw = float(yaw_cmd)

        q = tft.quaternion_from_euler(roll, pitch, yaw)
        msg.orientation.x = float(q[0])
        msg.orientation.y = float(q[1])
        msg.orientation.z = float(q[2])
        msg.orientation.w = float(q[3])
        msg.thrust = self._dT_to_thrust_norm(float(u_sent[0]))
        self.att_pub.publish(msg)
        return u_sent

    def _ref_at(self, step_idx: int) -> np.ndarray:
        k = int(step_idx)
        if k < self.entry_steps:
            return self.x_ref_entry[k].copy()
        circle_idx = (k - self.entry_steps) % int(self.x_ref_circle_period.shape[0])
        return self.x_ref_circle_period[circle_idx].copy()

    def _ref_window(self, step_idx: int) -> np.ndarray:
        return np.vstack([self._ref_at(int(step_idx) + j) for j in range(self.horizon + 1)])

    # --------------------- offboard helpers ---------------------
    def _try_set_offboard_and_arm(self, now: rospy.Time, last_request: rospy.Time) -> rospy.Time:
        if self.current_state is None:
            return last_request

        if (now - last_request) > rospy.Duration(1.0):
            if self.current_state.mode != "OFFBOARD":
                try:
                    self.mode_client(base_mode=0, custom_mode="OFFBOARD")
                except Exception:
                    pass
            elif not self.current_state.armed:
                try:
                    self.arming_client(True)
                except Exception:
                    pass
            return now
        return last_request

    def run(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        yaw_cmd = math.radians(self.yaw_deg)

        rospy.loginfo("[rtmpc_gz] waiting for MAVROS state+pose+vel...")
        while not rospy.is_shutdown():
            if self.current_state is not None and self.pose_msg is not None and self.vel_msg is not None:
                break
            rate.sleep()

        rospy.loginfo("[rtmpc_gz] priming attitude setpoints...")
        for _ in range(max(10, int(2.0 * self.rate_hz))):
            self._publish_attitude_cmd(dT=0.0, phi_cmd=0.0, theta_cmd=0.0, yaw_cmd=yaw_cmd)
            rate.sleep()

        last_request = rospy.Time.now()
        ready_since = None
        rospy.loginfo(
            "[rtmpc_gz] waiting for OFFBOARD+armed, then %.2fs startup delay before RTMPC clock starts",
            self.start_delay_sec,
        )
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.auto_offboard_arm:
                last_request = self._try_set_offboard_and_arm(now, last_request)
            ready = (
                self.current_state is not None
                and self.current_state.mode == "OFFBOARD"
                and bool(self.current_state.armed)
            )
            if ready:
                if ready_since is None:
                    ready_since = now
                if (now - ready_since).to_sec() >= max(0.0, self.start_delay_sec):
                    break
            else:
                ready_since = None
            self._publish_attitude_cmd(dT=0.0, phi_cmd=0.0, theta_cmd=0.0, yaw_cmd=yaw_cmd)
            rate.sleep()

        step_idx = 0
        loop_count = 0
        rospy.loginfo("[rtmpc_gz] control loop start: full-reference RTMPC")

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.auto_offboard_arm:
                last_request = self._try_set_offboard_and_arm(now, last_request)

            try:
                x = self._enu_to_ned_state()
                x_des = self._ref_window(step_idx)
                rtmpc_phase = "rtmpc_entry" if step_idx < self.entry_steps else "rtmpc_circle"
                if rtmpc_phase == "rtmpc_entry":
                    Qx_solve = self.Qx_entry
                    Ru_solve = self.Ru_entry
                    Px_solve = self.Px_entry
                else:
                    Qx_solve = self.Qx
                    Ru_solve = self.Ru
                    Px_solve = self.Px

                if self.use_gp and self.gp_model is not None and rtmpc_phase == "rtmpc_circle":
                    Xbar, Ubar, _, _ = solve_rtmc_qp_with_gp_stagewise(
                        A=self.A,
                        B=self.B,
                        Qx=Qx_solve,
                        Ru=Ru_solve,
                        Px=Px_solve,
                        x_meas=x,
                        x_des=x_des,
                        N=self.horizon,
                        z_half=self.z_half,
                        x_bounds=(self.x_min_t, self.x_max_t),
                        u_bounds=(self.u_min_t, self.u_max_t),
                        gp_model=self.gp_model,
                        gp_beta_sigma=self.gp_beta_sigma,
                        stagewise_refine_steps=self.gp_stagewise_refine_steps,
                    )
                else:
                    Xbar, Ubar = solve_rtmc_qp_paper(
                        A=self.A,
                        B=self.B,
                        Qx=Qx_solve,
                        Ru=Ru_solve,
                        Px=Px_solve,
                        x_meas=x,
                        x_des=x_des,
                        N=self.horizon,
                        z_half=self.z_half,
                        x_bounds=(self.x_min_t, self.x_max_t),
                        u_bounds=(self.u_min_t, self.u_max_t),
                        d_affine=None,
                    )

                x_bar = Xbar[0]
                u_bar = Ubar[0]
                u_raw = u_bar + self.K @ (x - x_bar)
                u_clipped = np.clip(u_raw, self.u_min_base, self.u_max_base)

                u_sent = self._publish_attitude_cmd(
                    dT=float(u_clipped[0]),
                    phi_cmd=float(u_clipped[1]),
                    theta_cmd=float(u_clipped[2]),
                    yaw_cmd=yaw_cmd,
                )

                err = x[:6] - x_des[0, :6]
                self._diag_log(
                    now=now,
                    phase=rtmpc_phase,
                    x=x,
                    target_pos_ned=x_des[0, [0, 1, 4]],
                    target_vel_ned=x_des[0, [2, 3, 5]],
                    u_cmd=u_sent,
                    fail_reason="tracking",
                    u_raw=u_clipped,
                )

                if step_idx % max(1, int(self.rate_hz)) == 0:
                    rospy.loginfo(
                        "[rtmpc_gz][%s] k=%d |pos_err|=%.3f |vel_err|=%.3f dT=%.3f phi=%.3f th=%.3f",
                        rtmpc_phase,
                        step_idx,
                        float(np.linalg.norm(err[[0, 1, 4]])),
                        float(np.linalg.norm(err[[2, 3, 5]])),
                        float(u_sent[0]),
                        float(u_sent[1]),
                        float(u_sent[2]),
                    )

                step_idx += 1

            except Exception as ex:
                rospy.logwarn_throttle(1.0, "[rtmpc_gz] solve/publish failed: %s", str(ex))

            loop_count += 1
            rate.sleep()


def main() -> None:
    rospy.init_node("rtmpc_gazebo_node", anonymous=False)
    node = RtmpcGazeboNode()
    node.run()


if __name__ == "__main__":
    main()
