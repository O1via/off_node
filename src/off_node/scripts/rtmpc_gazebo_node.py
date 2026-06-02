#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RTMPC Gazebo node (ROS1 + MAVROS).

Fixed workflow target:
- iris linear RTMPC (8-state)
- circular tracking
- receding-horizon QP solve each control cycle

Main loop:
1) read current UAV state from MAVROS (ENU)
2) convert to RTMPC state (NED)
3) solve RTMPC QP
4) publish attitude+thrust command to UAV
"""

import csv
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

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
    build_circle_reference,
    compute_infinite_lqr,
    compute_rpi_box,
    solve_rtmc_qp_with_gp_stagewise,
    solve_rtmc_qp_paper,
    tighten_box_bounds_with_auto_scale,
)


class RtmpcGazeboNode:
    def __init__(self) -> None:
        # --------------------- Parameters ---------------------
        self.rate_hz = float(rospy.get_param("~rate_hz", 10.0))
        self.dt = float(rospy.get_param("~dt", 0.1))
        self.horizon = int(rospy.get_param("~horizon", 30))

        self.auto_offboard_arm = bool(rospy.get_param("~auto_offboard_arm", True))
        self.yaw_deg = float(rospy.get_param("~yaw_deg", 0.0))

        # Circle task
        self.circle_radius = float(rospy.get_param("~circle_radius", 4.0))
        self.circle_period_steps = int(rospy.get_param("~circle_period_steps", 126))
        self.clockwise = bool(rospy.get_param("~clockwise", True))
        self.ref_pd = float(rospy.get_param("~ref_pd", -1.0))  # NED down, so altitude ~1m => pd=-1
        self.tracking_profile = str(rospy.get_param("~tracking_profile", "high_speed_extension"))

        # Pre-align stage: first hover at a staging point, then straight-line entry to
        # the circle start point with matched tangential speed, then switch to RTMPC.
        self.pre_align_enable = bool(rospy.get_param("~pre_align_enable", True))
        self.pre_align_hover_sec = float(rospy.get_param("~pre_align_hover_sec", 3.0))
        self.pre_align_pos_tol_m = float(rospy.get_param("~pre_align_pos_tol_m", 0.20))
        self.pre_align_vel_tol_mps = float(rospy.get_param("~pre_align_vel_tol_mps", 0.25))
        self.pre_align_kp_pos_xy = float(rospy.get_param("~pre_align_kp_pos_xy", 0.45))
        self.pre_align_kd_vel_xy = float(rospy.get_param("~pre_align_kd_vel_xy", 0.55))
        self.pre_align_kp_pos_d = float(rospy.get_param("~pre_align_kp_pos_d", 0.70))
        self.pre_align_kd_vel_d = float(rospy.get_param("~pre_align_kd_vel_d", 0.60))
        self.pre_align_max_tilt_deg = float(rospy.get_param("~pre_align_max_tilt_deg", 12.0))
        self.pre_align_timeout_sec = float(rospy.get_param("~pre_align_timeout_sec", 120.0))
        self.pre_align_force_start_on_timeout = bool(
            rospy.get_param("~pre_align_force_start_on_timeout", False)
        )

        # Straight-line entry stage (from staging point to circle start point).
        self.line_entry_enable = bool(rospy.get_param("~line_entry_enable", True))
        self.line_entry_staging_offset_m = float(rospy.get_param("~line_entry_staging_offset_m", 2.0))
        self.line_entry_hold_sec = float(rospy.get_param("~line_entry_hold_sec", 1.0))
        self.line_entry_timeout_sec = float(rospy.get_param("~line_entry_timeout_sec", 30.0))
        self.line_entry_force_start_on_timeout = bool(
            rospy.get_param("~line_entry_force_start_on_timeout", False)
        )
        self.line_entry_pos_tol_n_m = float(rospy.get_param("~line_entry_pos_tol_n_m", 0.15))
        self.line_entry_pos_tol_e_m = float(rospy.get_param("~line_entry_pos_tol_e_m", 0.15))
        self.line_entry_pos_tol_d_m = float(rospy.get_param("~line_entry_pos_tol_d_m", 0.10))
        self.line_entry_vel_tol_n_mps = float(rospy.get_param("~line_entry_vel_tol_n_mps", 0.20))
        self.line_entry_vel_tol_e_mps = float(rospy.get_param("~line_entry_vel_tol_e_mps", 0.20))
        self.line_entry_vel_tol_d_mps = float(rospy.get_param("~line_entry_vel_tol_d_mps", 0.20))
        self.line_entry_att_tol_rad = float(rospy.get_param("~line_entry_att_tol_rad", 0.20))
        self.line_entry_kp_pos_xy = float(rospy.get_param("~line_entry_kp_pos_xy", 0.45))
        self.line_entry_kd_vel_xy = float(rospy.get_param("~line_entry_kd_vel_xy", 0.55))
        self.line_entry_kp_pos_d = float(rospy.get_param("~line_entry_kp_pos_d", 0.70))
        self.line_entry_kd_vel_d = float(rospy.get_param("~line_entry_kd_vel_d", 0.60))
        self.line_entry_max_tilt_deg = float(rospy.get_param("~line_entry_max_tilt_deg", 12.0))
        self.line_entry_require_qp_feasible = bool(rospy.get_param("~line_entry_require_qp_feasible", True))
        self.line_entry_qp_feasible_eps = float(rospy.get_param("~line_entry_qp_feasible_eps", 1e-6))

        # Disturbance/tube config
        self.disturbance_mode = str(rospy.get_param("~disturbance_mode", "force_only"))
        self.force_bound_mg = float(rospy.get_param("~force_bound_mg", 0.05))
        self.force_d_axis_scale = float(rospy.get_param("~force_d_axis_scale", 0.15))

        # GP options
        self.use_gp = bool(rospy.get_param("~use_gp", False))
        self.gp_model_path = str(rospy.get_param("~gp_model_path", "/home/zxy/work/gp_model/iris_linear_residual_gp.npz"))
        self.gp_beta_sigma = float(rospy.get_param("~gp_beta_sigma", 1.0))
        self.gp_shrink_mode = str(rospy.get_param("~gp_shrink_mode", "residual"))  # none|residual
        self.gp_stagewise_refine_steps = int(rospy.get_param("~gp_stagewise_refine_steps", 1))
        self.gp_grid_points_per_dim = int(rospy.get_param("~gp_grid_points_per_dim", 9))

        # Thrust mapping
        self.mass_kg = float(rospy.get_param("~mass_kg", 1.5))
        self.g = float(rospy.get_param("~g", 9.81))
        self.hover_thrust_norm = float(rospy.get_param("~hover_thrust_norm", 0.60))
        # If <=0: auto set to mass*g/hover_thrust_norm
        self.thrust_to_dT_scale = float(rospy.get_param("~thrust_to_dT_scale", -1.0))

        # Optional sign tuning for frame convention mismatch
        self.state_roll_sign = float(rospy.get_param("~state_roll_sign", 1.0))
        self.state_pitch_sign = float(rospy.get_param("~state_pitch_sign", 1.0))
        self.cmd_roll_sign = float(rospy.get_param("~cmd_roll_sign", 1.0))
        self.cmd_pitch_sign = float(rospy.get_param("~cmd_pitch_sign", 1.0))

        # Topics
        self.pose_topic = str(rospy.get_param("~pose_topic", "mavros/local_position/pose"))
        self.vel_topic = str(rospy.get_param("~vel_topic", "mavros/local_position/velocity_local"))
        self.state_topic = str(rospy.get_param("~state_topic", "mavros/state"))
        self.att_sp_topic = str(rospy.get_param("~att_sp_topic", "mavros/setpoint_raw/attitude"))

        # Diagnostics
        self.diag_enable = bool(rospy.get_param("~diag_enable", True))
        self.diag_csv_path = str(rospy.get_param("~diag_csv_path", "/tmp/rtmpc_diag.csv"))
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
        self.Px, self.K = compute_infinite_lqr(self.A, self.B, self.Qx, self.Ru)

        self.x_min_base, self.x_max_base = base_state_bounds(self.dynamics)
        self.u_min_base, self.u_max_base = base_input_bounds(self.dynamics, mass=self.mass_kg, m=self.m)

        self.gp_model: Optional[VelocityResidualGP] = None
        self._prepare_tube_and_bounds()
        self._prepare_reference()
        self._init_diag_logger()

        rospy.loginfo("[rtmpc_gz] init done")
        rospy.loginfo("[rtmpc_gz] dt=%.3f, horizon=%d, radius=%.2f, period=%d", self.dt, self.horizon,
                      self.circle_radius, self.circle_period_steps)

    # --------------------- setup helpers ---------------------
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
        total_len = max(self.circle_period_steps, self.horizon + 1)
        xy_ref = build_circle_reference(
            x0=np.zeros((4,), dtype=float),
            total_len=total_len,
            dt=self.dt,
            radius=self.circle_radius,
            period_steps=self.circle_period_steps,
            clockwise=self.clockwise,
        )

        x_ref = np.zeros((total_len, self.n), dtype=float)
        # Keep the same convention as your existing python workflow:
        # state=[pn, pe, vn, ve, pd, vd, phi, theta]
        x_ref[:, 0] = xy_ref[:, 0]
        x_ref[:, 1] = xy_ref[:, 1]
        x_ref[:, 2] = xy_ref[:, 2]
        x_ref[:, 3] = xy_ref[:, 3]
        x_ref[:, 4] = float(self.ref_pd)

        self.x_ref_period = apply_tracking_profile_iris(
            x_ref,
            dt=self.dt,
            tracking_profile=self.tracking_profile,
            g=self.g,
            phi_bounds=(float(self.x_min_base[6]), float(self.x_max_base[6])),
            theta_bounds=(float(self.x_min_base[7]), float(self.x_max_base[7])),
        )

        # Circle start point and desired tangential speed for line-entry stage.
        self.start_pos_ned = self.x_ref_period[0, [0, 1, 4]].copy()
        self.start_vel_ned = self.x_ref_period[0, [2, 3, 5]].copy()
        ve_ref = float(self.start_vel_ned[1])
        ve_sign = float(np.sign(ve_ref)) if abs(ve_ref) > 1e-6 else (-1.0 if self.clockwise else 1.0)
        self.stage_pos_ned = self.start_pos_ned.copy()
        # Use (pn_start, pe_start +/- offset, pd_start) so vehicle can build the correct
        # tangential speed before entering RTMPC at the fixed start point.
        self.stage_pos_ned[1] = float(self.start_pos_ned[1] - ve_sign * self.line_entry_staging_offset_m)

    def _line_entry_qp_initial_feasible(self, x: np.ndarray) -> Tuple[bool, str]:
        """Check if current state can satisfy QP initial tube/state intersection.

        Mirrors the initial bound consistency check in solve_rtmc_qp_paper:
        x̄0 must satisfy both tightened state bounds and tube init bounds around x_meas.
        """
        eps = float(max(self.line_entry_qp_feasible_eps, 0.0))
        x_lb = np.maximum(self.x_min_t, x - self.z_half)
        x_ub = np.minimum(self.x_max_t, x + self.z_half)

        bad = np.where(x_ub <= (x_lb + eps))[0]
        if bad.size == 0:
            return True, "ok"

        names = ["pn", "pe", "vn", "ve", "pd", "vd", "phi", "theta"]
        i = int(bad[0])
        name = names[i] if i < len(names) else f"x{i}"
        detail = (
            f"dim={i}({name}), lb={float(x_lb[i]):.6f}, ub={float(x_ub[i]):.6f}, "
            f"x_meas={float(x[i]):.6f}, z={float(self.z_half[i]):.6f}, eps={eps:.2e}"
        )
        return False, detail

    # --------------------- callbacks ---------------------
    def _state_cb(self, msg: State) -> None:
        self.current_state = msg

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.pose_msg = msg

    def _vel_cb(self, msg: TwistStamped) -> None:
        self.vel_msg = msg

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
                "pos_ok", "vel_ok", "att_ok",
                "hold_elapsed", "hold_required",
                "u_dT", "u_phi", "u_theta",
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

    @staticmethod
    def _diag_fail_reason(pos_ok: bool, vel_ok: bool, att_ok: bool) -> str:
        reasons = []
        if not pos_ok:
            reasons.append("pos")
        if not vel_ok:
            reasons.append("vel")
        if not att_ok:
            reasons.append("att")
        return "ok" if not reasons else "+".join(reasons)

    def _diag_log(
        self,
        *,
        now: rospy.Time,
        phase: str,
        x: np.ndarray,
        target_pos_ned: Optional[np.ndarray],
        target_vel_ned: Optional[np.ndarray],
        pos_ok: Optional[bool],
        vel_ok: Optional[bool],
        att_ok: Optional[bool],
        hold_elapsed: Optional[float],
        hold_required: Optional[float],
        u_cmd: Optional[np.ndarray],
        fail_reason: str,
    ) -> None:
        if (not self.diag_enable) or (self._diag_writer is None):
            return

        t_now = float(now.to_sec())
        if self.diag_log_hz > 0.0 and self._diag_last_log_sec is not None:
            if (t_now - float(self._diag_last_log_sec)) < (1.0 / self.diag_log_hz):
                return

        tar_p = np.array([np.nan, np.nan, np.nan], dtype=float)
        tar_v = np.array([np.nan, np.nan, np.nan], dtype=float)
        pos_e = np.array([np.nan, np.nan, np.nan], dtype=float)
        vel_e = np.array([np.nan, np.nan, np.nan], dtype=float)

        if target_pos_ned is not None:
            tar_p = np.asarray(target_pos_ned, dtype=float).reshape(3)
            pos_e = tar_p - np.array([x[0], x[1], x[4]], dtype=float)
        if target_vel_ned is not None:
            tar_v = np.asarray(target_vel_ned, dtype=float).reshape(3)
            vel_e = tar_v - np.array([x[2], x[3], x[5]], dtype=float)

        u = np.array([np.nan, np.nan, np.nan], dtype=float)
        if u_cmd is not None:
            u = np.asarray(u_cmd, dtype=float).reshape(3)

        row = [
            t_now,
            phase,
            float(x[0]), float(x[1]), float(x[4]),
            float(x[2]), float(x[3]), float(x[5]),
            float(x[6]), float(x[7]),
            float(tar_p[0]), float(tar_p[1]), float(tar_p[2]),
            float(tar_v[0]), float(tar_v[1]), float(tar_v[2]),
            float(pos_e[0]), float(pos_e[1]), float(pos_e[2]),
            float(vel_e[0]), float(vel_e[1]), float(vel_e[2]),
            int(pos_ok) if pos_ok is not None else -1,
            int(vel_ok) if vel_ok is not None else -1,
            int(att_ok) if att_ok is not None else -1,
            float(hold_elapsed) if hold_elapsed is not None else float("nan"),
            float(hold_required) if hold_required is not None else float("nan"),
            float(u[0]), float(u[1]), float(u[2]),
            fail_reason,
        ]

        self._diag_writer.writerow(row)
        self._diag_file.flush()
        self._diag_last_log_sec = t_now

    # --------------------- conversion & command ---------------------
    def _enu_to_ned_state(self) -> np.ndarray:
        if self.pose_msg is None or self.vel_msg is None:
            raise RuntimeError("pose/velocity not ready")

        p = self.pose_msg.pose.position
        v = self.vel_msg.twist.linear
        q = self.pose_msg.pose.orientation

        # ENU -> NED
        pn = float(p.y)
        pe = float(p.x)
        pd = float(-p.z)

        vn = float(v.y)
        ve = float(v.x)
        vd = float(-v.z)

        # attitude (small-angle approximation with sign tuners)
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

    def _publish_attitude_cmd(self, dT: float, phi_cmd: float, theta_cmd: float, yaw_cmd: float) -> None:
        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE
            | AttitudeTarget.IGNORE_PITCH_RATE
            | AttitudeTarget.IGNORE_YAW_RATE
        )

        roll = self.cmd_roll_sign * float(phi_cmd)
        pitch = self.cmd_pitch_sign * float(theta_cmd)
        yaw = float(yaw_cmd)

        q = tft.quaternion_from_euler(roll, pitch, yaw)
        msg.orientation.x = float(q[0])
        msg.orientation.y = float(q[1])
        msg.orientation.z = float(q[2])
        msg.orientation.w = float(q[3])
        msg.thrust = self._dT_to_thrust_norm(dT)

        self.att_pub.publish(msg)

    def _ref_window(self, step_idx: int) -> np.ndarray:
        idx = (np.arange(self.horizon + 1, dtype=int) + int(step_idx)) % int(self.x_ref_period.shape[0])
        return self.x_ref_period[idx]

    def _pd_position_velocity_command(
        self,
        x: np.ndarray,
        target_pos_ned: np.ndarray,
        target_vel_ned: Optional[np.ndarray],
        *,
        kp_pos_xy: float,
        kd_vel_xy: float,
        kp_pos_d: float,
        kd_vel_d: float,
        max_tilt_deg: float,
    ) -> np.ndarray:
        """Generic PD controller used by pre-align and line-entry stages."""
        pn_ref, pe_ref, pd_ref = [float(v) for v in target_pos_ned]
        if target_vel_ned is None:
            target_vel_ned = np.zeros((3,), dtype=float)
        vn_ref, ve_ref, vd_ref = [float(v) for v in np.asarray(target_vel_ned).reshape(3)]

        pos_err = np.array([pn_ref - x[0], pe_ref - x[1], pd_ref - x[4]], dtype=float)
        vel_err = np.array([vn_ref - x[2], ve_ref - x[3], vd_ref - x[5]], dtype=float)

        a_n = kp_pos_xy * pos_err[0] + kd_vel_xy * vel_err[0]
        a_e = kp_pos_xy * pos_err[1] + kd_vel_xy * vel_err[1]
        a_d = kp_pos_d * pos_err[2] + kd_vel_d * vel_err[2]

        theta_cmd = float(a_n / max(self.g, 1e-6))
        phi_cmd = float(a_e / max(self.g, 1e-6))
        dT_cmd = float(-self.mass_kg * a_d)

        max_tilt = math.radians(max_tilt_deg)
        phi_cmd = float(np.clip(phi_cmd, -max_tilt, max_tilt))
        theta_cmd = float(np.clip(theta_cmd, -max_tilt, max_tilt))

        # Keep inside controller and physical limits.
        dT_cmd = float(np.clip(dT_cmd, self.u_min_base[0], self.u_max_base[0]))
        phi_cmd = float(np.clip(phi_cmd, self.u_min_base[1], self.u_max_base[1]))
        theta_cmd = float(np.clip(theta_cmd, self.u_min_base[2], self.u_max_base[2]))

        return np.array([dT_cmd, phi_cmd, theta_cmd], dtype=float)

    def _pre_align_command(
        self,
        x: np.ndarray,
        target_pos_ned: np.ndarray,
        target_vel_ned: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._pd_position_velocity_command(
            x=x,
            target_pos_ned=target_pos_ned,
            target_vel_ned=target_vel_ned,
            kp_pos_xy=self.pre_align_kp_pos_xy,
            kd_vel_xy=self.pre_align_kd_vel_xy,
            kp_pos_d=self.pre_align_kp_pos_d,
            kd_vel_d=self.pre_align_kd_vel_d,
            max_tilt_deg=self.pre_align_max_tilt_deg,
        )

    def _line_entry_command(
        self,
        x: np.ndarray,
        target_pos_ned: np.ndarray,
        target_vel_ned: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._pd_position_velocity_command(
            x=x,
            target_pos_ned=target_pos_ned,
            target_vel_ned=target_vel_ned,
            kp_pos_xy=self.line_entry_kp_pos_xy,
            kd_vel_xy=self.line_entry_kd_vel_xy,
            kp_pos_d=self.line_entry_kp_pos_d,
            kd_vel_d=self.line_entry_kd_vel_d,
            max_tilt_deg=self.line_entry_max_tilt_deg,
        )

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

        # Prime setpoints for OFFBOARD entry
        rospy.loginfo("[rtmpc_gz] priming attitude setpoints...")
        for _ in range(max(10, int(2.0 * self.rate_hz))):
            self._publish_attitude_cmd(dT=0.0, phi_cmd=0.0, theta_cmd=0.0, yaw_cmd=yaw_cmd)
            rate.sleep()

        last_request = rospy.Time.now()
        step_idx = 0
        loop_count = 0

        phase = "rtmpc"
        if self.pre_align_enable:
            phase = "pre_align_hover" if self.line_entry_enable else "pre_align_direct"

        pre_align_hold_start: Optional[rospy.Time] = None
        pre_align_t0 = rospy.Time.now()
        line_entry_hold_start: Optional[rospy.Time] = None
        line_entry_t0: Optional[rospy.Time] = None

        if phase == "pre_align_hover":
            rospy.loginfo(
                "[rtmpc_gz] pre_align_hover start -> staging_ned=(%.3f, %.3f, %.3f), hold=%.2fs",
                float(self.stage_pos_ned[0]),
                float(self.stage_pos_ned[1]),
                float(self.stage_pos_ned[2]),
                float(self.pre_align_hover_sec),
            )
        elif phase == "pre_align_direct":
            rospy.loginfo(
                "[rtmpc_gz] pre_align_direct start -> target_ned=(%.3f, %.3f, %.3f), hold=%.2fs",
                float(self.start_pos_ned[0]),
                float(self.start_pos_ned[1]),
                float(self.start_pos_ned[2]),
                float(self.pre_align_hover_sec),
            )
        else:
            rospy.loginfo("[rtmpc_gz] control loop start (rtmpc)")

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.auto_offboard_arm:
                last_request = self._try_set_offboard_and_arm(now, last_request)

            try:
                x = self._enu_to_ned_state()

                if phase in ("pre_align_hover", "pre_align_direct"):
                    target_pos = self.stage_pos_ned if phase == "pre_align_hover" else self.start_pos_ned
                    u_pre = self._pre_align_command(x=x, target_pos_ned=target_pos, target_vel_ned=np.zeros((3,), dtype=float))
                    pos_err = np.array(
                        [target_pos[0] - x[0], target_pos[1] - x[1], target_pos[2] - x[4]],
                        dtype=float,
                    )
                    vel_now = np.array([x[2], x[3], x[5]], dtype=float)
                    pos_ok = float(np.linalg.norm(pos_err)) <= self.pre_align_pos_tol_m
                    vel_ok = float(np.linalg.norm(vel_now)) <= self.pre_align_vel_tol_mps
                    if pos_ok and vel_ok:
                        if pre_align_hold_start is None:
                            pre_align_hold_start = now
                        hold_elapsed = (now - pre_align_hold_start).to_sec()
                    else:
                        pre_align_hold_start = None
                        hold_elapsed = 0.0

                    self._publish_attitude_cmd(
                        dT=float(u_pre[0]),
                        phi_cmd=float(u_pre[1]),
                        theta_cmd=float(u_pre[2]),
                        yaw_cmd=yaw_cmd,
                    )

                    pos_err_n = float(pos_err[0])
                    pos_err_e = float(pos_err[1])
                    pos_err_d = float(pos_err[2])
                    vel_err_n = float(-x[2])
                    vel_err_e = float(-x[3])
                    vel_err_d = float(-x[5])
                    fail_reason = self._diag_fail_reason(pos_ok=bool(pos_ok), vel_ok=bool(vel_ok), att_ok=True)
                    self._diag_log(
                        now=now,
                        phase=phase,
                        x=x,
                        target_pos_ned=target_pos,
                        target_vel_ned=np.zeros((3,), dtype=float),
                        pos_ok=bool(pos_ok),
                        vel_ok=bool(vel_ok),
                        att_ok=True,
                        hold_elapsed=float(hold_elapsed),
                        hold_required=float(self.pre_align_hover_sec),
                        u_cmd=u_pre,
                        fail_reason=fail_reason,
                    )

                    if loop_count % max(1, int(self.rate_hz)) == 0:
                        rospy.loginfo(
                            "[rtmpc_gz][%s] pos_err=(%.3f,%.3f,%.3f) vel_err=(%.3f,%.3f,%.3f) hold=%.2f/%.2f fail=%s",
                            phase,
                            pos_err_n,
                            pos_err_e,
                            pos_err_d,
                            vel_err_n,
                            vel_err_e,
                            vel_err_d,
                            float(hold_elapsed),
                            float(self.pre_align_hover_sec),
                            fail_reason,
                        )

                    if hold_elapsed >= self.pre_align_hover_sec:
                        if phase == "pre_align_hover":
                            phase = "line_entry"
                            line_entry_t0 = now
                            line_entry_hold_start = None
                            rospy.loginfo(
                                "[rtmpc_gz] pre_align_hover finished -> line_entry; start_ned=(%.3f, %.3f, %.3f), v_ref=(%.3f, %.3f, %.3f)",
                                float(self.start_pos_ned[0]),
                                float(self.start_pos_ned[1]),
                                float(self.start_pos_ned[2]),
                                float(self.start_vel_ned[0]),
                                float(self.start_vel_ned[1]),
                                float(self.start_vel_ned[2]),
                            )
                        else:
                            phase = "rtmpc"
                            step_idx = 0
                            rospy.loginfo("[rtmpc_gz] pre_align_direct finished, switch to RTMPC")
                        continue

                    if self.pre_align_timeout_sec > 0.0:
                        elapsed = (now - pre_align_t0).to_sec()
                        if elapsed >= self.pre_align_timeout_sec:
                            if self.pre_align_force_start_on_timeout:
                                if phase == "pre_align_hover":
                                    phase = "line_entry"
                                    line_entry_t0 = now
                                    line_entry_hold_start = None
                                    rospy.logwarn("[rtmpc_gz] pre_align timeout, force switch to line_entry")
                                else:
                                    phase = "rtmpc"
                                    step_idx = 0
                                    rospy.logwarn("[rtmpc_gz] pre_align timeout, force switch to RTMPC")
                            else:
                                rospy.logwarn_throttle(2.0, "[rtmpc_gz] pre_align timeout reached but still not settled")
                    loop_count += 1
                    continue

                if phase == "line_entry":
                    u_line = self._line_entry_command(
                        x=x,
                        target_pos_ned=self.start_pos_ned,
                        target_vel_ned=self.start_vel_ned,
                    )

                    pos_err_n = float(self.start_pos_ned[0] - x[0])
                    pos_err_e = float(self.start_pos_ned[1] - x[1])
                    pos_err_d = float(self.start_pos_ned[2] - x[4])
                    vel_err_n = float(self.start_vel_ned[0] - x[2])
                    vel_err_e = float(self.start_vel_ned[1] - x[3])
                    vel_err_d = float(self.start_vel_ned[2] - x[5])

                    pos_ok = (
                        abs(pos_err_n) <= self.line_entry_pos_tol_n_m
                        and abs(pos_err_e) <= self.line_entry_pos_tol_e_m
                        and abs(pos_err_d) <= self.line_entry_pos_tol_d_m
                    )
                    vel_ok = (
                        abs(vel_err_n) <= self.line_entry_vel_tol_n_mps
                        and abs(vel_err_e) <= self.line_entry_vel_tol_e_mps
                        and abs(vel_err_d) <= self.line_entry_vel_tol_d_mps
                    )
                    att_ok = abs(float(x[6])) <= self.line_entry_att_tol_rad and abs(float(x[7])) <= self.line_entry_att_tol_rad

                    if pos_ok and vel_ok and att_ok:
                        if line_entry_hold_start is None:
                            line_entry_hold_start = now
                        hold_elapsed = (now - line_entry_hold_start).to_sec()
                    else:
                        line_entry_hold_start = None
                        hold_elapsed = 0.0

                    self._publish_attitude_cmd(
                        dT=float(u_line[0]),
                        phi_cmd=float(u_line[1]),
                        theta_cmd=float(u_line[2]),
                        yaw_cmd=yaw_cmd,
                    )

                    fail_reason = self._diag_fail_reason(pos_ok=bool(pos_ok), vel_ok=bool(vel_ok), att_ok=bool(att_ok))
                    self._diag_log(
                        now=now,
                        phase=phase,
                        x=x,
                        target_pos_ned=self.start_pos_ned,
                        target_vel_ned=self.start_vel_ned,
                        pos_ok=bool(pos_ok),
                        vel_ok=bool(vel_ok),
                        att_ok=bool(att_ok),
                        hold_elapsed=float(hold_elapsed),
                        hold_required=float(self.line_entry_hold_sec),
                        u_cmd=u_line,
                        fail_reason=fail_reason,
                    )

                    if loop_count % max(1, int(self.rate_hz)) == 0:
                        rospy.loginfo(
                            "[rtmpc_gz][line_entry] pos_err=(%.3f,%.3f,%.3f) vel_err=(%.3f,%.3f,%.3f) att=(%.3f,%.3f) hold=%.2f/%.2f fail=%s",
                            pos_err_n,
                            pos_err_e,
                            pos_err_d,
                            vel_err_n,
                            vel_err_e,
                            vel_err_d,
                            float(x[6]),
                            float(x[7]),
                            float(hold_elapsed),
                            float(self.line_entry_hold_sec),
                            fail_reason,
                        )

                    if pos_ok and vel_ok and att_ok and (hold_elapsed >= self.line_entry_hold_sec):
                        can_switch = True
                        feas_detail = "disabled"
                        if self.line_entry_require_qp_feasible:
                            can_switch, feas_detail = self._line_entry_qp_initial_feasible(x)

                        if can_switch:
                            phase = "rtmpc"
                            step_idx = 0
                            rospy.loginfo("[rtmpc_gz] line_entry finished, switch to RTMPC")
                            continue

                        rospy.logwarn_throttle(1.0, "[rtmpc_gz] line_entry gate blocked by qp-feasibility: %s", feas_detail)

                    if self.line_entry_timeout_sec > 0.0 and line_entry_t0 is not None:
                        elapsed = (now - line_entry_t0).to_sec()
                        if elapsed >= self.line_entry_timeout_sec:
                            if self.line_entry_force_start_on_timeout:
                                phase = "rtmpc"
                                step_idx = 0
                                rospy.logwarn("[rtmpc_gz] line_entry timeout, force switch to RTMPC")
                            else:
                                rospy.logwarn_throttle(2.0, "[rtmpc_gz] line_entry timeout reached but criteria not met")
                    loop_count += 1
                    continue

                x_des = self._ref_window(step_idx)

                if self.use_gp and self.gp_model is not None:
                    Xbar, Ubar, _, _ = solve_rtmc_qp_with_gp_stagewise(
                        A=self.A,
                        B=self.B,
                        Qx=self.Qx,
                        Ru=self.Ru,
                        Px=self.Px,
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
                        Qx=self.Qx,
                        Ru=self.Ru,
                        Px=self.Px,
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
                u = u_bar + self.K @ (x - x_bar)
                u = np.clip(u, self.u_min_base, self.u_max_base)

                self._publish_attitude_cmd(
                    dT=float(u[0]),
                    phi_cmd=float(u[1]),
                    theta_cmd=float(u[2]),
                    yaw_cmd=yaw_cmd,
                )

                err = x[:6] - x_des[0, :6]
                self._diag_log(
                    now=now,
                    phase=phase,
                    x=x,
                    target_pos_ned=x_des[0, [0, 1, 4]],
                    target_vel_ned=x_des[0, [2, 3, 5]],
                    pos_ok=None,
                    vel_ok=None,
                    att_ok=None,
                    hold_elapsed=None,
                    hold_required=None,
                    u_cmd=u,
                    fail_reason="tracking",
                )

                if step_idx % max(1, int(self.rate_hz)) == 0:
                    rospy.loginfo(
                        "[rtmpc_gz][rtmpc] k=%d |pos_err|=%.3f |vel_err|=%.3f dT=%.3f phi=%.3f th=%.3f",
                        step_idx,
                        float(np.linalg.norm(err[[0, 1, 4]])),
                        float(np.linalg.norm(err[[2, 3, 5]])),
                        float(u[0]),
                        float(u[1]),
                        float(u[2]),
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
