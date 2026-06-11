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
from typing import Dict, Optional, Tuple

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, Wrench
from gazebo_msgs.srv import ApplyBodyWrench
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, SetMode

# Reuse RTMPC implementation modules from the main workspace.
WORK_PY = Path("/home/zxy/work/py")
if WORK_PY.exists() and str(WORK_PY) not in sys.path:
    sys.path.insert(0, str(WORK_PY))

DEFAULT_TORCH_SITE = Path("/home/zxy/work/.venv/lib/python3.8/site-packages")

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


def _import_torch(torch_site: str):
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
        return torch, nn
    except Exception as first_exc:
        site = Path(torch_site).expanduser()
        if site.exists() and str(site) not in sys.path:
            sys.path.insert(0, str(site))
        try:
            import torch  # type: ignore
            from torch import nn  # type: ignore
            return torch, nn
        except Exception as second_exc:
            raise RuntimeError(
                "controller_mode=policy requires torch. "
                f"Tried normal import and torch_site={site}. "
                f"Errors: {first_exc!r}; {second_exc!r}"
            ) from second_exc


def _build_mlp(nn, input_dim: int, output_dim: int, hidden: Tuple[int, ...]):
    layers = []
    dims = (int(input_dim), *tuple(int(h) for h in hidden), int(output_dim))
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _make_policy_input(x: np.ndarray, x_des_window: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=float).reshape(-1), np.asarray(x_des_window, dtype=float).reshape(-1)])


class RtmpcGazeboNode:
    def __init__(self) -> None:
        # --------------------- Parameters ---------------------
        self.rate_hz = float(rospy.get_param("~rate_hz", 10.0))
        self.dt = float(rospy.get_param("~dt", 0.1))
        self.horizon = int(rospy.get_param("~horizon", 30))

        self.auto_offboard_arm = bool(rospy.get_param("~auto_offboard_arm", True))
        self.start_delay_sec = float(rospy.get_param("~start_delay_sec", 1.0))
        self.yaw_deg = float(rospy.get_param("~yaw_deg", 90.0))

        # Controller mode: RTMPC expert or learned DAgger policy.
        self.controller_mode = str(rospy.get_param("~controller_mode", "rtmpc")).strip().lower()
        if self.controller_mode not in ("rtmpc", "policy"):
            raise ValueError("controller_mode must be 'rtmpc' or 'policy'")
        self.policy_checkpoint_path = str(
            rospy.get_param("~policy_checkpoint_path", "/home/zxy/work/dagger_runs/policy_cycle_08.pt")
        )
        self.policy_device = str(rospy.get_param("~policy_device", "cpu"))
        self.policy_torch_site = str(rospy.get_param("~policy_torch_site", str(DEFAULT_TORCH_SITE)))

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

        # Optional physical force injection in Gazebo. force_bound_mg above always
        # defines the RTMPC design bound; this block controls whether an actual
        # external wrench is applied to the simulated vehicle.
        self.disturbance_apply_enable = bool(rospy.get_param("~disturbance_apply_enable", False))
        self.disturbance_apply_only_circle = bool(rospy.get_param("~disturbance_apply_only_circle", True))
        self.disturbance_apply_force_bound_mg = float(
            rospy.get_param("~disturbance_apply_force_bound_mg", -1.0)
        )
        # Number of short wind-gust events to apply in each circular lap.
        # For the default value 2, events are triggered near 1/4 and 3/4 of each lap.
        self.disturbance_events_per_circle = int(rospy.get_param("~disturbance_events_per_circle", 2))
        self.disturbance_update_sec = float(rospy.get_param("~disturbance_update_sec", 0.5))  # legacy continuous mode parameter
        self.disturbance_seed = int(rospy.get_param("~disturbance_seed", 1))
        self.disturbance_body_name = str(rospy.get_param("~disturbance_body_name", "iris::base_link"))
        self.disturbance_reference_frame = str(rospy.get_param("~disturbance_reference_frame", "world"))
        self.disturbance_duration_sec = float(rospy.get_param("~disturbance_duration_sec", 0.25))
        self.disturbance_log_hz = float(rospy.get_param("~disturbance_log_hz", 1.0))
        self.disturbance_direction_mode = str(rospy.get_param("~disturbance_direction_mode", "random")).strip().lower()
        self.disturbance_force_direction_ned = np.asarray(
            rospy.get_param("~disturbance_force_direction_ned", [0.0, 1.0, 0.0]),
            dtype=float,
        ).reshape(3)
        if self.disturbance_direction_mode not in ("random", "fixed_ned"):
            raise ValueError("disturbance_direction_mode must be 'random' or 'fixed_ned'")
        self._disturbance_rng = np.random.default_rng(self.disturbance_seed)
        self._disturbance_force_ned = np.zeros((3,), dtype=float)
        self._disturbance_next_update = rospy.Time(0)
        self._disturbance_last_log = rospy.Time(0)
        self._disturbance_triggered_events = set()
        self._disturbance_active_until = rospy.Time(0)
        self._disturbance_event_id = -1
        self._disturbance_lap_idx = -1
        self._disturbance_event_idx = -1

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
        self.apply_wrench_client = None
        if self.disturbance_apply_enable:
            service_name = "gazebo/apply_body_wrench"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                self.apply_wrench_client = rospy.ServiceProxy(service_name, ApplyBodyWrench)
                rospy.loginfo(
                    "[rtmpc_gz] gazebo wrench disturbance enabled: body=%s, frame=%s, update=%.2fs",
                    self.disturbance_body_name,
                    self.disturbance_reference_frame,
                    self.disturbance_update_sec,
                )
            except Exception as exc:
                rospy.logwarn("[rtmpc_gz] failed to connect %s: %s", service_name, str(exc))
                self.disturbance_apply_enable = False

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
        self.policy_model = None
        self.policy_torch = None
        self._prepare_tube_and_bounds()
        self._prepare_reference()
        self._prepare_policy_if_needed()
        self._init_diag_logger()

        rospy.loginfo("[rtmpc_gz] init done")
        rospy.loginfo("[rtmpc_gz] controller_mode=%s", self.controller_mode)
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

    def _prepare_policy_if_needed(self) -> None:
        if self.controller_mode != "policy":
            return

        torch, nn = _import_torch(self.policy_torch_site)
        ckpt_path = Path(self.policy_checkpoint_path).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"policy checkpoint not found: {ckpt_path}")
        ckpt: Dict = torch.load(str(ckpt_path), map_location="cpu")
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError(f"invalid policy checkpoint format: {ckpt_path}")

        hidden = tuple(int(h) for h in ckpt.get("hidden", (64, 64)))
        input_dim = int(ckpt.get("input_dim", self.n + (self.horizon + 1) * self.n))
        output_dim = int(ckpt.get("output_dim", self.m))
        expected_input_dim = int(self.n + (self.horizon + 1) * self.n)
        if input_dim != expected_input_dim:
            raise ValueError(
                f"policy input_dim mismatch: checkpoint={input_dim}, current={expected_input_dim}. "
                "Check horizon/state dimension."
            )
        if output_dim != self.m:
            raise ValueError(f"policy output_dim mismatch: checkpoint={output_dim}, current={self.m}")

        checks = {
            "reference_mode": "takeoff_circle",
            "entry_steps": int(self.entry_steps),
            "reference_altitude_m": float(self.reference_altitude_m),
            "circle_radius": float(self.circle_radius),
            "circle_period_steps": int(self.circle_period_steps),
        }
        for key, expected in checks.items():
            if key not in ckpt:
                rospy.logwarn("[rtmpc_gz][policy] checkpoint missing metadata: %s", key)
                continue
            got = ckpt[key]
            if isinstance(expected, float):
                if abs(float(got) - expected) > 1e-9:
                    raise ValueError(f"policy checkpoint {key} mismatch: checkpoint={got}, current={expected}")
            else:
                if got != expected:
                    raise ValueError(f"policy checkpoint {key} mismatch: checkpoint={got}, current={expected}")

        model = _build_mlp(nn, input_dim=input_dim, output_dim=output_dim, hidden=hidden)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        if self.policy_device != "cpu":
            model.to(self.policy_device)
        self.policy_model = model
        self.policy_torch = torch
        rospy.loginfo(
            "[rtmpc_gz][policy] loaded checkpoint=%s input_dim=%d output_dim=%d hidden=%s device=%s",
            str(ckpt_path),
            input_dim,
            output_dim,
            hidden,
            self.policy_device,
        )

    def _policy_action(self, x: np.ndarray, x_des: np.ndarray) -> np.ndarray:
        if self.policy_model is None or self.policy_torch is None:
            raise RuntimeError("policy model is not loaded")
        inp = _make_policy_input(x, x_des)
        with self.policy_torch.no_grad():
            xb = self.policy_torch.as_tensor(inp, dtype=self.policy_torch.float32, device=self.policy_device).reshape(1, -1)
            ub = self.policy_model(xb).reshape(-1)
        return ub.detach().cpu().numpy()

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
                "step_idx",
                "pn", "pe", "pd", "vn", "ve", "vd", "phi", "theta",
                "target_pn", "target_pe", "target_pd",
                "target_vn", "target_ve", "target_vd",
                "pos_err_n", "pos_err_e", "pos_err_d",
                "vel_err_n", "vel_err_e", "vel_err_d",
                "u_dT", "u_phi", "u_theta",
                "u_raw_dT", "u_raw_phi", "u_raw_theta",
                "disturbance_active", "disturbance_event_id", "disturbance_lap", "disturbance_event_idx",
                "disturbance_fn", "disturbance_fe", "disturbance_fd",
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
        step_idx: Optional[int] = None,
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

        active = bool(self.disturbance_apply_enable and now <= self._disturbance_active_until)
        force = self._disturbance_force_ned if active else np.zeros(3, dtype=float)
        diag_step = -1 if step_idx is None else int(step_idx)

        self._diag_writer.writerow(
            [
                t_now,
                phase,
                diag_step,
                float(x[0]), float(x[1]), float(x[4]),
                float(x[2]), float(x[3]), float(x[5]),
                float(x[6]), float(x[7]),
                float(tar_p[0]), float(tar_p[1]), float(tar_p[2]),
                float(tar_v[0]), float(tar_v[1]), float(tar_v[2]),
                float(pos_e[0]), float(pos_e[1]), float(pos_e[2]),
                float(vel_e[0]), float(vel_e[1]), float(vel_e[2]),
                float(u[0]), float(u[1]), float(u[2]),
                float(raw[0]), float(raw[1]), float(raw[2]),
                int(active),
                int(self._disturbance_event_id if active else -1),
                int(self._disturbance_lap_idx if active else -1),
                int(self._disturbance_event_idx if active else -1),
                float(force[0]), float(force[1]), float(force[2]),
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


    # --------------------- Gazebo disturbance helpers ---------------------
    def _sample_disturbance_force_ned(self) -> np.ndarray:
        bound_mg = self.disturbance_apply_force_bound_mg
        if bound_mg < 0.0:
            bound_mg = self.force_bound_mg
        bound_mg = max(0.0, float(bound_mg))
        fmax = bound_mg * self.mass_kg * self.g
        if fmax <= 0.0:
            return np.zeros(3, dtype=float)

        if self.disturbance_direction_mode == "fixed_ned":
            direction = np.asarray(self.disturbance_force_direction_ned, dtype=float).reshape(3)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                rospy.logwarn_throttle(2.0, "[rtmpc_gz] fixed disturbance direction is zero; no force applied")
                return np.zeros(3, dtype=float)
            return fmax * direction / norm

        f_axis = fmax / np.sqrt(3.0)
        f_axis_d = f_axis * float(min(max(self.force_d_axis_scale, 0.0), 1.0))
        return np.array(
            [
                self._disturbance_rng.uniform(-f_axis, f_axis),
                self._disturbance_rng.uniform(-f_axis, f_axis),
                self._disturbance_rng.uniform(-f_axis_d, f_axis_d),
            ],
            dtype=float,
        )

    def _circle_disturbance_event_index(self, step_idx: int) -> Optional[int]:
        events = int(self.disturbance_events_per_circle)
        if events <= 0 or self.circle_period_steps <= 0 or step_idx < self.entry_steps:
            return None

        circle_step = int(step_idx - self.entry_steps) % int(self.circle_period_steps)
        event_steps = [
            int(round((i + 0.5) * float(self.circle_period_steps) / float(events)))
            for i in range(events)
        ]
        for event_idx, event_step in enumerate(event_steps):
            event_step = min(max(event_step, 0), int(self.circle_period_steps) - 1)
            if circle_step == event_step:
                return event_idx
        return None

    def _apply_gazebo_disturbance(self, phase: str, now: rospy.Time, step_idx: int) -> None:
        if not self.disturbance_apply_enable or self.apply_wrench_client is None:
            return
        if self.disturbance_apply_only_circle and phase != "rtmpc_circle":
            return

        if phase == "rtmpc_circle":
            event_idx = self._circle_disturbance_event_index(step_idx)
            if event_idx is None:
                return
            lap_idx = int(step_idx - self.entry_steps) // int(self.circle_period_steps)
            event_key = (lap_idx, event_idx)
            if event_key in self._disturbance_triggered_events:
                return
            self._disturbance_triggered_events.add(event_key)
            self._disturbance_force_ned = self._sample_disturbance_force_ned()
            self._disturbance_event_id += 1
            self._disturbance_lap_idx = int(lap_idx)
            self._disturbance_event_idx = int(event_idx)
        else:
            # Legacy fallback for non-circle disturbance experiments.
            update_sec = max(float(self.disturbance_update_sec), float(self.dt))
            if now < self._disturbance_next_update:
                return
            self._disturbance_force_ned = self._sample_disturbance_force_ned()
            self._disturbance_next_update = now + rospy.Duration(update_sec)
            self._disturbance_event_id += 1
            self._disturbance_lap_idx = -1
            self._disturbance_event_idx = -1

        # NED force [Fn, Fe, Fd] -> Gazebo ENU world force [Fx=Fe, Fy=Fn, Fz=-Fd].
        fn, fe, fd = [float(v) for v in self._disturbance_force_ned]
        wrench = Wrench()
        wrench.force.x = fe
        wrench.force.y = fn
        wrench.force.z = -fd
        duration = rospy.Duration(max(float(self.disturbance_duration_sec), 2.0 / max(self.rate_hz, 1e-6)))
        try:
            self.apply_wrench_client(
                body_name=self.disturbance_body_name,
                reference_frame=self.disturbance_reference_frame,
                reference_point=Point(0.0, 0.0, 0.0),
                wrench=wrench,
                start_time=rospy.Time(0),
                duration=duration,
            )
            self._disturbance_active_until = now + duration
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "[rtmpc_gz] apply_body_wrench failed: %s", str(exc))
            return

        self._disturbance_last_log = now
        rospy.loginfo(
            "[rtmpc_gz][disturbance] id=%d lap=%d event=%d mode=%s phase=%s step=%d F_ned=(%.3f, %.3f, %.3f)N F_enu=(%.3f, %.3f, %.3f)N duration=%.2fs",
            int(self._disturbance_event_id),
            int(self._disturbance_lap_idx),
            int(self._disturbance_event_idx),
            self.disturbance_direction_mode,
            phase,
            int(step_idx),
            fn,
            fe,
            fd,
            fe,
            fn,
            -fd,
            float(duration.to_sec()),
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
        rospy.loginfo("[rtmpc_gz] control loop start: full-reference %s", self.controller_mode.upper())

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.auto_offboard_arm:
                last_request = self._try_set_offboard_and_arm(now, last_request)

            try:
                x = self._enu_to_ned_state()
                x_des = self._ref_window(step_idx)
                rtmpc_phase = "rtmpc_entry" if step_idx < self.entry_steps else "rtmpc_circle"
                self._apply_gazebo_disturbance(rtmpc_phase, now, step_idx)
                if self.controller_mode == "policy":
                    u_raw = self._policy_action(x, x_des)
                    u_clipped = np.clip(u_raw, self.u_min_base, self.u_max_base)
                else:
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
                    u_raw=u_raw,
                    step_idx=step_idx,
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
