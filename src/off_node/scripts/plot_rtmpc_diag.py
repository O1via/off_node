#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3D projection
import numpy as np


DEFAULT_CSV = "/home/zxy/off_node/src/off_node/diag_outputs/rtmpc_diag.csv"
DEFAULT_OUT = "/home/zxy/off_node/src/off_node/diag_outputs/rtmpc_diag.png"
DEFAULT_OUT_3D = "/home/zxy/off_node/src/off_node/diag_outputs/rtmpc_diag_3d.png"
DEFAULT_OUT_3D_HTML = "/home/zxy/off_node/src/off_node/diag_outputs/rtmpc_diag_3d.html"


def load_csv(path: Path) -> dict:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty csv: {path}")

    def col(name, cast=float):
        out = []
        for r in rows:
            v = r.get(name, "")
            try:
                out.append(cast(v))
            except Exception:
                out.append(np.nan)
        return np.asarray(out)

    return {
        "rows": rows,
        "t": col("t"),
        "phase": np.asarray([r.get("phase", "") for r in rows], dtype=object),
        "pn": col("pn"),
        "pe": col("pe"),
        "pd": col("pd"),
        "vn": col("vn"),
        "ve": col("ve"),
        "vd": col("vd"),
        "phi": col("phi"),
        "theta": col("theta"),
        "target_pn": col("target_pn"),
        "target_pe": col("target_pe"),
        "target_pd": col("target_pd"),
        "target_vn": col("target_vn"),
        "target_ve": col("target_ve"),
        "target_vd": col("target_vd"),
        "pos_err_n": col("pos_err_n"),
        "pos_err_e": col("pos_err_e"),
        "pos_err_d": col("pos_err_d"),
        "vel_err_n": col("vel_err_n"),
        "vel_err_e": col("vel_err_e"),
        "vel_err_d": col("vel_err_d"),
        "u_dT": col("u_dT"),
        "u_phi": col("u_phi"),
        "u_theta": col("u_theta"),
        "u_raw_dT": col("u_raw_dT"),
        "u_raw_phi": col("u_raw_phi"),
        "u_raw_theta": col("u_raw_theta"),
        "fail_reason": np.asarray([r.get("fail_reason", "") for r in rows], dtype=object),
    }


def phase_mask(phases: np.ndarray, phase: str) -> np.ndarray:
    if phase == "all":
        return np.ones(phases.shape, dtype=bool)
    mask = phases == phase
    if not np.any(mask):
        print(f"[diag] phase '{phase}' not found, plotting all samples")
        return np.ones(phases.shape, dtype=bool)
    return mask


def plot_phase_spans(ax, t: np.ndarray, phases: np.ndarray) -> None:
    for ph, color in (("rtmpc_entry", "tab:green"), ("rtmpc_circle", "tab:purple")):
        idx = np.where(phases == ph)[0]
        if idx.size == 0:
            continue
        ax.axvspan(t[idx[0]], t[idx[-1]], color=color, alpha=0.08, label=ph)



def set_3d_box_aspect(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, z_exag: float) -> None:
    """Use a readable 3D aspect without changing the plotted data."""
    xr = float(np.nanmax(x) - np.nanmin(x))
    yr = float(np.nanmax(y) - np.nanmin(y))
    zr = float(np.nanmax(z) - np.nanmin(z))
    xr = max(xr, 1e-6)
    yr = max(yr, 1e-6)
    zr = max(zr, 1e-6) * max(float(z_exag), 1e-6)
    try:
        ax.set_box_aspect((xr, yr, zr))
    except Exception:
        pass


def plot_trajectory_3d(data: dict, out3d: Path, z_exag: float) -> None:
    """Plot actual/reference trajectory in ENU display coordinates."""
    actual_x = data["pe"]
    actual_y = data["pn"]
    actual_z = -data["pd"]
    ref_x = data["target_pe"]
    ref_y = data["target_pn"]
    ref_z = -data["target_pd"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref_x, ref_y, ref_z, "r--", lw=1.4, label="reference")
    ax.plot(actual_x, actual_y, actual_z, "b-", lw=1.6, label="actual")

    phases = data["phase"]
    for ph, color, marker in (("rtmpc_entry", "tab:green", "o"), ("rtmpc_circle", "tab:purple", "^")):
        idx = np.where(phases == ph)[0]
        if idx.size == 0:
            continue
        first = int(idx[0])
        ax.scatter(
            [actual_x[first]], [actual_y[first]], [actual_z[first]],
            color=color, marker=marker, s=45, label=f"{ph} start",
        )

    ax.scatter([actual_x[0]], [actual_y[0]], [actual_z[0]], color="k", s=35, label="actual start")
    ax.scatter([actual_x[-1]], [actual_y[-1]], [actual_z[-1]], color="tab:blue", s=35, label="actual end")
    ax.set_xlabel("ENU x / pe (m)")
    ax.set_ylabel("ENU y / pn (m)")
    ax.set_zlabel("ENU z / -pd (m)")
    ax.set_title("3D trajectory")
    ax.grid(True, alpha=0.3)
    set_3d_box_aspect(
        ax,
        np.concatenate([actual_x, ref_x]),
        np.concatenate([actual_y, ref_y]),
        np.concatenate([actual_z, ref_z]),
        z_exag=z_exag,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    out3d.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out3d, dpi=180)
    print(f"[diag] saved 3d: {out3d}")


def plot_trajectory_3d_interactive(data: dict, out_html: Path) -> None:
    """Write a self-contained rotatable Plotly 3D trajectory HTML."""
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError("plotly is required for --out3d-html; install with: python -m pip install --user plotly") from exc

    actual_x = data["pe"]
    actual_y = data["pn"]
    actual_z = -data["pd"]
    ref_x = data["target_pe"]
    ref_y = data["target_pn"]
    ref_z = -data["target_pd"]
    t_rel = data["t"] - data["t"][0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ref_x,
            y=ref_y,
            z=ref_z,
            mode="lines",
            name="reference",
            line=dict(color="red", width=5, dash="dash"),
            hovertemplate="ref<br>t=%{customdata:.2f}s<br>pe=%{x:.3f}<br>pn=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            customdata=t_rel,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=actual_x,
            y=actual_y,
            z=actual_z,
            mode="lines",
            name="actual",
            line=dict(color="blue", width=5),
            hovertemplate="actual<br>t=%{customdata:.2f}s<br>pe=%{x:.3f}<br>pn=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            customdata=t_rel,
        )
    )

    for ph, color, symbol in (("rtmpc_entry", "green", "circle"), ("rtmpc_circle", "purple", "diamond")):
        idx = np.where(data["phase"] == ph)[0]
        if idx.size == 0:
            continue
        first = int(idx[0])
        fig.add_trace(
            go.Scatter3d(
                x=[actual_x[first]],
                y=[actual_y[first]],
                z=[actual_z[first]],
                mode="markers",
                name=f"{ph} start",
                marker=dict(color=color, size=7, symbol=symbol),
                hovertemplate=f"{ph} start<br>t=%{{customdata:.2f}}s<br>pe=%{{x:.3f}}<br>pn=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
                customdata=[float(t_rel[first])],
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[actual_x[0], actual_x[-1]],
            y=[actual_y[0], actual_y[-1]],
            z=[actual_z[0], actual_z[-1]],
            mode="markers",
            name="actual start/end",
            marker=dict(color=["black", "royalblue"], size=6),
            text=["actual start", "actual end"],
            hovertemplate="%{text}<br>pe=%{x:.3f}<br>pn=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Interactive 3D trajectory",
        scene=dict(
            xaxis_title="ENU x / pe (m)",
            yaxis_title="ENU y / pn (m)",
            zaxis_title="ENU z / -pd (m)",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=45, b=0),
        height=820,
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
    print(f"[diag] saved interactive 3d: {out_html}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot full-reference RTMPC gazebo diagnostics")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="diagnostic csv path")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output 2D summary figure path")
    ap.add_argument("--out3d", default=DEFAULT_OUT_3D, help="output static 3D trajectory figure path; set empty to skip")
    ap.add_argument("--out3d-html", default=DEFAULT_OUT_3D_HTML, help="output interactive Plotly 3D HTML path; set empty to skip")
    ap.add_argument("--z-exag", type=float, default=3.0, help="vertical display exaggeration for 3D plot")
    ap.add_argument(
        "--phase",
        default="all",
        choices=["all", "rtmpc_entry", "rtmpc_circle"],
        help="phase to focus on",
    )
    ap.add_argument("--show", action="store_true", help="show interactive window")
    args = ap.parse_args()

    data = load_csv(Path(args.csv))
    t = data["t"] - data["t"][0]
    mask = phase_mask(data["phase"], args.phase)

    pos_err_norm = np.linalg.norm(
        np.vstack([data["pos_err_n"], data["pos_err_e"], data["pos_err_d"]]).T,
        axis=1,
    )
    vel_err_norm = np.linalg.norm(
        np.vstack([data["vel_err_n"], data["vel_err_e"], data["vel_err_d"]]).T,
        axis=1,
    )

    fig = plt.figure(figsize=(14, 11))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data["pe"], data["pn"], "b-", lw=1.5, label="actual")
    ax1.plot(data["target_pe"], data["target_pn"], "r--", lw=1.2, label="reference")
    ax1.set_xlabel("pe / ENU x (m)")
    ax1.set_ylabel("pn / ENU y (m)")
    ax1.set_title("NE trajectory")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 2, 2)
    plot_phase_spans(ax2, t, data["phase"])
    ax2.plot(t[mask], data["pos_err_n"][mask], label="pos_err_n")
    ax2.plot(t[mask], data["pos_err_e"][mask], label="pos_err_e")
    ax2.plot(t[mask], data["pos_err_d"][mask], label="pos_err_d")
    ax2.plot(t[mask], pos_err_norm[mask], "k--", lw=1.0, label="|pos_err|")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("position error (m)")
    ax2.set_title(f"Position error ({args.phase})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(2, 2, 3)
    plot_phase_spans(ax3, t, data["phase"])
    ax3.plot(t[mask], data["vel_err_n"][mask], label="vel_err_n")
    ax3.plot(t[mask], data["vel_err_e"][mask], label="vel_err_e")
    ax3.plot(t[mask], data["vel_err_d"][mask], label="vel_err_d")
    ax3.plot(t[mask], vel_err_norm[mask], "k--", lw=1.0, label="|vel_err|")
    ax3.set_xlabel("t (s)")
    ax3.set_ylabel("velocity error (m/s)")
    ax3.set_title(f"Velocity error ({args.phase})")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(2, 2, 4)
    plot_phase_spans(ax4, t, data["phase"])
    ax4.plot(t[mask], data["u_dT"][mask], label="dT_sent")
    ax4.plot(t[mask], data["u_phi"][mask], label="phi_sent")
    ax4.plot(t[mask], data["u_theta"][mask], label="theta_sent")
    if not np.all(np.isnan(data["u_raw_dT"])):
        ax4.plot(t[mask], data["u_raw_dT"][mask], ":", alpha=0.55, label="dT_raw")
        ax4.plot(t[mask], data["u_raw_phi"][mask], ":", alpha=0.55, label="phi_raw")
        ax4.plot(t[mask], data["u_raw_theta"][mask], ":", alpha=0.55, label="theta_raw")
    ax4.plot(t[mask], data["phi"][mask], "--", alpha=0.75, label="phi_state")
    ax4.plot(t[mask], data["theta"][mask], "--", alpha=0.75, label="theta_state")
    ax4.set_xlabel("t (s)")
    ax4.set_ylabel("input / attitude")
    ax4.set_title(f"Commands and attitude ({args.phase})")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")

    unique_phase, counts_phase = np.unique(data["phase"], return_counts=True)
    print("[diag] phase counts:")
    for u, c in zip(unique_phase, counts_phase):
        print(f"  {u}: {int(c)}")
    print(
        "[diag] pos_err norm mean/max:",
        f"{float(np.nanmean(pos_err_norm[mask])):.4f}",
        f"{float(np.nanmax(pos_err_norm[mask])):.4f}",
    )
    print(
        "[diag] vel_err norm mean/max:",
        f"{float(np.nanmean(vel_err_norm[mask])):.4f}",
        f"{float(np.nanmax(vel_err_norm[mask])):.4f}",
    )

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"[diag] saved: {out}")

    if args.out3d:
        plot_trajectory_3d(data, Path(args.out3d), z_exag=args.z_exag)
    if args.out3d_html:
        plot_trajectory_3d_interactive(data, Path(args.out3d_html))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
