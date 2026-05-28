#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    data = {
        "rows": rows,
        "t": col("t"),
        "phase": np.asarray([r.get("phase", "") for r in rows], dtype=object),
        "pn": col("pn"),
        "pe": col("pe"),
        "pd": col("pd"),
        "target_pn": col("target_pn"),
        "target_pe": col("target_pe"),
        "target_pd": col("target_pd"),
        "pos_err_n": col("pos_err_n"),
        "pos_err_e": col("pos_err_e"),
        "pos_err_d": col("pos_err_d"),
        "vel_err_n": col("vel_err_n"),
        "vel_err_e": col("vel_err_e"),
        "vel_err_d": col("vel_err_d"),
        "pos_ok": col("pos_ok"),
        "vel_ok": col("vel_ok"),
        "att_ok": col("att_ok"),
        "hold_elapsed": col("hold_elapsed"),
        "hold_required": col("hold_required"),
        "fail_reason": np.asarray([r.get("fail_reason", "") for r in rows], dtype=object),
    }
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot RTMPC gazebo diagnostics")
    ap.add_argument("--csv", default="/home/zxy/off_node/src/off_node/diag_outputs/rtmpc_diag.csv", help="diagnostic csv path")
    ap.add_argument("--out", default="/home/zxy/off_node/src/off_node/diag_outputs/rtmpc_diag.png", help="output figure path")
    ap.add_argument("--phase", default="line_entry", help="phase to focus on (default: line_entry)")
    ap.add_argument("--show", action="store_true", help="show interactive window")
    args = ap.parse_args()

    data = load_csv(Path(args.csv))
    t = data["t"]
    t = t - t[0]

    phase_mask = data["phase"] == args.phase
    if not np.any(phase_mask):
        print(f"[diag] phase '{args.phase}' not found, plotting all samples")
        phase_mask = np.ones_like(t, dtype=bool)

    fig = plt.figure(figsize=(13, 10))

    # 1) trajectory on NE plane
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data["pe"], data["pn"], "b-", lw=1.5, label="actual")
    ax1.plot(data["target_pe"], data["target_pn"], "r--", lw=1.2, label="target")
    ax1.set_xlabel("pe (m)")
    ax1.set_ylabel("pn (m)")
    ax1.set_title("NE trajectory")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # 2) position errors in selected phase
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t[phase_mask], data["pos_err_n"][phase_mask], label="pos_err_n")
    ax2.plot(t[phase_mask], data["pos_err_e"][phase_mask], label="pos_err_e")
    ax2.plot(t[phase_mask], data["pos_err_d"][phase_mask], label="pos_err_d")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("pos err (m)")
    ax2.set_title(f"Position error ({args.phase})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # 3) velocity errors in selected phase
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t[phase_mask], data["vel_err_n"][phase_mask], label="vel_err_n")
    ax3.plot(t[phase_mask], data["vel_err_e"][phase_mask], label="vel_err_e")
    ax3.plot(t[phase_mask], data["vel_err_d"][phase_mask], label="vel_err_d")
    ax3.set_xlabel("t (s)")
    ax3.set_ylabel("vel err (m/s)")
    ax3.set_title(f"Velocity error ({args.phase})")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best")

    # 4) criteria flags and hold timer
    ax4 = fig.add_subplot(2, 2, 4)
    pos_ok = np.nan_to_num(data["pos_ok"], nan=-1.0)
    vel_ok = np.nan_to_num(data["vel_ok"], nan=-1.0)
    att_ok = np.nan_to_num(data["att_ok"], nan=-1.0)
    ax4.plot(t[phase_mask], pos_ok[phase_mask], label="pos_ok")
    ax4.plot(t[phase_mask], vel_ok[phase_mask], label="vel_ok")
    ax4.plot(t[phase_mask], att_ok[phase_mask], label="att_ok")
    ax4.plot(t[phase_mask], data["hold_elapsed"][phase_mask], label="hold_elapsed")
    ax4.plot(t[phase_mask], data["hold_required"][phase_mask], "k--", label="hold_required")
    ax4.set_xlabel("t (s)")
    ax4.set_ylabel("flag / sec")
    ax4.set_title(f"Switch criteria ({args.phase})")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best")

    # summary in terminal
    unique, counts = np.unique(data["fail_reason"][phase_mask], return_counts=True)
    print("[diag] fail_reason counts:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {int(c)}")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"[diag] saved: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
