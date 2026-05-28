import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(x_pad, k, mode="valid")


def _estimate_first_order(y: np.ndarray, u: np.ndarray, dt: np.ndarray) -> Tuple[float, float, float]:
    """Estimate y_dot = a*y + b*u, then tau=-1/a, k=b*tau.

    Returns (k, tau, r2).
    """
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    dt = np.asarray(dt, dtype=float)

    ydot = np.diff(y) / dt
    yk = y[:-1]
    uk = u[:-1]

    Phi = np.column_stack([yk, uk])
    theta, *_ = np.linalg.lstsq(Phi, ydot, rcond=None)
    a, b = float(theta[0]), float(theta[1])

    if a >= -1e-8:
        raise ValueError(
            f"identified a={a:.6g} is not negative; cannot form stable first-order tau=-1/a."
        )

    tau = -1.0 / a
    k = b * tau

    yhat = Phi @ theta
    ss_res = float(np.sum((ydot - yhat) ** 2))
    ss_tot = float(np.sum((ydot - np.mean(ydot)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return k, tau, r2


def _load_npz(path: Path, t_col: str, phi_col: str, theta_col: str, phi_cmd_col: str, theta_cmd_col: str):
    data = np.load(path)
    return (
        np.asarray(data[t_col], dtype=float),
        np.asarray(data[phi_col], dtype=float),
        np.asarray(data[theta_col], dtype=float),
        np.asarray(data[phi_cmd_col], dtype=float),
        np.asarray(data[theta_cmd_col], dtype=float),
    )


def _load_csv(path: Path, t_col: str, phi_col: str, theta_col: str, phi_cmd_col: str, theta_cmd_col: str):
    import csv

    t, phi, theta, phi_cmd, theta_cmd = [], [], [], [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row[t_col]))
            phi.append(float(row[phi_col]))
            theta.append(float(row[theta_col]))
            phi_cmd.append(float(row[phi_cmd_col]))
            theta_cmd.append(float(row[theta_cmd_col]))

    return (
        np.asarray(t, dtype=float),
        np.asarray(phi, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(phi_cmd, dtype=float),
        np.asarray(theta_cmd, dtype=float),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Identify first-order attitude inner-loop params k,tau.")
    p.add_argument("--input", required=True, help="Path to CSV or NPZ with time/attitude/command signals")
    p.add_argument("--t-col", default="t", help="Time column/key")
    p.add_argument("--phi-col", default="phi", help="Roll signal column/key")
    p.add_argument("--theta-col", default="theta", help="Pitch signal column/key")
    p.add_argument("--phi-cmd-col", default="phi_cmd", help="Roll command column/key")
    p.add_argument("--theta-cmd-col", default="theta_cmd", help="Pitch command column/key")
    p.add_argument("--time-unit", choices=["s", "us", "ms"], default="s", help="Input time unit")
    p.add_argument("--angle-unit", choices=["rad", "deg"], default="rad", help="Input angle unit")
    p.add_argument("--smooth-window", type=int, default=1, help="Moving-average window on angle signals")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npz":
        t, phi, theta, phi_cmd, theta_cmd = _load_npz(
            path, args.t_col, args.phi_col, args.theta_col, args.phi_cmd_col, args.theta_cmd_col
        )
    else:
        t, phi, theta, phi_cmd, theta_cmd = _load_csv(
            path, args.t_col, args.phi_col, args.theta_col, args.phi_cmd_col, args.theta_cmd_col
        )

    if args.time_unit == "us":
        t = t * 1e-6
    elif args.time_unit == "ms":
        t = t * 1e-3

    if args.angle_unit == "deg":
        s = np.pi / 180.0
        phi, theta, phi_cmd, theta_cmd = phi * s, theta * s, phi_cmd * s, theta_cmd * s

    # sort + dedup time
    idx = np.argsort(t)
    t, phi, theta, phi_cmd, theta_cmd = t[idx], phi[idx], theta[idx], phi_cmd[idx], theta_cmd[idx]
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    t, phi, theta, phi_cmd, theta_cmd = t[keep], phi[keep], theta[keep], phi_cmd[keep], theta_cmd[keep]

    # optional smoothing
    phi = _moving_average(phi, args.smooth_window)
    theta = _moving_average(theta, args.smooth_window)
    phi_cmd = _moving_average(phi_cmd, args.smooth_window)
    theta_cmd = _moving_average(theta_cmd, args.smooth_window)

    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("non-increasing timestamps after preprocessing")

    k_phi, tau_phi, r2_phi = _estimate_first_order(phi, phi_cmd, dt)
    k_theta, tau_theta, r2_theta = _estimate_first_order(theta, theta_cmd, dt)

    out = {
        "k_phi": k_phi,
        "tau_phi": tau_phi,
        "r2_phi": r2_phi,
        "k_theta": k_theta,
        "tau_theta": tau_theta,
        "r2_theta": r2_theta,
        "n_samples": int(len(t)),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
