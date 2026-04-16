"""CLI for simulating and plotting the SDDE sunspot-number proxy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate the SDDE model and plot the resulting sn time series."
    )
    parser.add_argument("--tau", type=float, required=True, help="Theta parameter tau.")
    parser.add_argument("--T", type=float, required=True, help="Theta parameter T.")
    parser.add_argument("--Nd", type=float, required=True, help="Theta parameter Nd.")
    parser.add_argument("--sigma", type=float, required=True, help="Theta parameter sigma.")
    parser.add_argument("--Bmax", type=float, required=True, help="Theta parameter Bmax.")
    parser.add_argument(
        "--T-warmup",
        dest="T_warmup",
        type=int,
        required=True,
        help="Warmup duration before cropping the output.",
    )
    parser.add_argument(
        "--T-obs",
        dest="T_obs",
        type=int,
        required=True,
        help="Observed duration to retain after warmup.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--saveat",
        type=float,
        default=1.0,
        help="Sampling interval passed to the Julia solver. Default: 1.0.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Euler-Maruyama integration step. Default: 0.1.",
    )
    parser.add_argument(
        "--nsim",
        type=int,
        default=1,
        help="Number of stochastic simulations to overlay. Default: 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the plot is shown interactively.",
    )
    return parser.parse_args()


def _configure_matplotlib(output: Path | None):
    # Use a non-interactive backend only when saving to disk.
    if output is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def main() -> None:
    args = _parse_args()
    if args.nsim < 1:
        raise ValueError("--nsim must be at least 1")

    from . import init_julia, sn

    init_julia()

    theta = (args.tau, args.T, args.Nd, args.sigma, args.Bmax)
    seeds = [None] * args.nsim
    if args.seed is not None:
        seeds = [args.seed + i for i in range(args.nsim)]

    plt = _configure_matplotlib(args.output)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for idx, seed in enumerate(seeds, start=1):
        y = np.asarray(
            sn(
                theta,
                Twarmup=args.T_warmup,
                Tobs=args.T_obs,
                dt=args.dt,
                saveat=args.saveat,
                seed=seed,
            ),
            dtype=float,
        )
        t = np.arange(1, len(y) + 1, dtype=float) * args.saveat
        label = f"sim {idx}" if args.nsim > 1 else None
        ax.plot(t, y, linewidth=1.5, label=label)

    ax.set_xlabel("Observation time")
    ax.set_ylabel("sn")
    title = "SDDE sn trajectory" if args.nsim == 1 else f"SDDE sn trajectories (n={args.nsim})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if args.nsim > 1:
        ax.legend()

    if args.output is not None:
        param_lines = [
            f"tau={args.tau}",
            f"T={args.T}",
            f"Nd={args.Nd}",
            f"sigma={args.sigma}",
            f"Bmax={args.Bmax}",
            f"T_warmup={args.T_warmup}",
            f"T_obs={args.T_obs}",
            f"dt={args.dt}",
            f"saveat={args.saveat}",
            f"nsim={args.nsim}",
            f"seed={args.seed if args.seed is not None else 'random'}",
        ]
        fig.subplots_adjust(bottom=0.28)
        fig.text(
            0.02,
            0.04,
            "Parameters: " + ", ".join(param_lines),
            ha="left",
            va="bottom",
            fontsize=9,
            wrap=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
        return

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
