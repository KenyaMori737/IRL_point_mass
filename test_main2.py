#!/usr/bin/env python3
"""Four-pole PM2-style MO-IRL evaluation based on the original notebook."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from PointMass_utils import Costs, Obstacle, UReg, XReg, check_collision
from test_main1 import (
    Trajectory,
    learn_weights,
    print_cost_comparison,
    scale_weights,
    solve_oc,
)


def build_problem() -> tuple[Costs, list[Obstacle], np.ndarray, np.ndarray, np.ndarray]:
    """Build the four-obstacle setup from PointMass_IRL_AutoReg_6.ipynb."""
    nx, nu = 4, 2
    start = np.array([0.0, 5.0, 0.0, 0.0])
    target = np.array([8.0, 5.0, 0.0, 0.0])
    activation_margin = 1.5
    obstacles = [
        Obstacle(5.5, 2.5, 1.5, activation_margin, "Obs1"),
        Obstacle(5.5, 4.5, 1.5, activation_margin, "Obs2"),
        Obstacle(7.5, 2.5, 1.5, activation_margin, "Obs3"),
        Obstacle(5.5, 6.5, 1.5, activation_margin, "Obs4"),
    ]

    costs = Costs()
    costs.add_cost(XReg(nx, target, "Goal"))
    costs.add_cost(XReg(nx, start, "XReg"))
    costs.add_cost(UReg(nu, None, "UReg"))
    for obstacle in obstacles:
        costs.add_cost(obstacle)

    # Original AutoReg_6 values, with terminal UReg set to structural zero.
    #running = np.array([10.0, 1.0, 1.0, 2000.0, 2000.0, 2000.0, 2000.0])
    #terminal = np.array([300.0, 1.0, 0.0, 10.0, 10.0, 10.0, 10.0])
    running = np.array([0.193776, 0.000000, 0.000000, 0.010681, 7.477067, 1.137010, 5.137517])
    terminal = np.array([2.512091, 0.000000, 0.000000, 0.010116, 4.794077, 2.848665, 1.020947])
    desired_weights = np.concatenate((running, terminal))
    desired_weights /= desired_weights.max()
    return costs, obstacles, start, target, desired_weights


def save_results(
    output_path: Path,
    data_path: Path,
    obstacles: list[Obstacle],
    start: np.ndarray,
    target: np.ndarray,
    expert: Trajectory,
    samples: list[Trajectory],
    desired_weights: np.ndarray,
    learned_weights: np.ndarray,
    history: list[dict[str, float]],
    show_activation_margin: bool,
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    for index, obstacle in enumerate(obstacles, start=1):
        ax.add_patch(
            plt.Circle(obstacle.c, obstacle.R, color="0.25", alpha=0.58)
        )
        ax.text(
            obstacle.c[0],
            obstacle.c[1],
            str(index),
            ha="center",
            va="center",
            color="black",
            fontsize=11,
        )
        if show_activation_margin:
            ax.add_patch(
                plt.Circle(
                    obstacle.c,
                    obstacle.R + obstacle.act,
                    fill=False,
                    linestyle="--",
                    color="0.5",
                    alpha=0.45,
                )
            )

    for sample in samples[:-1]:
        ax.plot(
            sample.xs[:, 0],
            sample.xs[:, 1],
            color="deeppink",
            linestyle=":",
            linewidth=1.8,
            alpha=0.35,
        )
    ax.plot(
        expert.xs[:, 0], expert.xs[:, 1], "g-", linewidth=3, label="Expert"
    )
    ax.plot(
        samples[-1].xs[:, 0],
        samples[-1].xs[:, 1],
        color="tab:blue",
        linewidth=2.5,
        label="MO-IRL",
    )
    ax.scatter(start[0], start[1], s=70, color="black", label="Start", zorder=5)
    ax.scatter(
        target[0],
        target[1],
        marker="s",
        s=180,
        color="#10b94d",
        label="Goal",
        zorder=5,
    )
    ax.set(xlim=(-2, 12), ylim=(-2, 12), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)

    history_array = np.array(
        [
            [
                row["iteration"],
                row["alpha"],
                row["normalized_feature_error"],
                row["weight_step"],
                row["optimizer_success"],
            ]
            for row in history
        ]
    )
    np.savez(
        data_path,
        expert_xs=expert.xs,
        expert_us=expert.us,
        learned_xs=samples[-1].xs,
        learned_us=samples[-1].us,
        desired_weights=desired_weights,
        learned_weights=learned_weights,
        learned_weights_scaled=scale_weights(learned_weights),
        obstacle_centers=np.asarray([obstacle.c for obstacle in obstacles]),
        obstacle_radii=np.asarray([obstacle.R for obstacle in obstacles]),
        obstacle_activation_margins=np.asarray(
            [obstacle.act for obstacle in obstacles]
        ),
        history=history_array,
    )
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-irl-iterations", type=int, default=12)
    parser.add_argument("--max-oc-iterations", type=int, default=100)
    parser.add_argument("--subsamples", type=int, default=20)
    parser.add_argument("--l1", type=float, default=1e-6)
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--feature-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/mo_irl_four_poles.png")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("outputs/mo_irl_four_poles.npz")
    )
    parser.add_argument("--show-activation-margin", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    horizon, dt = 50, 0.05  # 2.5 seconds, matching the paper's PM2 case.
    costs, obstacles, start, target, desired_weights = build_problem()

    print("Four-pole configuration")
    for obstacle in obstacles:
        print(
            f"  {obstacle.name}: center={obstacle.c}, R={obstacle.R}, "
            f"margin={obstacle.act}, activation radius={obstacle.R + obstacle.act}"
        )

    print("Generating the expert trajectory ...", flush=True)
    expert = solve_oc(
        costs,
        start,
        desired_weights,
        horizon,
        dt,
        args.max_oc_iterations,
        verbose=args.verbose_solver,
    )
    print(
        f"Expert collision: {check_collision(expert.xs, obstacles)}; "
        f"terminal position: {expert.xs[-1, :2]}",
        flush=True,
    )

    learned_weights, samples, history = learn_weights(
        costs,
        start,
        expert,
        horizon,
        dt,
        args.max_irl_iterations,
        args.max_oc_iterations,
        args.subsamples,
        args.l1,
        args.l2,
        args.feature_tolerance,
        args.verbose_solver,
        obstacles,
    )
    save_results(
        args.output,
        args.data,
        obstacles,
        start,
        target,
        expert,
        samples,
        desired_weights,
        learned_weights,
        history,
        args.show_activation_margin,
        args.show,
    )

    nr = costs.nr
    learned_weights_scaled = scale_weights(learned_weights)
    print("\nLearned weights (desired -> learned raw -> learned scaled)")
    for index, name in enumerate(costs.names):
        print(
            f"  running  {name:5s}: {desired_weights[index]:10.6f}"
            f" -> {learned_weights[index]:10.6f}"
            f" -> {learned_weights_scaled[index]:10.6f}"
        )
        print(
            f"  terminal {name:5s}: {desired_weights[nr + index]:10.6f}"
            f" -> {learned_weights[nr + index]:10.6f}"
            f" -> {learned_weights_scaled[nr + index]:10.6f}"
        )
    print_cost_comparison(
        costs, expert, samples[-1], desired_weights, learned_weights, dt
    )
    print(f"\nResult plot: {args.output}")
    print(f"Result data: {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
