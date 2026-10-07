#!/usr/bin/env python3
"""Minimal one-obstacle MO-IRL example for the planar point mass.

The script reuses the point-mass dynamics and cost features in this repository
and implements the main ingredients of MO-IRL described in arXiv:2505.08619:
trajectory importance weights, suffix sub-sampling, elastic-net regularization,
step acceptance, and a one-trajectory moving window.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.special import logsumexp

from PointMass_model import get_results_from_model
from PointMass_utils import Costs, Obstacle, UReg, XReg, check_collision


@dataclass
class Trajectory:
    xs: np.ndarray
    us: np.ndarray


def build_problem() -> tuple[Costs, list[Obstacle], np.ndarray, np.ndarray, np.ndarray]:
    """Create the simplest diagonal reaching task with one central obstacle."""
    nx, nu = 4, 2
    start = np.array([0.0, 5.0, 0.0, 0.0])
    target = np.array([10.0, 5.0, 0.0, 0.0])
    obstacle = Obstacle(5.0, 5.0, 2.0, 1.0, "Obs")

    costs = Costs()
    costs.add_cost(XReg(nx, target, "Goal"))
    costs.add_cost(XReg(nx, start, "XReg"))
    costs.add_cost(UReg(nu, None, "UReg"))
    costs.add_cost(obstacle)

    # [Goal, XReg, UReg, Obs] for running and terminal costs.  The terminal
    # UReg weight is zero because there is no terminal control input.
    desired_weights = np.array(
        [10.0, 1.0, 1.0, 20000.0, 2000.0, 1.0, 0.0, 10.0]
    )
    desired_weights /= desired_weights.max()
    return costs, [obstacle], start, target, desired_weights


def solve_oc(
    costs: Costs,
    start: np.ndarray,
    weights: np.ndarray,
    horizon: int,
    dt: float,
    max_iterations: int,
    initial: Trajectory | None = None,
    verbose: bool = False,
) -> Trajectory:
    """Solve one point-mass optimal-control problem."""
    if initial is None:
        xs_init = [start.copy() for _ in range(horizon + 1)]
        us_init = [np.zeros(2) for _ in range(horizon)]
    else:
        xs_init = [x.copy() for x in initial.xs]
        us_init = [u.copy() for u in initial.us]

    xs, us, _ = get_results_from_model(
        costs,
        xs_init,
        us_init,
        horizon,
        weights,
        dt,
        max_iterations,
        with_callback=verbose,
    )
    return Trajectory(xs, us)


def suffix_features(
    costs: Costs,
    trajectory: Trajectory,
    dt: float,
    start_indices: np.ndarray,
) -> np.ndarray:
    """Return separate running/terminal feature vectors for trajectory suffixes."""
    nr = costs.nr
    features = []
    for start_index in start_indices:
        running = np.zeros(nr)
        for state, control in zip(
            trajectory.xs[start_index:-1], trajectory.us[start_index:]
        ):
            running += 0.5 * costs.residuals(state, control) ** 2 * dt
        terminal = 0.5 * costs.residuals(trajectory.xs[-1], None) ** 2
        features.append(np.concatenate((running, terminal)))
    return np.asarray(features)


def mo_irl_step(
    weights: np.ndarray,
    expert_features: np.ndarray,
    sample_feature_sets: list[np.ndarray],
    suffix_weights: np.ndarray,
    l1_regularization: float,
    l2_regularization: float,
) -> tuple[np.ndarray, object]:
    """Compute the MO-IRL weight-improvement direction Δw."""
    differences = [features - expert_features for features in sample_feature_sets]

    def objective(delta: np.ndarray) -> float:
        loss = 0.0
        candidate_weights = weights + delta
        for suffix_index, theta in enumerate(suffix_weights):
            log_terms = np.array(
                [
                    -candidate_weights @ feature_differences[suffix_index]
                    for feature_differences in differences
                ]
            )
            # -log(1 / (1 + sum(exp(log_terms)))) evaluated stably.
            loss += theta * np.logaddexp(0.0, logsumexp(log_terms))
        loss += l1_regularization * np.linalg.norm(delta, ord=1)
        loss += 0.5 * l2_regularization * np.dot(delta, delta)
        return float(loss)

    lower_bounds = -weights + 1e-12
    result = minimize(
        objective,
        np.zeros_like(weights),
        method="L-BFGS-B",
        bounds=Bounds(lower_bounds, np.full_like(weights, np.inf)),
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
    )
    return np.asarray(result.x), result


def merit_values(
    weights: np.ndarray, expert_features: np.ndarray, sample_features: np.ndarray
) -> tuple[float, float, np.ndarray]:
    difference = expert_features - sample_features
    weighted_difference = float(weights @ difference)
    m1 = 0.5 * weighted_difference**2
    m2 = float(np.linalg.norm(difference))
    gradient_m1 = weighted_difference * difference
    return m1, m2, gradient_m1


def scale_weights(weights: np.ndarray) -> np.ndarray:
    """Scale non-negative weights so their largest entry is one."""
    maximum = float(np.max(weights))
    if maximum <= 0.0:
        return weights.copy()
    return weights / maximum


def print_cost_comparison(
    costs: Costs,
    expert: Trajectory,
    learned: Trajectory,
    desired_weights: np.ndarray,
    learned_weights: np.ndarray,
    dt: float,
) -> None:
    """Print weight-scaled trajectory costs and their feature contributions."""
    learned_weights_scaled = scale_weights(learned_weights)
    expert_features = suffix_features(costs, expert, dt, np.array([0]))[0]
    learned_features = suffix_features(costs, learned, dt, np.array([0]))[0]

    print("\nTrajectory cost comparison")
    print("  (All learned costs below use learned weights scaled to max=1.)")
    print(
        f"  {'weights':18s} {'expert':>14s} {'MO-IRL':>14s}"
        f" {'MO-IRL - expert':>18s}"
    )
    for label, weights in (
        ("desired", desired_weights),
        ("learned (scaled)", learned_weights_scaled),
    ):
        expert_cost = float(weights @ expert_features)
        learned_cost = float(weights @ learned_features)
        print(
            f"  {label:18s} {expert_cost:14.6e} {learned_cost:14.6e}"
            f" {learned_cost - expert_cost:18.6e}"
        )

    print("\nCost contributions under desired weights")
    print(
        f"  {'feature':18s} {'expert':>14s} {'MO-IRL':>14s}"
        f" {'difference':>14s}"
    )
    expert_contributions = desired_weights * expert_features
    learned_contributions = desired_weights * learned_features
    feature_labels = [f"running {name}" for name in costs.names] + [
        f"terminal {name}" for name in costs.names
    ]
    for label, expert_value, learned_value in zip(
        feature_labels, expert_contributions, learned_contributions
    ):
        print(
            f"  {label:18s} {expert_value:14.6e} {learned_value:14.6e}"
            f" {learned_value - expert_value:14.6e}"
        )
    print(
        f"  {'TOTAL':18s} {expert_contributions.sum():14.6e}"
        f" {learned_contributions.sum():14.6e}"
        f" {(learned_contributions - expert_contributions).sum():14.6e}"
    )


def learn_weights(
    costs: Costs,
    start: np.ndarray,
    expert: Trajectory,
    horizon: int,
    dt: float,
    max_irl_iterations: int,
    max_oc_iterations: int,
    number_of_suffixes: int,
    l1_regularization: float,
    l2_regularization: float,
    feature_tolerance: float,
    verbose_solver: bool,
    obstacles: list[Obstacle] | None = None,
) -> tuple[np.ndarray, list[Trajectory], list[dict[str, float]]]:
    """Run MO-IRL with a moving window containing only the latest sample."""
    nr = costs.nr
    weights = np.full(2 * nr, 0.01)
    weights[nr + 2] = 0.0  # Terminal UReg is structurally zero.

    suffix_indices = np.unique(
        np.linspace(0, horizon - 1, number_of_suffixes, dtype=int)
    )
    suffix_weights = (horizon - suffix_indices + 1) / (horizon + 1)
    expert_features = suffix_features(costs, expert, dt, suffix_indices)

    current = solve_oc(
        costs,
        start,
        weights,
        horizon,
        dt,
        max_oc_iterations,
        verbose=verbose_solver,
    )
    samples = [current]
    history: list[dict[str, float]] = []

    initial_features = suffix_features(costs, current, dt, suffix_indices)
    normalization = max(
        float(np.linalg.norm(expert_features[0] - initial_features[0])), 1e-12
    )
    current_m1, current_m2, current_gradient = merit_values(
        weights, expert_features[0], initial_features[0]
    )
    collision_obstacles = obstacles if obstacles is not None else costs.costs[-1:]

    print(
        "iter  alpha       m2_norm      weight_step  collision  optimizer",
        flush=True,
    )
    print(
        f"{0:4d}  {'-':>5}  {current_m2 / normalization:12.6f}"
        f"  {0.0:11.3e}  {str(check_collision(current.xs, collision_obstacles)):>9}"
        "  initial",
        flush=True,
    )

    c1, c2 = 1e-4, 0.9
    for iteration in range(1, max_irl_iterations + 1):
        current_feature_sets = [
            suffix_features(costs, sample, dt, suffix_indices)
            for sample in samples[-1:]
        ]
        delta, optimizer_result = mo_irl_step(
            weights,
            expert_features,
            current_feature_sets,
            suffix_weights,
            l1_regularization,
            l2_regularization,
        )

        if np.linalg.norm(delta) < 1e-10:
            print("MO-IRL stopped: the weight step is numerically zero.")
            break

        directional_derivative = float(current_gradient @ delta)
        accepted = False
        alpha = 1.0
        for _ in range(10):
            candidate_weights = np.maximum(weights + alpha * delta, 0.0)
            candidate = solve_oc(
                costs,
                start,
                candidate_weights,
                horizon,
                dt,
                max_oc_iterations,
                initial=current,
                verbose=verbose_solver,
            )
            candidate_features = suffix_features(
                costs, candidate, dt, suffix_indices
            )
            candidate_m1, candidate_m2, candidate_gradient = merit_values(
                candidate_weights, expert_features[0], candidate_features[0]
            )

            armijo = candidate_m1 <= (
                current_m1 + c1 * alpha * directional_derivative
            )
            curvature = abs(float(candidate_gradient @ delta)) <= (
                c2 * abs(directional_derivative)
            )
            if candidate_m2 < current_m2 or (armijo and curvature):
                accepted = True
                break
            alpha /= 4.0

        if not accepted:
            print("MO-IRL stopped: no acceptable step was found in 10 trials.")
            break

        step_norm = float(np.linalg.norm(candidate_weights - weights))
        weights = candidate_weights
        current = candidate
        samples.append(current)
        current_m1 = candidate_m1
        current_m2 = candidate_m2
        current_gradient = candidate_gradient
        normalized_m2 = current_m2 / normalization
        collision = check_collision(current.xs, collision_obstacles)
        history.append(
            {
                "iteration": float(iteration),
                "alpha": float(alpha),
                "normalized_feature_error": float(normalized_m2),
                "weight_step": step_norm,
                "optimizer_success": float(optimizer_result.success),
            }
        )
        print(
            f"{iteration:4d}  {alpha:5.3f}  {normalized_m2:12.6f}"
            f"  {step_norm:11.3e}  {str(collision):>9}"
            f"  {optimizer_result.message}",
            flush=True,
        )

        if normalized_m2 <= feature_tolerance:
            print("MO-IRL converged: feature-error tolerance reached.")
            break

    return weights, samples, history


def save_results(
    output_path: Path,
    data_path: Path,
    obstacle: Obstacle,
    start: np.ndarray,
    target: np.ndarray,
    expert: Trajectory,
    samples: list[Trajectory],
    desired_weights: np.ndarray,
    learned_weights: np.ndarray,
    history: list[dict[str, float]],
    show: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(
        plt.Circle(obstacle.c, obstacle.R, color="0.25", alpha=0.55, label="Obstacle")
    )
    ax.add_patch(
        plt.Circle(
            obstacle.c,
            obstacle.R + obstacle.act,
            fill=False,
            linestyle="--",
            color="0.5",
            alpha=0.6,
            label="Activation margin",
        )
    )
    ax.scatter(start[0], start[1], marker="o", s=70, color="black", label="Start")
    ax.scatter(target[0], target[1], marker="*", s=180, color="green", label="Goal")
    ax.plot(expert.xs[:, 0], expert.xs[:, 1], "g-", linewidth=3, label="Expert")
    for sample in samples[:-1]:
        ax.plot(sample.xs[:, 0], sample.xs[:, 1], color="tab:red", alpha=0.2)
    ax.plot(
        samples[-1].xs[:, 0],
        samples[-1].xs[:, 1],
        color="tab:blue",
        linewidth=2.5,
        label="MO-IRL",
    )
    ax.set(xlim=(-1, 11), ylim=(-1, 11), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
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
        "--output", type=Path, default=Path("outputs/mo_irl_one_obstacle.png")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("outputs/mo_irl_one_obstacle.npz")
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    horizon, dt = 30, 0.05  # 1.5 seconds, matching the paper's PM1 case.
    costs, obstacles, start, target, desired_weights = build_problem()

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
        obstacles[0],
        start,
        target,
        expert,
        samples,
        desired_weights,
        learned_weights,
        history,
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
        costs,
        expert,
        samples[-1],
        desired_weights,
        learned_weights,
        dt,
    )
    print(f"\nResult plot: {args.output}")
    print(f"Result data: {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
