"""Run measured algorithm evaluations and write an academic evaluation report."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from timetable.constraints import solve_csp, validate_schedule
from timetable.data import load_dataset
from timetable.ml import create_preference_model, model_description
from timetable.optimizer import objective_cost, optimize_detailed, schedule_metrics, soft_score


def main() -> None:
    problem = load_dataset({
        name: ROOT / "demo_data" / f"{name}.csv"
        for name in ("courses", "teachers", "rooms", "groups")
    })
    model = create_preference_model()
    csp_start = time.perf_counter()
    seed, solver_stats = solve_csp(problem, seed=42, return_stats=True, timeout_seconds=90)
    csp_seconds = time.perf_counter() - csp_start
    results: list[dict[str, object]] = []
    initial_cost = objective_cost(problem, seed, model)
    results.append({
        "Algorithm": "Traditional CSP seed",
        "Initial cost": initial_cost,
        "Final cost": initial_cost,
        "Improvement (%)": 0.0,
        "Score": soft_score(initial_cost, len(problem.meetings)),
        "Runtime (s)": csp_seconds,
        "Iterations": 0,
        "Accepted moves": 0,
        "Student idle periods": schedule_metrics(problem, seed)["student_idle_periods"],
        "Teacher fairness (%)": schedule_metrics(problem, seed)["teacher_fairness_pct"],
        "Hard violations": len(validate_schedule(problem, seed)),
    })
    traces: dict[str, list[float]] = {}
    for label, method in (
        ("Simulated Annealing", "annealing"),
        ("Tabu Search", "tabu"),
        ("Hill Climbing", "hill-climbing"),
    ):
        result = optimize_detailed(
            problem,
            seed,
            method,
            model,
            iterations=150,
            seed=42,
            max_runtime_seconds=12.0,
        )
        metrics = schedule_metrics(problem, result.schedule)
        violations = len(validate_schedule(problem, result.schedule))
        improvement = 100.0 * (
            result.initial_cost - result.best_cost
        ) / max(abs(result.initial_cost), 1e-9)
        results.append({
            "Algorithm": label,
            "Initial cost": result.initial_cost,
            "Final cost": result.best_cost,
            "Improvement (%)": improvement,
            "Score": soft_score(result.best_cost, len(problem.meetings)),
            "Runtime (s)": result.runtime_seconds,
            "Iterations": result.iterations,
            "Accepted moves": result.accepted_moves,
            "Student idle periods": metrics["student_idle_periods"],
            "Teacher fairness (%)": metrics["teacher_fairness_pct"],
            "Hard violations": violations,
        })
        traces[label] = [
            float(row["best_cost"])
            for row in result.convergence
        ]

    report = [
        "# IntelliSchedule AI Academic Evaluation Report",
        "",
        f"Measured on: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## 1. Problem definition",
        "",
        f"The instance schedules {len(problem.meetings)} weekly meetings from "
        f"{len({meeting.course_id for meeting in problem.meetings})} sections across "
        f"{len(problem.teachers)} teachers, {len(problem.rooms)} rooms, and "
        f"{len({group for meeting in problem.meetings for group in meeting.groups})} student groups.",
        "Each meeting receives a weekday, contiguous start slot, and compatible room.",
        "",
        "## 2. AI algorithms implemented",
        "",
        "- **CSP backtracking:** dynamic MRV, degree tie-break, forward-checked domains, bounded nodes/time.",
        "- **Simulated Annealing:** feasible single-meeting neighborhood moves, exponential cooling, Metropolis acceptance, best-so-far tracking.",
        "- **Tabu Search:** sampled feasible neighborhood, reverse-move tabu tenure, aspiration when a move beats the global best.",
        "- **Hill Climbing:** best sampled feasible neighbor; stops after a stagnant sampled-neighborhood limit.",
        "- **Means-Ends repair:** compares current assignments with a desired state and uses bounded CSP repair to minimize disruption.",
        "",
        "## 3. Mathematical constraint formulation",
        "",
        "For each meeting `i`, choose exactly one placement `x[i]=(day,start,room)`. "
        "For every pair of meetings sharing a teacher, student group, or room, occupied slot "
        "intervals must be disjoint. Feasible placements also require `room.capacity >= "
        "official_enrollment`, room type/equipment compatibility, teacher availability, "
        "daily teacher hours within the configured maximum, and a two-period contiguous lab block. "
        "Shared elective groups are modeled as a union of occupied student-group resources.",
        "",
        "## 4. Soft-constraint objective function",
        "",
        "The minimized cost combines weighted student idle gaps, per-group daily-load variance, "
        "adjacent heavy/same-subject penalties, teacher preference misses, teacher idle gaps, "
        "classroom-load balance, teacher-load variance, and attendance-weighted slot utility. "
        "The weights shown in the application are configurable/learned only from explicit "
        "preference-feedback rows; attendance prediction does not train preference weights. "
        "Predicted attendance never relaxes official enrollment-based room capacity.",
        "",
        "## 5. Machine-learning methodology",
        "",
        model_description(problem),
        f"Attendance Random Forest temporal/row-order holdout metrics: MAE "
        f"{model.attendance_metrics['mae']:.4f} attendance fraction, RMSE "
        f"{model.attendance_metrics['rmse']:.4f}, R² {model.attendance_metrics['r2']:.3f}.",
        f"Demand Random Forest holdout metrics: MAE {model.demand_metrics['mae']:.2f} students, "
        f"RMSE {model.demand_metrics['rmse']:.2f}, R² {model.demand_metrics['r2']:.3f}.",
        "",
        "## 6. Experimental setup",
        "",
        f"- Synthetic demo: {len(problem.meetings)} weekly meetings; deterministic seed 42.",
        f"- CSP: {solver_stats.nodes:,} search nodes, {solver_stats.backtracks:,} backtracks, "
        f"{csp_seconds:.3f}s measured CSP wall time.",
        "- All metaheuristics start from the same CSP-feasible seed and use the same objective.",
        "- Each optimizer gets up to 150 iterations or 12 seconds; completed work is reported.",
        "- No optimizer may accept a hard-infeasible neighbor; all final schedules are revalidated.",
        "",
        "## 7. Algorithm comparison",
        "",
        "| Algorithm | Initial cost | Final cost | Improvement (%) | Soft score | Runtime (s) | Iterations | Student idle | Teacher fairness (%) | Hard violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        report.append(
            f"| {row['Algorithm']} | {row['Initial cost']:.3f} | {row['Final cost']:.3f} | "
            f"{row['Improvement (%)']:.2f} | {row['Score']:.2f} | {row['Runtime (s)']:.3f} | "
            f"{row['Iterations']} | {row['Student idle periods']:.0f} | "
            f"{row['Teacher fairness (%)']:.2f} | {row['Hard violations']} |"
        )
    valid_results = [row for row in results if row["Hard violations"] == 0]
    best = min(valid_results, key=lambda row: float(row["Final cost"])) if valid_results else None
    report.extend([
        "",
        "## 8. Results and convergence visualization",
        "",
    ])
    if best:
        report.append(
            f"On this measured instance, **{best['Algorithm']}** achieved the lowest final cost "
            f"({best['Final cost']:.3f}) among the zero-violation results. This is an instance-specific "
            "observation, not a universal algorithm ranking."
        )
    report.extend(["", "Best-cost trace (lower is better; sampled every 10 points):", ""])
    for algorithm, values in traces.items():
        points = values[:: max(1, len(values) // 10)]
        report.append(f"- **{algorithm}:** " + " → ".join(f"{value:.1f}" for value in points))
    report.extend([
        "",
        "The Streamlit Optimization Lab plots complete per-iteration traces and measured "
        "improvements interactively. Runtimes vary by hardware and dataset.",
        "",
        "## 9. Limitations",
        "",
        "- Demo attendance and enrollment data are synthetic; their metrics do not establish real-world prediction quality.",
        "- The weekly discretization uses equal-length slots and omits travel time, multi-campus distances, and individual student enrollment lists.",
        "- CSP and repair search are bounded; failure/timeout means no solution was found within the stated bound, not proof of infeasibility.",
        "- Preference feedback must be explicit, representative, and governed; expert defaults are used when no feedback is supplied.",
        "",
        "## 10. Future improvements",
        "",
        "Integrate authorized SIS attendance/enrollment histories, add campus travel-time and room accessibility constraints, "
        "test on multiple real-world instances, add multi-seed confidence intervals, and benchmark the custom CSP against OR-Tools.",
        "",
    ])
    output = ROOT / "evaluation_report.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output}")
    print("\n".join(report[70:77]))


if __name__ == "__main__":
    main()
