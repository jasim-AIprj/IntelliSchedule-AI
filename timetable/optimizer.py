"""Soft-constraint objective and metaheuristic optimization baselines."""

from __future__ import annotations

import math
import random
import time
import copy
from collections import defaultdict
from dataclasses import dataclass
from .constraints import Occupancy, meeting_domains, validate_schedule
from .ml import PreferenceModel
from .models import Meeting, Placement, Problem


@dataclass(frozen=True)
class OptimizationResult:
    schedule: dict[str, Placement]
    initial_cost: float
    best_cost: float
    convergence: tuple[dict[str, float | int | bool], ...]
    iterations: int
    accepted_moves: int
    runtime_seconds: float
    method: str
    stop_reason: str
    rejected_moves: int = 0
    reheats: int = 0
    rejection_reasons: tuple[tuple[str, int], ...] = ()


def schedule_metrics(problem: Problem, schedule: dict[str, Placement]) -> dict[str, float]:
    meetings = {meeting.meeting_id: meeting for meeting in problem.meetings}
    group_slots: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    teacher_slots: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    room_hours: dict[str, int] = defaultdict(int)
    if not schedule:
        return {
            "student_idle_periods": 0.0,
            "teacher_idle_periods": 0.0,
            "classroom_utilization_pct": 0.0,
            "rooms_used": 0.0,
            "teacher_fairness_pct": 100.0,
            "teacher_preference_satisfaction_pct": 100.0,
            "teacher_preference_periods": 0.0,
            "scheduled_hours": 0.0,
        }
    preferred_hits = 0
    preferred_total = 0
    assigned_hours = 0
    for meeting_id, placement in schedule.items():
        meeting = meetings[meeting_id]
        slots = set(range(placement.start, placement.start + meeting.duration))
        assigned_hours += meeting.duration
        room_hours[placement.room_id] += meeting.duration
        teacher_slots[meeting.teacher_id][placement.day].update(slots)
        for group_id in meeting.groups:
            group_slots[group_id][placement.day].update(slots)
        preferences = problem.teachers[meeting.teacher_id].preferred_slots
        if preferences:
            preferred_total += meeting.duration
            preferred_hits += sum((placement.day, slot) in preferences for slot in slots)

    def idle_hours(resource_slots: dict[str, dict[int, set[int]]]) -> int:
        total = 0
        for by_day in resource_slots.values():
            for slots in by_day.values():
                if slots:
                    total += max(slots) - min(slots) + 1 - len(slots)
        return total

    teacher_loads = {
        teacher_id: sum(len(slots) for slots in by_day.values())
        for teacher_id, by_day in teacher_slots.items()
    }
    mean = sum(teacher_loads.values()) / max(1, len(problem.teachers))
    deviation = math.sqrt(
        sum((teacher_loads.get(teacher_id, 0) - mean) ** 2 for teacher_id in problem.teachers)
        / max(1, len(problem.teachers))
    )
    fairness = 100.0 if mean == 0 else max(0.0, 100.0 * (1.0 - deviation / mean))
    possible_room_hours = len(problem.rooms) * len(problem.days) * problem.slots_per_day
    return {
        "student_idle_periods": float(idle_hours(group_slots)),
        "teacher_idle_periods": float(idle_hours(teacher_slots)),
        "classroom_utilization_pct": 100.0 * assigned_hours / max(1, possible_room_hours),
        "rooms_used": float(len(room_hours)),
        "teacher_fairness_pct": fairness,
        "teacher_preference_satisfaction_pct": (
            100.0 * preferred_hits / preferred_total if preferred_total else 100.0
        ),
        "teacher_preference_periods": float(preferred_total),
        "scheduled_hours": float(assigned_hours),
    }


def objective_cost(
    problem: Problem,
    schedule: dict[str, Placement],
    model: PreferenceModel | None = None,
) -> float:
    if model is not None:
        model.precompute(problem.meetings, len(problem.days), problem.slots_per_day)
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    by_group_day: dict[tuple[str, int], list[tuple[int, int, Meeting]]] = defaultdict(list)
    group_loads: dict[str, list[int]] = defaultdict(lambda: [0] * len(problem.days))
    teacher_loads: dict[str, int] = defaultdict(int)
    teacher_day_slots: dict[tuple[str, int], set[int]] = defaultdict(set)
    room_loads: dict[str, int] = defaultdict(int)
    weights = model.preference_weights if model is not None else {
        "student_gaps": 2.0,
        "daily_load": 0.09,
        "heavy_adjacent": 2.0,
        "teacher_preferences": 1.25,
        "teacher_gaps": 1.0,
        "room_balance": 0.005,
        "attendance": 10.0,
    }
    cost = 0.0
    for meeting_id, placement in schedule.items():
        meeting = meeting_by_id[meeting_id]
        end = placement.start + meeting.duration
        teacher_loads[meeting.teacher_id] += meeting.duration
        room_loads[placement.room_id] += meeting.duration
        teacher_day_slots[(meeting.teacher_id, placement.day)].update(range(placement.start, end))
        for group_id in meeting.groups:
            by_group_day[(group_id, placement.day)].append((placement.start, end, meeting))
            group_loads[group_id][placement.day] += meeting.duration
        preferred = problem.teachers[meeting.teacher_id].preferred_slots
        if preferred:
            misses = sum(
                (placement.day, slot) not in preferred
                for slot in range(placement.start, end)
            )
            cost += weights["teacher_preferences"] * misses
        if model is not None:
            demand = model.predict_demand(meeting)
            attend = model.predict_attendance(meeting, placement.day, placement.start)
            best = model.best_attendance(meeting, len(problem.days), problem.slots_per_day)
            cost += (
                weights["attendance"]
                * demand
                / max(1.0, meeting.enrollment)
                * max(0.0, best - attend)
            )

    for (_group_id, _day), blocks in by_group_day.items():
        occupied = set()
        for start, end, _meeting in blocks:
            occupied.update(range(start, end))
        if occupied:
            cost += weights["student_gaps"] * (
                max(occupied) - min(occupied) + 1 - len(occupied)
            )
    for loads in group_loads.values():
        mean = sum(loads) / len(problem.days)
        cost += weights["daily_load"] * sum((load - mean) ** 2 for load in loads)

    for (group_id, day), blocks in by_group_day.items():
        ordered = sorted(blocks, key=lambda block: block[0])
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] == current[0] and (
                previous[2].heavy and current[2].heavy
                or previous[2].subject == current[2].subject
            ):
                cost += weights["heavy_adjacent"]

    for occupied in teacher_day_slots.values():
        if occupied:
            cost += weights["teacher_gaps"] * (
                max(occupied) - min(occupied) + 1 - len(occupied)
            )

    if room_loads:
        mean_room_load = sum(room_loads.values()) / len(problem.rooms)
        cost += weights["room_balance"] * sum(
            (room_loads.get(room_id, 0) - mean_room_load) ** 2
            for room_id in problem.rooms
        )

    if problem.teachers:
        mean_load = sum(teacher_loads.values()) / len(problem.teachers)
        cost += 0.02 * sum(
            (teacher_loads.get(teacher_id, 0) - mean_load) ** 2
            for teacher_id in problem.teachers
        )
    return cost


def objective_breakdown(
    problem: Problem,
    schedule: dict[str, Placement],
    model: PreferenceModel,
) -> dict[str, float]:
    """Return measured additive objective contributions under the active learned weights."""
    weight_names = tuple(model.preference_weights)
    contributions: dict[str, float] = {}
    balance_model = copy.copy(model)
    balance_model.preference_weights = {
        key: 0.0 for key in model.preference_weights
    }
    balance_cost = objective_cost(problem, schedule, balance_model)
    for selected_weight in weight_names:
        isolated_model = copy.copy(model)
        isolated_model.preference_weights = {
            key: value if key == selected_weight else 0.0
            for key, value in model.preference_weights.items()
        }
        contributions[selected_weight] = (
            objective_cost(problem, schedule, isolated_model) - balance_cost
        )
    contributions["teacher_load_balance"] = balance_cost
    return contributions


def soft_score(cost: float, meeting_count: int) -> float:
    return round(100.0 / (1.0 + cost / max(1, meeting_count)), 2)


def teacher_load_fairness(problem: Problem, schedule: dict[str, Placement]) -> float:
    loads = [0] * len(problem.teachers)
    teacher_index = {teacher_id: index for index, teacher_id in enumerate(problem.teachers)}
    for meeting in problem.meetings:
        placement = schedule.get(meeting.meeting_id)
        if placement is not None:
            loads[teacher_index[meeting.teacher_id]] += meeting.duration
    if not loads or max(loads, default=0) == 0:
        return 100.0
    mean = sum(loads) / len(loads)
    if mean == 0:
        return 100.0
    deviation = math.sqrt(sum((load - mean) ** 2 for load in loads) / len(loads))
    return round(max(0.0, 100.0 * (1.0 - deviation / mean)), 2)


def optimize(
    problem: Problem,
    initial: dict[str, Placement],
    method: str,
    model: PreferenceModel | None = None,
    iterations: int = 500,
    seed: int = 42,
) -> tuple[dict[str, Placement], float]:
    result = optimize_detailed(
        problem, initial, method, model, iterations=iterations, seed=seed
    )
    return result.schedule, result.best_cost


def optimize_detailed(
    problem: Problem,
    initial: dict[str, Placement],
    method: str,
    model: PreferenceModel | None = None,
    iterations: int = 500,
    seed: int = 42,
    max_runtime_seconds: float | None = 20.0,
    initial_temperature: float = 8.0,
    cooling_rate: float = 0.99,
    adaptive_neighborhood: bool = True,
    reheat_after: int = 80,
    early_stopping_rounds: int = 500,
    max_reheats: int = 2,
) -> OptimizationResult:
    method = method.lower()
    if method not in {"annealing", "tabu", "hill-climbing"}:
        raise ValueError(f"Unknown optimizer: {method}")
    if (
        initial_temperature <= 0
        or not 0 < cooling_rate < 1
        or reheat_after < 1
        or early_stopping_rounds < 1
        or max_reheats < 0
    ):
        raise ValueError("Simulated Annealing needs positive temperature and a cooling rate in (0, 1).")
    errors = validate_schedule(problem, initial)
    if errors:
        raise ValueError(
            "Optimizer requires a complete hard-feasible seed schedule: "
            + "; ".join(errors[:5])
        )
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    domains = meeting_domains(problem)
    rng = random.Random(seed)
    current = dict(initial)
    best = dict(current)
    current_cost = objective_cost(problem, current, model)
    best_cost = current_cost
    initial_cost = current_cost
    convergence: list[dict[str, float | int | bool]] = [
        {"iteration": 0, "current_cost": current_cost, "best_cost": best_cost, "temperature": 0.0, "accepted": True}
    ]
    accepted_moves = 0
    rejected_moves = 0
    reheats = 0
    rejection_reasons: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    state = Occupancy(problem)
    for meeting in problem.meetings:
        state.add(meeting, current[meeting.meeting_id])
    tabu_until: dict[tuple[str, Placement], int] = {}
    tabu_tenure = max(7, min(25, len(problem.meetings) // 3))
    iterations_run = 0
    stop_reason = "iteration budget reached"
    stagnant_steps = 0
    last_reheat_step = 0
    recent_improvements: list[bool] = []
    for step in range(max(0, iterations)):
        if max_runtime_seconds is not None and time.perf_counter() - started >= max_runtime_seconds:
            stop_reason = "runtime limit reached"
            break
        reheated = False
        if method == "annealing" and stagnant_steps >= reheat_after:
            if reheats < max_reheats:
                reheats += 1
                last_reheat_step = step
                stagnant_steps = 0
                reheated = True
            elif stagnant_steps >= early_stopping_rounds:
                stop_reason = "early stopping after annealing stagnation"
                break
        iterations_run += 1
        temperature_step = step - last_reheat_step if method == "annealing" else step
        temperature = max(0.025, initial_temperature * cooling_rate**temperature_step)
        meeting = rng.choice(problem.meetings)
        old = current[meeting.meeting_id]
        state.remove(meeting)
        improvement_rate = (
            sum(recent_improvements) / len(recent_improvements)
            if recent_improvements
            else 0.1
        )
        domain_sample_size = 75 if adaptive_neighborhood and improvement_rate < 0.08 else 45
        candidate_sample_size = 16 if adaptive_neighborhood and improvement_rate < 0.08 else 10
        sample = rng.sample(
            domains[meeting.meeting_id],
            min(domain_sample_size, len(domains[meeting.meeting_id])),
        )
        legal = [candidate for candidate in sample if state.can_place(meeting, candidate)]
        if not legal:
            rejected_moves += 1
            rejection_reasons["no hard-feasible neighbor in sampled domain"] += 1
            recent_improvements.append(False)
            recent_improvements = recent_improvements[-30:]
            state.add(meeting, old)
            convergence.append(
                {
                    "iteration": step + 1,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "temperature": temperature if method == "annealing" else 0.0,
                    "accepted": False,
                    "reheated": reheated,
                }
            )
            stagnant_steps += 1
            if method == "hill-climbing" and stagnant_steps >= 25:
                stop_reason = "sampled-neighborhood local optimum reached"
                break
            continue
        candidates = rng.sample(legal, min(candidate_sample_size, len(legal)))
        evaluated: list[tuple[Placement, float, bool]] = []
        for candidate in candidates:
            state.add(meeting, candidate)
            current[meeting.meeting_id] = candidate
            candidate_cost = objective_cost(problem, current, model)
            candidate_tabu = tabu_until.get((meeting.meeting_id, candidate), -1) > step
            evaluated.append((candidate, candidate_cost, candidate_tabu))
            state.remove(meeting)
        current[meeting.meeting_id] = old
        acceptable = [
            item for item in evaluated
            if not item[2] or item[1] < best_cost
        ]
        if method == "annealing":
            candidate, candidate_cost, _ = rng.choice(evaluated)
            delta = candidate_cost - current_cost
            accept = delta <= 0 or rng.random() < math.exp(-delta / temperature)
        elif method == "hill-climbing":
            candidate, candidate_cost, _ = min(evaluated, key=lambda item: item[1])
            accept = candidate_cost < current_cost
        else:
            if acceptable:
                candidate, candidate_cost, _ = min(acceptable, key=lambda item: item[1])
                accept = True
            else:
                candidate, candidate_cost, _ = min(evaluated, key=lambda item: item[1])
                accept = False

        improved_best = False
        if accept:
            state.add(meeting, candidate)
            current[meeting.meeting_id] = candidate
            current_cost = candidate_cost
            accepted_moves += 1
            if method == "tabu":
                tabu_until[(meeting.meeting_id, old)] = step + tabu_tenure
            if candidate_cost < best_cost:
                best, best_cost = dict(current), candidate_cost
                improved_best = True
        else:
            rejected_moves += 1
            rejection_reason = {
                "annealing": "Metropolis acceptance rejected move",
                "hill-climbing": "move did not improve current objective",
                "tabu": "move was tabu or did not beat the best objective",
            }[method]
            rejection_reasons[rejection_reason] += 1
            state.add(meeting, old)
            current[meeting.meeting_id] = old
        if step % 100 == 0:
            expired = [key for key, expiry in tabu_until.items() if expiry <= step]
            for key in expired:
                del tabu_until[key]
        convergence.append(
            {
                "iteration": step + 1,
                "current_cost": current_cost,
                "best_cost": best_cost,
                "temperature": temperature if method == "annealing" else 0.0,
                "accepted": accept,
                "reheated": reheated,
            }
        )
        recent_improvements.append(improved_best)
        recent_improvements = recent_improvements[-30:]
        stagnant_steps = 0 if improved_best else stagnant_steps + 1
        if method == "hill-climbing" and stagnant_steps >= 25:
            stop_reason = "sampled-neighborhood local optimum reached"
            break
    if iterations_run == iterations:
        stop_reason = "iteration budget reached"
    if max_runtime_seconds is not None and time.perf_counter() - started >= max_runtime_seconds:
        stop_reason = "runtime limit reached"
    return OptimizationResult(
        schedule=best,
        initial_cost=initial_cost,
        best_cost=best_cost,
        convergence=tuple(convergence),
        iterations=iterations_run,
        accepted_moves=accepted_moves,
        runtime_seconds=time.perf_counter() - started,
        method=method,
        stop_reason=stop_reason,
        rejected_moves=rejected_moves,
        reheats=reheats,
        rejection_reasons=tuple(sorted(rejection_reasons.items())),
    )
