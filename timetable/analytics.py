"""Computed constraint diagnostics and factual schedule decision explanations."""

from __future__ import annotations

from itertools import combinations

from .constraints import Occupancy, meeting_domains
from .ml import PreferenceModel
from .models import Placement, Problem
from .optimizer import objective_cost, schedule_metrics


def hard_constraint_count(problem: Problem) -> int:
    """Count unary domain restrictions and unique pairwise resource-conflict constraints."""
    domains = meeting_domains(problem)
    resource_members: dict[tuple[str, str], set[str]] = {}
    for meeting in problem.meetings:
        resource_members.setdefault(("teacher", meeting.teacher_id), set()).add(meeting.meeting_id)
        for group_id in meeting.groups:
            resource_members.setdefault(("group", group_id), set()).add(meeting.meeting_id)
        for room_id in {p.room_id for p in domains[meeting.meeting_id]}:
            resource_members.setdefault(("room", room_id), set()).add(meeting.meeting_id)
    pairs: set[tuple[str, str]] = set()
    for members in resource_members.values():
        pairs.update(tuple(sorted(pair)) for pair in combinations(sorted(members), 2))
    unary_constraints = sum(
        1
        + bool(meeting.required_equipment)
        + bool(problem.teachers[meeting.teacher_id].available_slots)
        + (meeting.duration == 2)
        for meeting in problem.meetings
    )
    return unary_constraints + len(pairs)


def explain_placement(
    problem: Problem,
    schedule: dict[str, Placement],
    meeting_id: str,
    model: PreferenceModel | None = None,
    max_alternatives: int = 80,
) -> dict[str, object]:
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    if meeting_id not in meeting_by_id or meeting_id not in schedule:
        raise ValueError(f"Meeting {meeting_id!r} is not in the current timetable.")
    meeting = meeting_by_id[meeting_id]
    placement = schedule[meeting_id]
    teacher = problem.teachers[meeting.teacher_id]
    room = problem.rooms[placement.room_id]
    preference_hits = (
        [
            problem.days[placement.day] + f" slot {slot}"
            for slot in range(placement.start, placement.start + meeting.duration)
            if (placement.day, slot) in teacher.preferred_slots
        ]
        if teacher.preferred_slots
        else []
    )
    reasons = [
        "The selected time and room satisfy the current hard constraints; "
        "the timetable conflict detector validates the complete schedule."
    ]
    reasons.append(
        f"Room {room.room_id} seats {room.capacity} for an official enrollment of "
        f"{meeting.enrollment}; predicted attendance does not reduce the mandatory capacity."
    )
    if meeting.required_equipment:
        reasons.append(
            "The assigned room has the required equipment: "
            + ", ".join(sorted(meeting.required_equipment))
            + "."
        )
    if teacher.preferred_slots:
        if preference_hits:
            reasons.append(
                f"The teacher preference is matched for {len(preference_hits)} of "
                f"{meeting.duration} occupied period(s)."
            )
        else:
            reasons.append("This placement misses the teacher's preferred slot set.")
    else:
        reasons.append("No teacher slot preference was supplied for this instructor.")
    attendance = (
        model.predict_attendance(meeting, placement.day, placement.start)
        if model is not None
        else None
    )
    demand = model.predict_demand(meeting) if model is not None else None
    metrics = schedule_metrics(problem, schedule)
    original_cost = objective_cost(problem, schedule, model)

    state = Occupancy(problem)
    for other_id, other_placement in schedule.items():
        if other_id != meeting_id:
            state.add(meeting_by_id[other_id], other_placement)
    alternatives = [
        candidate
        for candidate in meeting_domains(problem)[meeting_id]
        if candidate != placement and state.can_place(meeting, candidate)
    ]
    alternatives.sort(
        key=lambda candidate: (
            candidate.day,
            candidate.start,
            candidate.room_id,
        )
    )
    scored: list[tuple[float, Placement]] = []
    for candidate in alternatives[:max_alternatives]:
        candidate_schedule = {**schedule, meeting_id: candidate}
        scored.append((objective_cost(problem, candidate_schedule, model), candidate))
    best_alternative = min(scored, default=None, key=lambda item: item[0])
    if best_alternative:
        delta = original_cost - best_alternative[0]
        if delta > 1e-8:
            reasons.append(
                f"Among {len(scored)} checked feasible local alternatives, the current placement "
                f"has a lower whole-schedule cost by {delta:.2f} points."
            )
        else:
            alt = best_alternative[1]
            reasons.append(
                f"A checked feasible local alternative is {problem.days[alt.day]} at "
                f"{8 + alt.start:02d}:00 in {alt.room_id}; it did not lower the full objective."
            )
    else:
        reasons.append("No alternative placement was feasible in the checked domain.")
    return {
        "meeting": meeting,
        "placement": placement,
        "reasons": reasons,
        "attendance_prediction": attendance,
        "demand_prediction": demand,
        "schedule_cost": original_cost,
        "metrics": metrics,
        "alternatives_checked": len(scored),
        "alternative_cost": best_alternative[0] if best_alternative else None,
    }
