"""Means-Ends Analysis and CSP-guided repair for timetable disruptions."""

from __future__ import annotations

from dataclasses import replace

from .constraints import Occupancy, meeting_domains, validate_schedule
from .ml import PreferenceModel
from .models import Meeting, Placement, Problem
from .optimizer import objective_cost


def disruption_cost(
    original: dict[str, Placement],
    repaired: dict[str, Placement],
    problem: Problem,
) -> float:
    """Weighted changes from the original schedule; a new session costs one move."""
    meetings = {meeting.meeting_id: meeting for meeting in problem.meetings}
    cost = 0.0
    for meeting_id, placement in repaired.items():
        old = original.get(meeting_id)
        if old is None:
            cost += 1.0
            continue
        if placement == old:
            continue
        meeting = meetings[meeting_id]
        cost += 1.0
        if placement.day != old.day:
            cost += 0.5
        if placement.start != old.start:
            cost += 0.25 * abs(placement.start - old.start)
        if placement.room_id != old.room_id:
            cost += 0.25
        if meeting.duration == 2:
            cost += 0.25
    return cost


def repair_schedule(
    problem: Problem,
    schedule: dict[str, Placement],
    *,
    absent_teacher: str | None = None,
    absent_day: int | None = None,
    unavailable_room: str | None = None,
    unavailable_equipment: str | None = None,
    equipment_room: str | None = None,
    teacher_unavailable_slots: dict[str, set[tuple[int, int]]] | None = None,
    enrollment_updates: dict[str, int] | None = None,
    added_meeting: Meeting | None = None,
    model: PreferenceModel | None = None,
) -> tuple[dict[str, Placement], list[str]]:
    """Use differences from a desired state to guide a low-disruption CSP repair."""
    teacher_unavailable_slots = teacher_unavailable_slots or {}
    enrollment_updates = enrollment_updates or {}
    initial_errors = validate_schedule(problem, schedule)
    if initial_errors:
        raise ValueError(
            "Cannot repair an invalid or incomplete timetable: "
            + "; ".join(initial_errors[:5])
        )
    if absent_teacher and absent_teacher not in problem.teachers:
        raise ValueError(f"Unknown teacher: {absent_teacher}")
    if absent_teacher and absent_day is None:
        raise ValueError("Choose an absence day for the teacher.")
    if absent_day is not None and not 0 <= absent_day < len(problem.days):
        raise ValueError("Absence day is outside the configured timetable.")
    if unavailable_room and unavailable_room not in problem.rooms:
        raise ValueError(f"Unknown room: {unavailable_room}")
    if equipment_room and equipment_room not in problem.rooms:
        raise ValueError(f"Unknown room for equipment outage: {equipment_room}")
    if unavailable_equipment and not equipment_room:
        raise ValueError("Choose the lab room where the equipment is unavailable.")
    if unavailable_equipment and unavailable_equipment.lower() not in problem.rooms[equipment_room].equipment:
        raise ValueError(
            f"Room {equipment_room} does not list equipment {unavailable_equipment!r}."
        )
    if any(teacher_id not in problem.teachers for teacher_id in teacher_unavailable_slots):
        unknown = sorted(set(teacher_unavailable_slots) - set(problem.teachers))
        raise ValueError(f"Unknown teacher(s) in availability changes: {', '.join(unknown)}.")
    if added_meeting and added_meeting.meeting_id in {m.meeting_id for m in problem.meetings}:
        raise ValueError(f"Meeting {added_meeting.meeting_id} already exists.")
    meeting_ids = {meeting.meeting_id for meeting in problem.meetings}
    unknown_enrollment_ids = set(enrollment_updates) - meeting_ids
    if unknown_enrollment_ids:
        raise ValueError(
            "Unknown meeting(s) for enrollment update: "
            + ", ".join(sorted(unknown_enrollment_ids))
            + "."
        )
    if any(enrollment < 1 for enrollment in enrollment_updates.values()):
        raise ValueError("Updated official enrollment must be positive.")
    blocked = {
        teacher_id: set(slots)
        for teacher_id, slots in teacher_unavailable_slots.items()
    }
    if absent_teacher and absent_day is not None:
        blocked.setdefault(absent_teacher, set()).update(
            (absent_day, slot) for slot in range(problem.slots_per_day)
        )
    updated_teachers = dict(problem.teachers)
    all_slots = {
        (day, slot)
        for day in range(len(problem.days))
        for slot in range(problem.slots_per_day)
    }
    for teacher_id, unavailable in blocked.items():
        teacher = updated_teachers[teacher_id]
        available = set(teacher.available_slots) if teacher.available_slots is not None else set(all_slots)
        available.difference_update(unavailable)
        updated_teachers[teacher_id] = replace(
            teacher,
            available_slots=frozenset(available),
        )
    closed_rooms = set(problem.closed_rooms)
    if unavailable_room:
        closed_rooms.add(unavailable_room)
    equipment_outages = {
        room_id: set(tags)
        for room_id, tags in problem.unavailable_equipment_by_room.items()
    }
    if unavailable_equipment and equipment_room:
        equipment_outages.setdefault(equipment_room, set()).add(unavailable_equipment.lower())
    active_problem = replace(
        problem,
        teachers=updated_teachers,
        closed_rooms=frozenset(closed_rooms),
        unavailable_equipment_by_room={
            room_id: frozenset(tags) for room_id, tags in equipment_outages.items()
        },
        meetings=tuple(
            replace(meeting, enrollment=enrollment_updates.get(meeting.meeting_id, meeting.enrollment))
            for meeting in problem.meetings
        ) + ((added_meeting,) if added_meeting else ()),
    )
    if added_meeting:
        if added_meeting.teacher_id not in active_problem.teachers:
            raise ValueError(f"New meeting references unknown teacher {added_meeting.teacher_id}.")
        if not set(added_meeting.groups).issubset(
            {group for meeting in problem.meetings for group in meeting.groups}
        ):
            raise ValueError("New meeting references a student group not present in the timetable.")

    meeting_by_id = {meeting.meeting_id: meeting for meeting in active_problem.meetings}
    affected_ids: set[str] = set()
    for meeting_id, placement in schedule.items():
        meeting = meeting_by_id[meeting_id]
        occupied = {(placement.day, slot) for slot in range(placement.start, placement.start + meeting.duration)}
        if (
            unavailable_room == placement.room_id
            or unavailable_equipment
            and equipment_room == placement.room_id
            and unavailable_equipment.lower() in meeting.required_equipment
            or occupied.intersection(blocked.get(meeting.teacher_id, set()))
        ):
            affected_ids.add(meeting_id)
    if added_meeting:
        affected_ids.add(added_meeting.meeting_id)
    affected_ids.update(enrollment_updates)
    if not affected_ids:
        return dict(schedule), []

    domains = meeting_domains(active_problem, blocked, closed_rooms)
    # Build a resource graph so a blocked alternate slot can cascade to its direct blocker.
    resource_members: dict[tuple[str, str], set[str]] = {}
    for meeting in active_problem.meetings:
        resource_members.setdefault(("teacher", meeting.teacher_id), set()).add(meeting.meeting_id)
        for group_id in meeting.groups:
            resource_members.setdefault(("group", group_id), set()).add(meeting.meeting_id)
        for room_id in {candidate.room_id for candidate in domains[meeting.meeting_id]}:
            resource_members.setdefault(("room", room_id), set()).add(meeting.meeting_id)
    neighbors: dict[str, set[str]] = {meeting.meeting_id: set() for meeting in active_problem.meetings}
    for members in resource_members.values():
        for meeting_id in members:
            neighbors[meeting_id].update(members - {meeting_id})
    distance = {meeting_id: 0 for meeting_id in affected_ids}
    frontier = list(affected_ids)
    while frontier:
        current_id = frontier.pop(0)
        for neighbor_id in neighbors[current_id]:
            if neighbor_id not in distance:
                distance[neighbor_id] = distance[current_id] + 1
                frontier.append(neighbor_id)
    repair_candidates = sorted(
        (meeting_by_id[meeting_id] for meeting_id in distance if meeting_id not in affected_ids),
        key=lambda meeting: (
            distance[meeting.meeting_id],
            len(domains[meeting.meeting_id]),
            -len(meeting.groups),
            meeting.meeting_id,
        ),
    )
    node_limit = 25_000
    nodes = 0
    result: dict[str, Placement] | None = None

    def search(
        index: int,
        repair_meetings: list[Meeting],
        state: Occupancy,
        fixed: dict[str, Placement],
        placed: dict[str, Placement],
    ) -> bool:
        nonlocal nodes
        if index == len(repair_meetings):
            return True
        meeting = repair_meetings[index]
        old = schedule.get(meeting.meeting_id)
        valid = [
            candidate
            for candidate in domains[meeting.meeting_id]
            if state.can_place(meeting, candidate)
        ]
        valid.sort(
            key=lambda candidate: (
                old is not None and candidate != old,
                old is not None and candidate.day != old.day,
                abs(candidate.start - old.start) if old else candidate.day * problem.slots_per_day + candidate.start,
                old is not None and candidate.room_id != old.room_id,
                objective_cost(
                    active_problem,
                    {**fixed, **placed, meeting.meeting_id: candidate},
                    model,
                ),
            )
        )
        for candidate in valid:
            nodes += 1
            if nodes > node_limit:
                return False
            state.add(meeting, candidate)
            placed[meeting.meeting_id] = candidate
            if search(index + 1, repair_meetings, state, fixed, placed):
                return True
            del placed[meeting.meeting_id]
            state.remove(meeting)
        return False

    max_cascade = min(8, len(repair_candidates))
    for cascade_count in range(max_cascade + 1):
        repair_ids = affected_ids | {
            meeting.meeting_id for meeting in repair_candidates[:cascade_count]
        }
        repair_meetings = [meeting_by_id[meeting_id] for meeting_id in repair_ids]
        repair_meetings.sort(
            key=lambda meeting: (
                len(domains[meeting.meeting_id]),
                -len(meeting.groups),
                meeting.meeting_id,
            )
        )
        fixed = {
            meeting_id: placement
            for meeting_id, placement in schedule.items()
            if meeting_id not in repair_ids
        }
        state = Occupancy(active_problem)
        for meeting_id, placement in fixed.items():
            meeting = meeting_by_id[meeting_id]
            if not state.can_place(meeting, placement):
                raise ValueError(f"The existing timetable has a hard conflict at {meeting_id}; repair stopped.")
            state.add(meeting, placement)
        placed: dict[str, Placement] = {}
        if search(0, repair_meetings, state, fixed, placed):
            result = {**fixed, **placed}
            break
        if nodes >= node_limit:
            break

    if result is None:
        raise RuntimeError(
            f"Means-Ends repair could not place all affected meetings within {node_limit:,} "
            f"search nodes and an {max_cascade}-meeting cascade. Add capacity or adjust "
            "the disruption constraints and retry."
        )
    errors = validate_schedule(active_problem, result)
    if errors:
        raise RuntimeError("Repair failed hard-constraint validation: " + "; ".join(errors[:5]))
    changed = [
        meeting_id
        for meeting_id, placement in result.items()
        if schedule.get(meeting_id) != placement
    ]
    return result, changed
