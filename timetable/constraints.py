"""Hard-constraint checking and the MRV/degree/forward-checking CSP solver."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .models import Meeting, Placement, Problem


@dataclass(frozen=True)
class SolverStats:
    variables: int
    constraints: int
    nodes: int
    backtracks: int
    elapsed_seconds: float


def slot_mask(problem: Problem, placement: Placement, duration: int) -> int:
    offset = placement.day * problem.slots_per_day + placement.start
    return ((1 << duration) - 1) << offset


def meeting_domains(
    problem: Problem,
    blocked_teachers: Optional[dict[str, set[tuple[int, int]]]] = None,
    closed_rooms: Optional[set[str]] = None,
) -> dict[str, tuple[Placement, ...]]:
    blocked_teachers = blocked_teachers or {}
    closed_rooms = set(problem.closed_rooms) | (closed_rooms or set())
    domains: dict[str, tuple[Placement, ...]] = {}
    for meeting in problem.meetings:
        teacher = problem.teachers[meeting.teacher_id]
        candidates: list[Placement] = []
        for day in range(len(problem.days)):
            for start in range(problem.slots_per_day - meeting.duration + 1):
                covered = {(day, slot) for slot in range(start, start + meeting.duration)}
                if teacher.available_slots is not None and not covered.issubset(teacher.available_slots):
                    continue
                if covered.intersection(blocked_teachers.get(meeting.teacher_id, set())):
                    continue
                for room in problem.rooms.values():
                    if room.room_id in closed_rooms:
                        continue
                    if (
                        room.capacity < meeting.enrollment
                        or room.room_type.lower() != meeting.room_type.lower()
                        or not meeting.required_equipment.issubset(room.equipment)
                        or meeting.required_equipment.intersection(
                            problem.unavailable_equipment_by_room.get(room.room_id, frozenset())
                        )
                    ):
                        continue
                    candidates.append(Placement(day, start, room.room_id))
        domains[meeting.meeting_id] = tuple(candidates)
    return domains


class Occupancy:
    """Incremental occupancy bitsets used by both search and constraint checks."""

    def __init__(self, problem: Problem):
        self.problem = problem
        self.events = {meeting.meeting_id: meeting for meeting in problem.meetings}
        self.teacher_masks: dict[str, int] = defaultdict(int)
        self.room_masks: dict[str, int] = defaultdict(int)
        self.group_masks: dict[str, int] = defaultdict(int)
        self.teacher_hours: dict[tuple[str, int], int] = defaultdict(int)
        self.schedule: dict[str, Placement] = {}

    def can_place(self, meeting: Meeting, placement: Placement) -> bool:
        if meeting.room_type.lower() == "lab" and meeting.duration != 2:
            return False
        if placement.day < 0 or placement.day >= len(self.problem.days):
            return False
        if placement.start < 0 or placement.start + meeting.duration > self.problem.slots_per_day:
            return False
        room = self.problem.rooms.get(placement.room_id)
        if room is None or room.capacity < meeting.enrollment:
            return False
        if room.room_type.lower() != meeting.room_type.lower():
            return False
        if not meeting.required_equipment.issubset(room.equipment):
            return False
        if meeting.required_equipment.intersection(
            self.problem.unavailable_equipment_by_room.get(room.room_id, frozenset())
        ):
            return False
        if room.room_id in self.problem.closed_rooms:
            return False
        mask = slot_mask(self.problem, placement, meeting.duration)
        if self.teacher_masks[meeting.teacher_id] & mask:
            return False
        if self.room_masks[placement.room_id] & mask:
            return False
        if any(self.group_masks[group_id] & mask for group_id in meeting.groups):
            return False
        teacher = self.problem.teachers[meeting.teacher_id]
        if self.teacher_hours[(meeting.teacher_id, placement.day)] + meeting.duration > teacher.max_hours_per_day:
            return False
        return True

    def add(self, meeting: Meeting, placement: Placement) -> None:
        mask = slot_mask(self.problem, placement, meeting.duration)
        self.teacher_masks[meeting.teacher_id] |= mask
        self.room_masks[placement.room_id] |= mask
        for group_id in meeting.groups:
            self.group_masks[group_id] |= mask
        self.teacher_hours[(meeting.teacher_id, placement.day)] += meeting.duration
        self.schedule[meeting.meeting_id] = placement

    def remove(self, meeting: Meeting) -> Placement:
        placement = self.schedule.pop(meeting.meeting_id)
        mask = slot_mask(self.problem, placement, meeting.duration)
        self.teacher_masks[meeting.teacher_id] &= ~mask
        self.room_masks[placement.room_id] &= ~mask
        for group_id in meeting.groups:
            self.group_masks[group_id] &= ~mask
        self.teacher_hours[(meeting.teacher_id, placement.day)] -= meeting.duration
        return placement


def validate_schedule(problem: Problem, schedule: dict[str, Placement]) -> list[str]:
    errors: list[str] = []
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    unknown = sorted(set(schedule) - set(meeting_by_id))
    errors.extend(f"Unknown meeting in schedule: {meeting_id}" for meeting_id in unknown)
    missing = sorted(set(meeting_by_id) - set(schedule))
    errors.extend(f"Meeting is not scheduled: {meeting_id}" for meeting_id in missing)
    state = Occupancy(problem)
    for meeting_id, placement in schedule.items():
        meeting = meeting_by_id.get(meeting_id)
        if meeting is None:
            continue
        teacher = problem.teachers[meeting.teacher_id]
        if teacher.available_slots is not None:
            covered = {(placement.day, slot) for slot in range(placement.start, placement.start + meeting.duration)}
            if not covered.issubset(teacher.available_slots):
                errors.append(f"{meeting_id}: teacher is unavailable at the assigned time.")
        if not state.can_place(meeting, placement):
            errors.append(f"{meeting_id}: teacher, room, group, capacity, type, or daily-hours conflict.")
        else:
            state.add(meeting, placement)
    return errors


def solve_csp(
    problem: Problem,
    max_nodes: int = 100_000,
    seed: int = 42,
    timeout_seconds: float | None = 60.0,
    return_stats: bool = False,
) -> tuple[dict[str, Placement], int] | tuple[dict[str, Placement], SolverStats]:
    """Solve the hard constraints with dynamic MRV, degree tie-break, and forward checking."""
    started = time.perf_counter()
    domains = meeting_domains(problem)
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    empty = [meeting_id for meeting_id, values in domains.items() if not values]
    if empty:
        diagnostics = []
        for meeting_id in empty[:8]:
            meeting = meeting_by_id[meeting_id]
            compatible_rooms = [
                room
                for room in problem.rooms.values()
                if room.room_type.lower() == meeting.room_type.lower()
                and room.capacity >= meeting.enrollment
                and meeting.required_equipment.issubset(room.equipment)
            ]
            teacher = problem.teachers[meeting.teacher_id]
            reason = (
                "no room matches type/capacity/required equipment"
                if not compatible_rooms
                else "teacher availability leaves no consecutive slot window"
            )
            diagnostics.append(f"{meeting_id}: {reason}")
        raise ValueError(
            "CSP found meetings with empty feasible domains: "
            + "; ".join(diagnostics)
            + ". Check room inventory, enrollment, equipment, and teacher availability."
        )

    resource_index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for meeting in problem.meetings:
        resource_index[("teacher", meeting.teacher_id)].add(meeting.meeting_id)
        for group_id in meeting.groups:
            resource_index[("group", group_id)].add(meeting.meeting_id)
        for room_id in {placement.room_id for placement in domains[meeting.meeting_id]}:
            resource_index[("room", room_id)].add(meeting.meeting_id)
    neighbors: dict[str, set[str]] = {meeting.meeting_id: set() for meeting in problem.meetings}
    for ids in resource_index.values():
        for meeting_id in ids:
            neighbors[meeting_id].update(ids - {meeting_id})

    state = Occupancy(problem)
    rng = random.Random(seed)
    nodes = 0
    backtracks = 0
    timed_out = False

    def preference_cost(meeting: Meeting, placement: Placement) -> int:
        preferred = problem.teachers[meeting.teacher_id].preferred_slots
        if not preferred:
            return 0
        return sum(
            (placement.day, slot) not in preferred
            for slot in range(placement.start, placement.start + meeting.duration)
        )

    current_domains = {
        meeting_id: list(values) for meeting_id, values in domains.items()
    }

    def search(unassigned: set[str]) -> bool:
        nonlocal nodes, backtracks, timed_out
        if not unassigned:
            return True
        if nodes >= max_nodes:
            return False
        if timeout_seconds is not None and time.perf_counter() - started >= timeout_seconds:
            timed_out = True
            return False

        chosen_id = min(
            unassigned,
            key=lambda item: (
                len(current_domains[item]),
                -len(neighbors[item] & unassigned),
                item,
            ),
        )
        chosen_values = list(current_domains[chosen_id])
        if not chosen_values:
            return False
        meeting = meeting_by_id[chosen_id]
        rng.shuffle(chosen_values)
        chosen_values.sort(key=lambda candidate: preference_cost(meeting, candidate))
        rest = unassigned - {chosen_id}
        impacted = neighbors[chosen_id] & rest
        for candidate in chosen_values:
            nodes += 1
            state.add(meeting, candidate)
            viable = True
            previous_domains: dict[str, list[Placement]] = {}
            for neighbor_id in impacted:
                previous = current_domains[neighbor_id]
                filtered = [
                    value
                    for value in previous
                    if state.can_place(meeting_by_id[neighbor_id], value)
                ]
                if len(filtered) != len(previous):
                    previous_domains[neighbor_id] = previous
                    current_domains[neighbor_id] = filtered
                if not filtered:
                    viable = False
                    break
            if viable and search(rest):
                return True
            state.remove(meeting)
            current_domains.update(previous_domains)
            backtracks += 1
            if nodes >= max_nodes or timed_out:
                break
        return False

    if not search(set(meeting_by_id)):
        if timed_out:
            raise RuntimeError(
                f"CSP search timed out after {time.perf_counter() - started:.1f}s "
                f"and {nodes:,} nodes; no complete timetable was found."
            )
        raise RuntimeError(
            f"CSP did not find a feasible timetable within {nodes:,} search nodes. "
            "Check the data/capacity or increase the CSP node limit."
        )
    variables = len(problem.meetings)
    stats = SolverStats(
        variables=variables,
        constraints=sum(len(values) for values in neighbors.values()) // 2
        + sum(
            1
            + bool(meeting.required_equipment)
            + bool(problem.teachers[meeting.teacher_id].available_slots)
            + (meeting.duration == 2)
            for meeting in problem.meetings
        ),
        nodes=nodes,
        backtracks=backtracks,
        elapsed_seconds=time.perf_counter() - started,
    )
    if return_stats:
        return dict(state.schedule), stats
    return dict(state.schedule), nodes
