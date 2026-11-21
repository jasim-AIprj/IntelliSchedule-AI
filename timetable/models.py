"""Typed domain models for timetable generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")
SLOTS_PER_DAY = 8
SLOT_LABELS = tuple(
    f"{hour:02d}:00-{hour + 1:02d}:00" for hour in range(8, 16)
)


@dataclass(frozen=True)
class Teacher:
    teacher_id: str
    name: str
    max_hours_per_day: int = 6
    available_slots: Optional[frozenset[tuple[int, int]]] = None
    preferred_slots: frozenset[tuple[int, int]] = frozenset()


@dataclass(frozen=True)
class Room:
    room_id: str
    capacity: int
    room_type: str
    equipment: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    course_id: str
    course_name: str
    subject: str
    teacher_id: str
    groups: tuple[str, ...]
    enrollment: int
    room_type: str
    duration: int
    session_index: int
    sessions_per_week: int = 1
    heavy: bool = False
    department: str = "General"
    required_equipment: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Placement:
    day: int
    start: int
    room_id: str


@dataclass
class Problem:
    teachers: dict[str, Teacher]
    rooms: dict[str, Room]
    meetings: tuple[Meeting, ...]
    slots_per_day: int = SLOTS_PER_DAY
    days: tuple[str, ...] = DAYS
    metadata: dict[str, str] = field(default_factory=dict)
    closed_rooms: frozenset[str] = frozenset()
    unavailable_equipment_by_room: dict[str, frozenset[str]] = field(default_factory=dict)
