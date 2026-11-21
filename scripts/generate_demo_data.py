"""Generate the deterministic synthetic 100+ section demonstration dataset."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "demo_data"


def write_csv(name: str, headers: list[str], rows: list[dict[str, object]]) -> None:
    with (ROOT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    groups = [
        {"group_id": f"G{i:02d}", "name": f"Year {((i - 1) // 2) + 1} Section {(i - 1) % 2 + 1}", "size": 30 + (i * 7) % 16}
        for i in range(1, 11)
    ]
    teachers = [
        {
            "teacher_id": f"T{i:02d}",
            "name": f"Faculty {i:02d}",
            "max_hours_per_day": 6,
            "available_slots": "*",
            "preferred_slots": f"Mon:{(i % 5) + 1}|Wed:{(i % 5) + 1}|Thu:{(i % 5) + 1}",
        }
        for i in range(1, 36)
    ]
    rooms = [
        *[
            {"room_id": f"L{i:02d}", "capacity": 60 + (i % 3) * 20, "room_type": "lecture"}
            for i in range(1, 13)
        ],
        *[
            {
                "room_id": f"B{i:02d}",
                "capacity": 50 + (i % 2) * 20,
                "room_type": "lab",
                "equipment": "computing|electronics",
            }
            for i in range(1, 5)
        ],
    ]
    courses = []
    subjects = ("Algorithms", "Statistics", "Networks", "Databases", "Systems", "Design", "AI", "Security", "Math", "Writing")
    for index in range(1, 111):
        group_number = (index - 1) % 10 + 1
        group_id = f"G{group_number:02d}"
        is_elective = index in (29, 59, 89, 109)
        groups_for_course = group_id
        if is_elective:
            second = group_number % 10 + 1
            groups_for_course = f"{group_id}|G{second:02d}"
        is_lab = index % 9 == 0
        subject = subjects[(index - 1) % len(subjects)]
        courses.append(
            {
                "course_id": f"C{index:03d}",
                "course_name": f"{subject} {((index - 1) // 10) + 1}.{(index - 1) % 10 + 1}",
                "subject": subject,
                "department": ("Computing", "Engineering", "Business", "Science", "Arts")[(index - 1) % 5],
                "teacher_id": f"T{(index - 1) % 35 + 1:02d}",
                "group_ids": groups_for_course,
                "enrollment": 30 + (group_number * 7) % 16,
                "room_type": "lab" if is_lab else "lecture",
                "duration": 2 if is_lab else 1,
                "sessions_per_week": 2,
                "heavy": "true" if index % 4 == 0 else "false",
                "required_equipment": "computing" if is_lab else "",
            }
        )
    write_csv("groups.csv", ["group_id", "name", "size"], groups)
    write_csv("teachers.csv", ["teacher_id", "name", "max_hours_per_day", "available_slots", "preferred_slots"], teachers)
    write_csv("rooms.csv", ["room_id", "capacity", "room_type", "equipment"], rooms)
    write_csv(
        "courses.csv",
        ["course_id", "course_name", "subject", "department", "teacher_id", "group_ids", "enrollment", "room_type", "duration", "sessions_per_week", "heavy", "required_equipment"],
        courses,
    )
    quick_root = ROOT / "quick"
    quick_root.mkdir(exist_ok=True)
    write_quick_data(quick_root)
    print(f"Generated {len(courses)} sections, {len(teachers)} teachers, {len(groups)} groups, and {len(rooms)} rooms in {ROOT}.")


def write_quick_data(root: Path) -> None:
    quick_courses = [
        {"course_id": "Q101", "course_name": "Intro to Computing", "subject": "Computing", "department": "Computing", "teacher_id": "QT1", "group_ids": "QG1", "enrollment": 28, "room_type": "lecture", "duration": 1, "sessions_per_week": 2, "heavy": "false", "required_equipment": ""},
        {"course_id": "Q102", "course_name": "Programming Lab", "subject": "Programming", "department": "Computing", "teacher_id": "QT2", "group_ids": "QG1", "enrollment": 28, "room_type": "lab", "duration": 2, "sessions_per_week": 1, "heavy": "false", "required_equipment": "computing"},
        {"course_id": "Q103", "course_name": "Applied Statistics", "subject": "Statistics", "department": "Science", "teacher_id": "QT3", "group_ids": "QG1|QG2", "enrollment": 48, "room_type": "lecture", "duration": 1, "sessions_per_week": 2, "heavy": "true", "required_equipment": ""},
        {"course_id": "Q104", "course_name": "Engineering Design", "subject": "Design", "department": "Engineering", "teacher_id": "QT4", "group_ids": "QG2", "enrollment": 24, "room_type": "lecture", "duration": 1, "sessions_per_week": 2, "heavy": "false", "required_equipment": ""},
        {"course_id": "Q105", "course_name": "Electronics Lab", "subject": "Electronics", "department": "Engineering", "teacher_id": "QT2", "group_ids": "QG2", "enrollment": 24, "room_type": "lab", "duration": 2, "sessions_per_week": 1, "heavy": "false", "required_equipment": "electronics"},
        {"course_id": "Q106", "course_name": "Academic Writing", "subject": "Writing", "department": "Arts", "teacher_id": "QT5", "group_ids": "QG1|QG2", "enrollment": 48, "room_type": "lecture", "duration": 1, "sessions_per_week": 1, "heavy": "false", "required_equipment": ""},
        {"course_id": "Q107", "course_name": "Data Systems", "subject": "Databases", "department": "Computing", "teacher_id": "QT1", "group_ids": "QG2", "enrollment": 24, "room_type": "lecture", "duration": 1, "sessions_per_week": 2, "heavy": "true", "required_equipment": ""},
        {"course_id": "Q108", "course_name": "Digital Logic", "subject": "Systems", "department": "Engineering", "teacher_id": "QT4", "group_ids": "QG1", "enrollment": 28, "room_type": "lecture", "duration": 1, "sessions_per_week": 2, "heavy": "false", "required_equipment": ""},
    ]
    quick_teachers = [
        {"teacher_id": f"QT{i}", "name": f"Quick Demo Faculty {i}", "max_hours_per_day": 6, "available_slots": "*", "preferred_slots": f"Mon:{i % 4 + 1}|Wed:{i % 4 + 1}"}
        for i in range(1, 6)
    ]
    quick_groups = [
        {"group_id": "QG1", "name": "Computing Section A", "size": 28},
        {"group_id": "QG2", "name": "Engineering Section B", "size": 24},
    ]
    quick_rooms = [
        {"room_id": "QL1", "capacity": 60, "room_type": "lecture", "equipment": ""},
        {"room_id": "QL2", "capacity": 40, "room_type": "lecture", "equipment": ""},
        {"room_id": "QBL1", "capacity": 36, "room_type": "lab", "equipment": "computing|electronics"},
    ]
    write_csv_to(root, "courses.csv", list(quick_courses[0]), quick_courses)
    write_csv_to(root, "teachers.csv", list(quick_teachers[0]), quick_teachers)
    write_csv_to(root, "groups.csv", list(quick_groups[0]), quick_groups)
    write_csv_to(root, "rooms.csv", list(quick_rooms[0]), quick_rooms)


def write_csv_to(root: Path, name: str, headers: list[str], rows: list[dict[str, object]]) -> None:
    with (root / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
