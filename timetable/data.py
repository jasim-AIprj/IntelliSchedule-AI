"""CSV/Excel dataset loading and schedule exports."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable

from .models import Meeting, Problem, Room, Teacher

REQUIRED_TABLES = ("courses", "teachers", "rooms", "groups")


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if file_path.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise RuntimeError("Excel support requires openpyxl.") from exc
        workbook = load_workbook(file_path, read_only=True, data_only=True)
        sheet = workbook.active
        values = sheet.iter_rows(values_only=True)
        headers = next(values, None)
        if not headers:
            return []
        names = [str(value).strip() if value is not None else "" for value in headers]
        return [
            {name: value for name, value in zip(names, row)}
            for row in values
            if any(value is not None for value in row)
        ]
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _slot_set(value: Any, days: tuple[str, ...]) -> frozenset[tuple[int, int]]:
    if value is None or not str(value).strip() or str(value).strip() == "*":
        return frozenset()
    day_lookup = {day.lower(): index for index, day in enumerate(days)}
    result: set[tuple[int, int]] = set()
    for token in str(value).replace(",", "|").split("|"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2 or parts[0].strip().lower() not in day_lookup:
            raise ValueError(f"Invalid day/slot token {token!r}; use Mon:0|Tue:3.")
        day_index = day_lookup[parts[0].strip().lower()]
        try:
            slot_index = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid slot index in {token!r}.") from exc
        result.add((day_index, slot_index))
    return frozenset(result)


def _tag_set(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    return frozenset(
        part.strip().lower()
        for part in str(value).replace(",", "|").split("|")
        if part.strip()
    )


def validate_tables(tables: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """Return actionable row-level warnings before strict problem construction."""
    issues: dict[str, list[str]] = {}
    required_fields = {
        "courses": ("course_id", "teacher_id"),
        "teachers": ("teacher_id",),
        "rooms": ("room_id", "room_type", "capacity"),
        "groups": ("group_id",),
    }
    for table_name in REQUIRED_TABLES:
        rows = tables.get(table_name)
        if rows is None:
            issues.setdefault("missing_tables", []).append(table_name)
            continue
        if not rows:
            issues.setdefault(table_name, []).append("Table has no data rows.")
            continue
        key = {
            "courses": "course_id",
            "teachers": "teacher_id",
            "rooms": "room_id",
            "groups": "group_id",
        }[table_name]
        seen: set[str] = set()
        for row_number, row in enumerate(rows, start=2):
            value = row.get(key)
            if value is None or not str(value).strip():
                issues.setdefault(table_name, []).append(
                    f"Row {row_number}: missing required identifier {key}."
                )
                continue
            normalized = str(value).strip()
            if normalized in seen:
                issues.setdefault(table_name, []).append(
                    f"Row {row_number}: duplicate {key} {normalized!r}."
                )
            seen.add(normalized)
            for field in required_fields[table_name]:
                field_value = row.get(field)
                if field_value is None or not str(field_value).strip():
                    issues.setdefault(table_name, []).append(
                        f"Row {row_number}: missing required value for {field}."
                    )
            if table_name == "courses" and not (
                str(row.get("group_id") or "").strip()
                or str(row.get("group_ids") or "").strip()
            ):
                issues.setdefault(table_name, []).append(
                    f"Row {row_number}: missing group_id/group_ids."
                )
            if not any(value is not None and str(value).strip() for value in row.values()):
                issues.setdefault(table_name, []).append(f"Row {row_number}: empty row.")
    return issues


def build_problem(
    tables: dict[str, list[dict[str, Any]]],
    days: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri"),
    slots_per_day: int = 8,
) -> Problem:
    if not days or slots_per_day < 1:
        raise ValueError("The timetable must have at least one day and one slot per day.")
    missing = [name for name in REQUIRED_TABLES if name not in tables]
    if missing:
        raise ValueError(f"Missing required data tables: {', '.join(missing)}")

    def require(row: dict[str, Any], key: str, table: str) -> str:
        value = row.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"{table} row is missing required field {key!r}.")
        return str(value).strip()

    groups: dict[str, dict[str, Any]] = {}
    for row in tables["groups"]:
        group_id = require(row, "group_id", "groups")
        if group_id in groups:
            raise ValueError(f"Duplicate group_id: {group_id}")
        groups[group_id] = row
    teachers: dict[str, Teacher] = {}
    for row in tables["teachers"]:
        teacher_id = require(row, "teacher_id", "teachers")
        if teacher_id in teachers:
            raise ValueError(f"Duplicate teacher_id: {teacher_id}")
        available = _slot_set(row.get("available_slots"), days)
        preferred = _slot_set(row.get("preferred_slots"), days)
        valid_slots = {
            (day, slot) for day in range(len(days)) for slot in range(slots_per_day)
        }
        if not available.issubset(valid_slots) or not preferred.issubset(valid_slots):
            raise ValueError(f"Teacher {teacher_id} has availability outside the timetable.")
        daily_value = row.get("max_hours_per_day")
        daily_limit = int(daily_value) if daily_value not in (None, "") else 6
        if daily_limit < 1:
            raise ValueError(f"Teacher {teacher_id} must have a positive max_hours_per_day.")
        teachers[teacher_id] = Teacher(
            teacher_id=teacher_id,
            name=str(row.get("name") or teacher_id).strip(),
            max_hours_per_day=daily_limit,
            available_slots=available or None,
            preferred_slots=preferred,
        )
    rooms: dict[str, Room] = {}
    for row in tables["rooms"]:
        room_id = require(row, "room_id", "rooms")
        if room_id in rooms:
            raise ValueError(f"Duplicate room_id: {room_id}")
        capacity = int(row.get("capacity") or 0)
        if capacity < 1:
            raise ValueError(f"Room {room_id} must have a positive capacity.")
        rooms[room_id] = Room(
            room_id=room_id,
            capacity=capacity,
            room_type=require(row, "room_type", "rooms").lower(),
            equipment=_tag_set(row.get("equipment")),
        )
        if rooms[room_id].room_type not in {"lecture", "lab"}:
            raise ValueError(f"Room {room_id} room_type must be 'lecture' or 'lab'.")

    meetings: list[Meeting] = []
    seen_courses: set[str] = set()
    for row in tables["courses"]:
        course_id = require(row, "course_id", "courses")
        if course_id in seen_courses:
            raise ValueError(f"Duplicate course_id: {course_id}")
        seen_courses.add(course_id)
        teacher_id = require(row, "teacher_id", "courses")
        if teacher_id not in teachers:
            raise ValueError(f"Course {course_id} references unknown teacher {teacher_id}.")
        group_ids = tuple(dict.fromkeys(
            part.strip()
            for part in str(row.get("group_ids") or row.get("group_id") or "").replace(",", "|").split("|")
            if part.strip()
        ))
        if not group_ids:
            raise ValueError(f"Course {course_id} must have at least one group_id.")
        unknown_groups = sorted(set(group_ids) - set(groups))
        if unknown_groups:
            raise ValueError(
                f"Course {course_id} references unknown groups: {', '.join(unknown_groups)}."
            )
        sessions_value = row.get("sessions_per_week")
        session_count = int(sessions_value) if sessions_value not in (None, "") else 1
        room_type = str(row.get("room_type") or "lecture").strip().lower()
        duration_value = row.get("duration")
        default_duration = 2 if room_type == "lab" else 1
        duration = int(duration_value) if duration_value not in (None, "") else default_duration
        if session_count < 1 or duration not in (1, 2):
            raise ValueError(f"Course {course_id} needs positive sessions and duration 1 or 2.")
        if room_type not in {"lecture", "lab"}:
            raise ValueError(f"Course {course_id} room_type must be 'lecture' or 'lab'.")
        if room_type == "lab" and duration != 2:
            raise ValueError(f"Lab course {course_id} must occupy exactly two consecutive slots.")
        enrollment = int(row.get("enrollment") or 0)
        if enrollment < 0:
            raise ValueError(f"Course {course_id} enrollment cannot be negative.")
        if enrollment <= 0:
            enrollment = max(int(groups[group_id].get("size") or 0) for group_id in group_ids)
        if enrollment <= 0:
            raise ValueError(f"Course {course_id} needs a positive enrollment or group size.")
        subject = str(row.get("subject") or row.get("course_name") or course_id).strip()
        course_name = str(row.get("course_name") or course_id).strip()
        for session_index in range(1, session_count + 1):
            required_equipment = _tag_set(row.get("required_equipment"))
            meeting_id = f"{course_id}-S{session_index}"
            meetings.append(
                Meeting(
                    meeting_id=meeting_id,
                    course_id=course_id,
                    course_name=course_name,
                    subject=subject,
                    teacher_id=teacher_id,
                    groups=group_ids,
                    enrollment=enrollment,
                    room_type=room_type,
                    duration=duration,
                    session_index=session_index,
                    sessions_per_week=session_count,
                    heavy=str(row.get("heavy") or "").strip().lower() in ("1", "true", "yes"),
                    department=str(row.get("department") or "General").strip(),
                    required_equipment=required_equipment,
                )
            )
    if not meetings:
        raise ValueError("The courses table contains no schedulable courses.")
    if not rooms:
        raise ValueError("The rooms table contains no rooms.")
    return Problem(teachers, rooms, tuple(meetings), slots_per_day, days)


def load_dataset(paths: dict[str, str | Path]) -> Problem:
    return build_problem({name: _read_rows(path) for name, path in paths.items()})


def read_uploaded_tables(files: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for upload in files:
        name = Path(upload.name).stem.lower()
        if name not in (*REQUIRED_TABLES, "attendance", "enrollment", "preference_feedback"):
            raise ValueError(
                f"Unrecognized data file {upload.name!r}; use courses, teachers, rooms, "
                "groups, attendance, enrollment, or preference_feedback."
            )
        suffix = Path(upload.name).suffix.lower()
        if suffix == ".csv":
            tables[name] = list(csv.DictReader(io.StringIO(upload.getvalue().decode("utf-8-sig"))))
        elif suffix in (".xlsx", ".xlsm"):
            try:
                from openpyxl import load_workbook
            except ImportError as exc:
                raise RuntimeError("Excel uploads require openpyxl.") from exc
            workbook = load_workbook(io.BytesIO(upload.getvalue()), read_only=True, data_only=True)
            values = workbook.active.iter_rows(values_only=True)
            headers = next(values, None)
            if headers:
                names = [str(value).strip() if value is not None else "" for value in headers]
                tables[name] = [
                    {key: value for key, value in zip(names, row)}
                    for row in values
                    if any(value is not None for value in row)
                ]
        else:
            raise ValueError(f"Unsupported upload type: {upload.name}")
    return tables


def load_uploaded_tables(files: Iterable[Any]) -> Problem:
    tables = read_uploaded_tables(files)
    return build_problem(tables)


def schedule_rows(problem: Problem, schedule: dict[str, Any]) -> list[dict[str, Any]]:
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    rows = []
    for meeting_id, placement in schedule.items():
        meeting = meeting_by_id[meeting_id]
        rows.append(
            {
                "meeting_id": meeting_id,
                "course_id": meeting.course_id,
                "course_name": meeting.course_name,
                "department": meeting.department,
                "subject": meeting.subject,
                "teacher_id": meeting.teacher_id,
                "groups": ", ".join(meeting.groups),
                "day": problem.days[placement.day],
                "start_slot": placement.start,
                "start_time": f"{8 + placement.start:02d}:00",
                "end_time": f"{8 + placement.start + meeting.duration:02d}:00",
                "room_id": placement.room_id,
                "room_type": meeting.room_type,
                "required_equipment": ", ".join(sorted(meeting.required_equipment)),
                "enrollment": meeting.enrollment,
                "duration": meeting.duration,
            }
        )
    return sorted(rows, key=lambda row: (problem.days.index(row["day"]), row["start_slot"], row["room_id"]))


def _filtered_schedule(
    problem: Problem,
    schedule: dict[str, Any],
    filter_by: str | None,
    filter_value: str | None,
) -> dict[str, Any]:
    if not filter_by or not filter_value or filter_value == "All":
        return schedule
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    if filter_by == "group":
        return {
            meeting_id: placement
            for meeting_id, placement in schedule.items()
            if filter_value in meeting_by_id[meeting_id].groups
        }
    if filter_by == "teacher":
        return {
            meeting_id: placement
            for meeting_id, placement in schedule.items()
            if meeting_by_id[meeting_id].teacher_id == filter_value
        }
    if filter_by == "room":
        return {
            meeting_id: placement
            for meeting_id, placement in schedule.items()
            if placement.room_id == filter_value
        }
    raise ValueError(f"Unknown timetable export filter: {filter_by}")


def export_excel(
    problem: Problem,
    schedule: dict[str, Any],
    *,
    filter_by: str | None = None,
    filter_value: str | None = None,
) -> bytes:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("Excel export requires openpyxl.") from exc
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Timetable"
    workbook.properties.title = (
        "IntelliSchedule AI – AI-Driven University Timetable Optimization and Predictive Scheduling"
    )
    workbook.properties.creator = "IntelliSchedule AI"
    rows = schedule_rows(problem, _filtered_schedule(problem, schedule, filter_by, filter_value))
    headers = list(rows[0]) if rows else ["meeting_id"]
    sheet.append(["IntelliSchedule AI – AI-Driven University Timetable Optimization and Predictive Scheduling"])
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    sheet["A1"].font = Font(bold=True, color="FFFFFF", size=14)
    sheet["A1"].fill = PatternFill("solid", fgColor="234F68")
    sheet.append(headers)
    for row in rows:
        sheet.append([row[key] for key in headers])
    for cell in sheet[2]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="234F68")
    sheet.freeze_panes = "A3"
    sheet.auto_filter.ref = sheet.dimensions
    for column_index, column in enumerate(sheet.columns, start=1):
        width = min(32, max(len(str(cell.value or "")) for cell in column) + 2)
        sheet.column_dimensions[get_column_letter(column_index)].width = width
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def export_pdf(
    problem: Problem,
    schedule: dict[str, Any],
    *,
    filter_by: str | None = None,
    filter_value: str | None = None,
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires reportlab.") from exc
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(letter))
    doc.title = "IntelliSchedule AI - AI-Driven University Timetable Optimization and Predictive Scheduling"
    doc.author = "IntelliSchedule AI"
    doc.creator = "IntelliSchedule AI"
    rows = schedule_rows(problem, _filtered_schedule(problem, schedule, filter_by, filter_value))
    headers = ["Day", "Time", "Course", "Groups", "Teacher", "Room"]
    data = [headers] + [
        [
            row["day"],
            f'{row["start_time"]}-{row["end_time"]}',
            row["course_name"],
            row["groups"],
            row["teacher_id"],
            row["room_id"],
        ]
        for row in rows
    ]
    styles = getSampleStyleSheet()
    title = Paragraph(
        "IntelliSchedule AI – AI-Driven University Timetable Optimization and Predictive Scheduling",
        styles["Title"],
    )
    subtitle = Paragraph("University timetable", styles["Normal"])
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#234f68")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf3f6")]),
            ]
        )
    )
    doc.build([title, Spacer(1, 10), subtitle, Spacer(1, 12), table])
    return output.getvalue()
