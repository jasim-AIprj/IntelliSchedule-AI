"""Core scheduling and repair tests."""

from __future__ import annotations

import unittest
import io
from dataclasses import replace

from timetable.analytics import explain_placement, hard_constraint_count
from timetable.constraints import solve_csp, validate_schedule
from timetable.data import build_problem, export_excel, export_pdf, validate_tables
from timetable.ml import create_preference_model
from timetable.models import Meeting, Placement
from timetable.optimizer import (
    objective_breakdown,
    objective_cost,
    optimize,
    optimize_detailed,
    schedule_metrics,
)
from timetable.reschedule import disruption_cost, repair_schedule


def small_problem():
    return build_problem(
        {
            "groups": [{"group_id": "G1", "size": 20}, {"group_id": "G2", "size": 20}],
            "teachers": [
                {"teacher_id": "T1", "max_hours_per_day": 4, "available_slots": "*", "preferred_slots": "Mon:0|Mon:1"},
                {"teacher_id": "T2", "max_hours_per_day": 4},
            ],
            "rooms": [
                {"room_id": "R1", "capacity": 30, "room_type": "lecture"},
                {"room_id": "LAB1", "capacity": 30, "room_type": "lab", "equipment": "spectrometer|computing"},
                {"room_id": "LAB2", "capacity": 30, "room_type": "lab", "equipment": "spectrometer|computing"},
            ],
            "courses": [
                {"course_id": "C1", "teacher_id": "T1", "group_id": "G1", "room_type": "lecture", "sessions_per_week": 2},
                {"course_id": "C2", "teacher_id": "T2", "group_ids": "G1|G2", "room_type": "lab", "required_equipment": "spectrometer", "sessions_per_week": 1},
                {"course_id": "C3", "teacher_id": "T2", "group_id": "G2", "room_type": "lecture", "sessions_per_week": 1},
            ],
        }
    )


class TimetableTests(unittest.TestCase):
    def test_csp_builds_valid_schedule_and_labs_use_two_slots(self):
        problem = small_problem()
        schedule, stats = solve_csp(problem, return_stats=True)
        self.assertEqual(validate_schedule(problem, schedule), [])
        self.assertEqual(stats.variables, len(problem.meetings))
        self.assertGreater(stats.constraints, 0)
        self.assertGreaterEqual(stats.nodes, len(problem.meetings))
        for meeting in problem.meetings:
            if meeting.room_type == "lab":
                self.assertEqual(meeting.duration, 2)
            if meeting.required_equipment:
                self.assertTrue(
                    meeting.required_equipment.issubset(
                        problem.rooms[schedule[meeting.meeting_id].room_id].equipment
                    )
                )

    def test_overlapping_elective_groups_cannot_overlap(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        elective = next(meeting for meeting in problem.meetings if len(meeting.groups) == 2)
        for meeting in problem.meetings:
            if meeting.meeting_id == elective.meeting_id or not set(meeting.groups).intersection(elective.groups):
                continue
            left = schedule[elective.meeting_id]
            right = schedule[meeting.meeting_id]
            if left.day == right.day:
                left_slots = set(range(left.start, left.start + elective.duration))
                right_slots = set(range(right.start, right.start + meeting.duration))
                self.assertFalse(left_slots.intersection(right_slots))

    def test_optimizer_preserves_hard_constraints(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        seed_cost = objective_cost(problem, schedule)
        optimized, cost = optimize(problem, schedule, "annealing", iterations=80, seed=3)
        self.assertEqual(validate_schedule(problem, optimized), [])
        self.assertLessEqual(cost, seed_cost)

    def test_all_optimizers_return_feasible_convergence_traces(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        for method in ("annealing", "tabu", "hill-climbing"):
            result = optimize_detailed(problem, schedule, method, iterations=12, seed=7)
            self.assertEqual(validate_schedule(problem, result.schedule), [])
            self.assertEqual(len(result.convergence), 13)
            self.assertLessEqual(result.best_cost, result.initial_cost)
            self.assertGreaterEqual(result.accepted_moves, 0)
            self.assertGreaterEqual(result.rejected_moves, 0)

    def test_objective_breakdown_sums_to_the_measured_cost(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        model = create_preference_model()
        breakdown = objective_breakdown(problem, schedule, model)
        self.assertAlmostEqual(
            sum(breakdown.values()),
            objective_cost(problem, schedule, model),
            places=6,
        )

    def test_adaptive_annealing_reheats_and_stops_reproducibly(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        options = {
            "iterations": 20,
            "seed": 5,
            "reheat_after": 1,
            "max_reheats": 1,
            "early_stopping_rounds": 3,
            "max_runtime_seconds": 3,
        }
        first = optimize_detailed(problem, schedule, "annealing", **options)
        second = optimize_detailed(problem, schedule, "annealing", **options)
        self.assertEqual(first.reheats, 1)
        self.assertTrue(any(row.get("reheated") for row in first.convergence))
        self.assertEqual(first.stop_reason, "early stopping after annealing stagnation")
        self.assertEqual(first.schedule, second.schedule)
        self.assertEqual(validate_schedule(problem, first.schedule), [])

    def test_means_ends_repair_handles_teacher_absence_and_room_outage(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        teacher_meeting = next(m for m in problem.meetings if m.teacher_id == "T1")
        absent_day = schedule[teacher_meeting.meeting_id].day
        repaired, moved = repair_schedule(
            problem, schedule, absent_teacher="T1", absent_day=absent_day
        )
        self.assertIn(teacher_meeting.meeting_id, moved)
        self.assertNotEqual(repaired[teacher_meeting.meeting_id].day, absent_day)
        self.assertEqual(validate_schedule(problem, repaired), [])

        lab_meeting = next(m for m in problem.meetings if m.room_type == "lab")
        repaired_room, room_moved = repair_schedule(
            problem,
            schedule,
            unavailable_room=schedule[lab_meeting.meeting_id].room_id,
        )
        self.assertIn(lab_meeting.meeting_id, room_moved)
        self.assertNotEqual(repaired_room[lab_meeting.meeting_id].room_id, schedule[lab_meeting.meeting_id].room_id)
        self.assertEqual(validate_schedule(problem, repaired_room), [])
        closed_problem = replace(
            problem,
            closed_rooms=frozenset({schedule[lab_meeting.meeting_id].room_id}),
        )
        self.assertEqual(validate_schedule(closed_problem, repaired_room), [])
        equipment_meeting = next(m for m in problem.meetings if m.required_equipment)
        equipment_room = schedule[equipment_meeting.meeting_id].room_id
        repaired_equipment, equipment_moved = repair_schedule(
            problem,
            schedule,
            unavailable_equipment="spectrometer",
            equipment_room=equipment_room,
        )
        self.assertIn(equipment_meeting.meeting_id, equipment_moved)
        self.assertNotEqual(repaired_equipment[equipment_meeting.meeting_id].room_id, equipment_room)
        self.assertEqual(validate_schedule(problem, repaired_equipment), [])
        equipment_problem = replace(
            problem,
            unavailable_equipment_by_room={
                equipment_room: frozenset({"spectrometer"})
            },
        )
        self.assertEqual(validate_schedule(equipment_problem, repaired_equipment), [])

    def test_teacher_availability_change_and_new_class_repair(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        teacher_meeting = next(meeting for meeting in problem.meetings if meeting.teacher_id == "T1")
        placement = schedule[teacher_meeting.meeting_id]
        changed, moved = repair_schedule(
            problem,
            schedule,
            teacher_unavailable_slots={
                "T1": {
                    (placement.day, slot)
                    for slot in range(placement.start, placement.start + teacher_meeting.duration)
                }
            },
        )
        self.assertIn(teacher_meeting.meeting_id, moved)
        self.assertEqual(validate_schedule(problem, changed), [])

        added = Meeting(
            meeting_id="C4-S1",
            course_id="C4",
            course_name="New lecture",
            subject="New lecture",
            teacher_id="T1",
            groups=("G1",),
            enrollment=20,
            room_type="lecture",
            duration=1,
            session_index=1,
        )
        with_new, new_moved = repair_schedule(
            problem, schedule, added_meeting=added
        )
        from dataclasses import replace

        expanded = replace(problem, meetings=problem.meetings + (added,))
        self.assertIn("C4-S1", new_moved)
        self.assertEqual(validate_schedule(expanded, with_new), [])
        self.assertGreaterEqual(disruption_cost(schedule, with_new, expanded), 1.0)

    def test_increased_enrollment_scenario_repairs_against_new_capacity(self):
        problem = small_problem()
        lecture = next(meeting for meeting in problem.meetings if meeting.room_type == "lecture")
        expanded_room = replace(problem.rooms["R1"], capacity=60)
        scenario_problem = replace(
            problem,
            rooms={**problem.rooms, "R1": expanded_room},
        )
        scenario_schedule, _nodes = solve_csp(scenario_problem)
        new_enrollment = lecture.enrollment + 5
        repaired, _moved = repair_schedule(
            scenario_problem,
            scenario_schedule,
            enrollment_updates={lecture.meeting_id: new_enrollment},
        )
        updated_problem = replace(
            scenario_problem,
            meetings=tuple(
                replace(meeting, enrollment=new_enrollment)
                if meeting.meeting_id == lecture.meeting_id
                else meeting
                for meeting in scenario_problem.meetings
            ),
        )
        self.assertEqual(validate_schedule(updated_problem, repaired), [])

    def test_enrollment_scenario_rejects_unknown_meetings(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        with self.assertRaisesRegex(ValueError, "Unknown meeting"):
            repair_schedule(problem, schedule, enrollment_updates={"UNKNOWN": 45})

    def test_repair_can_cascade_to_minimally_move_a_direct_blocker(self):
        problem = build_problem(
            {
                "groups": [{"group_id": "G1", "size": 20}],
                "teachers": [
                    {"teacher_id": "T1", "available_slots": "Mon:0|Mon:1", "max_hours_per_day": 2},
                    {"teacher_id": "T2", "available_slots": "Mon:1|Mon:2", "max_hours_per_day": 2},
                ],
                "rooms": [{"room_id": "R1", "capacity": 30, "room_type": "lecture"}],
                "courses": [
                    {"course_id": "A", "teacher_id": "T1", "group_id": "G1"},
                    {"course_id": "B", "teacher_id": "T2", "group_id": "G1"},
                ],
            },
            days=("Mon",),
            slots_per_day=3,
        )
        original = {
            "A-S1": Placement(0, 0, "R1"),
            "B-S1": Placement(0, 1, "R1"),
        }
        self.assertEqual(validate_schedule(problem, original), [])
        repaired, moved = repair_schedule(
            problem,
            original,
            teacher_unavailable_slots={"T1": {(0, 0)}},
        )
        self.assertEqual(set(moved), {"A-S1", "B-S1"})
        self.assertEqual(repaired["A-S1"].start, 1)
        self.assertEqual(repaired["B-S1"].start, 2)
        self.assertEqual(validate_schedule(problem, repaired), [])

    def test_conflict_detector_reports_missing_meetings(self):
        problem = small_problem()
        errors = validate_schedule(problem, {})
        self.assertEqual(len(errors), len(problem.meetings))

    def test_lab_duration_cannot_be_overridden_to_one_slot(self):
        tables = {
            "groups": [{"group_id": "G1", "size": 20}],
            "teachers": [{"teacher_id": "T1"}],
            "rooms": [{"room_id": "LAB", "capacity": 30, "room_type": "lab"}],
            "courses": [{
                "course_id": "LAB1",
                "teacher_id": "T1",
                "group_id": "G1",
                "room_type": "lab",
                "duration": 1,
            }],
        }
        with self.assertRaisesRegex(ValueError, "exactly two consecutive"):
            build_problem(tables)

    def test_excel_and_pdf_exports_are_generated(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        excel = export_excel(problem, schedule)
        pdf = export_pdf(problem, schedule)
        self.assertTrue(excel.startswith(b"PK"))
        self.assertTrue(pdf.startswith(b"%PDF"))
        from openpyxl import load_workbook

        workbook = load_workbook(io.BytesIO(excel), read_only=True)
        self.assertIn("IntelliSchedule AI", workbook.properties.title)
        self.assertIn("IntelliSchedule AI", workbook.active["A1"].value)
        self.assertIn(b"IntelliSchedule AI", pdf)
        group_excel = export_excel(problem, schedule, filter_by="group", filter_value="G1")
        self.assertTrue(group_excel.startswith(b"PK"))

    def test_ml_training_metrics_prediction_and_separate_preference_learning(self):
        model = create_preference_model()
        self.assertTrue(0 <= model.attendance_metrics["mae"] <= 1)
        self.assertGreaterEqual(model.attendance_metrics["r2"], -1)
        self.assertGreater(model.demand_metrics["mae"], 0)
        meeting = small_problem().meetings[0]
        prediction = model.predict_attendance(meeting, 1, 3)
        self.assertTrue(0 <= prediction <= 1)
        demand = model.predict_demand(meeting)
        self.assertGreater(demand, 0)
        feedback = [
            {
                "student_gaps": index % 4,
                "daily_load_imbalance": index % 3,
                "heavy_adjacent": index % 2,
                "teacher_preference_misses": index % 5,
                "teacher_gaps": index % 3,
                "preference_rating": 1 + index % 5,
            }
            for index in range(12)
        ]
        model.fit_preference_feedback(feedback)
        self.assertEqual(model.preference_training_rows, 12)
        self.assertIn("12 explicit", model.preference_learning_source)

    def test_ml_can_train_from_uploaded_history(self):
        attendance = [
            {
                "date": index,
                "subject": "AI" if index % 2 else "Math",
                "weekday": ("Mon", "Tue", "Wed", "Thu", "Fri")[index % 5],
                "time_slot": index % 8,
                "enrollment": 30 + index % 15,
                "course_type": "lecture",
                "attendance": 0.65 + (index % 20) / 100,
                "previous_attendance": 0.7,
            }
            for index in range(80)
        ]
        enrollment = [
            {
                "date": index,
                "department": "Computing",
                "course_type": "lecture",
                "historical_enrollment": 30 + index % 20,
                "offered_sections": 1 + index % 3,
                "elective": index % 2,
                "expected_enrollment": 35 + index % 25,
            }
            for index in range(60)
        ]
        model = create_preference_model(
            attendance_records=attendance,
            enrollment_records=enrollment,
        )
        self.assertEqual(model.attendance_source, "uploaded historical records")
        self.assertEqual(model.demand_source, "uploaded historical records")
        self.assertEqual(len(model.attendance_holdout), 16)

    def test_schedule_metrics_explanations_and_data_validation(self):
        problem = small_problem()
        schedule, _nodes = solve_csp(problem)
        metrics = schedule_metrics(problem, schedule)
        self.assertGreaterEqual(metrics["student_idle_periods"], 0)
        self.assertGreater(metrics["classroom_utilization_pct"], 0)
        self.assertGreaterEqual(hard_constraint_count(problem), len(problem.meetings))
        explanation = explain_placement(problem, schedule, next(iter(schedule)))
        self.assertTrue(explanation["reasons"])
        self.assertIsNotNone(explanation["schedule_cost"])
        issues = validate_tables({
            "groups": [{"group_id": "G1"}, {"group_id": "G1"}],
            "teachers": [{"teacher_id": "T1"}],
            "rooms": [{"room_id": "R1"}],
            "courses": [{"course_id": "C1"}],
        })
        self.assertTrue(any("duplicate" in item.lower() for item in issues["groups"]))

    def test_optimizer_rejects_an_invalid_seed(self):
        problem = small_problem()
        with self.assertRaisesRegex(ValueError, "complete hard-feasible"):
            optimize_detailed(problem, {}, "tabu", iterations=2)


if __name__ == "__main__":
    unittest.main()
