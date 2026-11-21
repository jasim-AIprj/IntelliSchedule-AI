"""Attendance/demand regressors and separately learned preference weights."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from .models import Meeting, Problem

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")
ATTENDANCE_FEATURES = (
    "subject",
    "weekday",
    "time_slot",
    "enrollment",
    "course_type",
    "previous_attendance",
)
DEMAND_FEATURES = ("department", "course_type", "historical_enrollment", "offered_sections", "elective")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else 0.0,
    }


def _normalize_attendance(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Attendance targets must not be blank or non-finite.")
    if result > 1:
        result /= 100.0
    if not 0 <= result <= 1:
        raise ValueError("Attendance targets must be between 0 and 1, or 0 and 100 percent.")
    return result


def _weekday_number(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    normalized = str(value).strip().lower()[:3]
    for index, day in enumerate(WEEKDAYS):
        if day.lower() == normalized:
            return index
    raise ValueError(f"Unknown weekday {value!r}; use Mon-Fri or a zero-based day index.")


def _encode_frame(frame: pd.DataFrame, categories: tuple[str, ...]) -> pd.DataFrame:
    encoded = pd.get_dummies(frame, columns=list(categories), dtype=float)
    return encoded.astype(float)


class PreferenceModel:
    """Random Forest attendance and demand predictions with honest holdout metrics."""

    def __init__(
        self,
        seed: int = 42,
        attendance_records: list[dict[str, Any]] | None = None,
        enrollment_records: list[dict[str, Any]] | None = None,
    ):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise RuntimeError("The ML layer requires scikit-learn; install requirements.txt.") from exc
        self.seed = seed
        self.attendance_model = RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=seed, n_jobs=1
        )
        self.demand_model = RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=3, random_state=seed, n_jobs=1
        )
        self.attendance_source = "uploaded historical records" if attendance_records else "synthetic"
        self.demand_source = "uploaded historical records" if enrollment_records else "synthetic"
        self.attendance_rows = self._attendance_data(attendance_records)
        self.demand_rows = self._demand_data(enrollment_records)
        self._fit_attendance()
        self._fit_demand()
        self._attendance_cache: dict[tuple[str, int, int], float] = {}
        self._demand_cache: dict[str, float] = {}
        self._best_attendance_cache: dict[str, float] = {}
        self.preference_weights = {
            "student_gaps": 2.0,
            "daily_load": 0.09,
            "heavy_adjacent": 2.0,
            "teacher_preferences": 1.25,
            "teacher_gaps": 1.0,
            "room_balance": 0.005,
            "attendance": 10.0,
        }
        self.preference_learning_source = "expert-configured defaults (no feedback file supplied)"
        self.preference_training_rows = 0

    def precompute(
        self,
        meetings: tuple[Meeting, ...],
        days: int = 5,
        slots_per_day: int = 8,
    ) -> None:
        """Batch model inference for the search domain instead of many tiny forest calls."""
        attendance_rows: list[dict[str, Any]] = []
        attendance_keys: list[tuple[str, int, int]] = []
        demand_rows: list[dict[str, Any]] = []
        demand_keys: list[str] = []
        for meeting in meetings:
            missing_slots = [
                (day, slot)
                for day in range(min(days, 5))
                for slot in range(slots_per_day - meeting.duration + 1)
                if (meeting.meeting_id, day, slot) not in self._attendance_cache
            ]
            if missing_slots:
                subject_history = self.attendance_rows.loc[
                    self.attendance_rows["subject"].astype(str).str.lower() == meeting.subject.lower(),
                    "attendance",
                ]
                previous = float(subject_history.tail(10).mean()) if not subject_history.empty else float(
                    self.attendance_rows["attendance"].mean()
                )
                for day, slot in missing_slots:
                    attendance_rows.append(
                        {
                            "subject": meeting.subject,
                            "weekday": day,
                            "time_slot": slot,
                            "enrollment": meeting.enrollment,
                            "course_type": meeting.room_type,
                            "previous_attendance": previous,
                        }
                    )
                    attendance_keys.append((meeting.meeting_id, day, slot))
            if meeting.meeting_id not in self._demand_cache:
                demand_rows.append(
                    {
                        "department": meeting.department,
                        "course_type": meeting.room_type,
                        "historical_enrollment": meeting.enrollment,
                        "offered_sections": meeting.sessions_per_week,
                        "elective": int(len(meeting.groups) > 1),
                    }
                )
                demand_keys.append(meeting.meeting_id)
        if attendance_rows:
            attendance_frame = self._encode_attendance(pd.DataFrame(attendance_rows))
            predictions = self.attendance_model.predict(attendance_frame)
            self._attendance_cache.update(
                {
                    key: min(1.0, max(0.0, float(prediction)))
                    for key, prediction in zip(attendance_keys, predictions)
                }
            )
        if demand_rows:
            demand_frame = _encode_frame(
                pd.DataFrame(demand_rows), ("department", "course_type")
            ).reindex(columns=self.demand_columns, fill_value=0)
            predictions = self.demand_model.predict(demand_frame)
            self._demand_cache.update(
                {
                    key: max(1.0, float(prediction))
                    for key, prediction in zip(demand_keys, predictions)
                }
            )
        for meeting in meetings:
            if meeting.meeting_id not in self._best_attendance_cache:
                possible = [
                    self._attendance_cache[(meeting.meeting_id, day, slot)]
                    for day in range(min(days, 5))
                    for slot in range(slots_per_day - meeting.duration + 1)
                    if (meeting.meeting_id, day, slot) in self._attendance_cache
                ]
                self._best_attendance_cache[meeting.meeting_id] = max(possible, default=0.0)

    def _attendance_data(self, records: list[dict[str, Any]] | None) -> pd.DataFrame:
        if records:
            frame = pd.DataFrame(records).copy()
            required = {"subject", "weekday", "time_slot", "enrollment", "course_type", "attendance"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"Attendance history is missing columns: {', '.join(sorted(missing))}.")
            if len(frame) < 5:
                raise ValueError("Attendance history needs at least 5 dated/ordered records.")
            if frame[list(required)].isna().any().any():
                raise ValueError("Attendance history has missing required feature/target values.")
            if frame[["subject", "course_type"]].astype(str).apply(
                lambda column: column.str.strip().eq("").any()
            ).any():
                raise ValueError("Attendance subject and course_type values must not be blank.")
            if "date" in frame:
                frame = frame.sort_values("date")
            if "previous_attendance" not in frame:
                frame["previous_attendance"] = frame.groupby("subject")["attendance"].shift(1)
                frame["previous_attendance"] = frame["previous_attendance"].fillna(
                    frame["attendance"].mean()
                )
            frame["weekday"] = frame["weekday"].map(_weekday_number)
            frame["attendance"] = frame["attendance"].map(_normalize_attendance)
            frame["previous_attendance"] = frame["previous_attendance"].map(_normalize_attendance)
            frame["time_slot"] = pd.to_numeric(frame["time_slot"], errors="raise").astype(int)
            frame["enrollment"] = pd.to_numeric(frame["enrollment"], errors="raise")
            if not frame["weekday"].between(0, 4).all():
                raise ValueError("Attendance history weekday values must be Monday-Friday.")
            if not frame["time_slot"].between(0, 7).all():
                raise ValueError("Attendance time_slot values must be between 0 and 7.")
            if (frame["enrollment"] <= 0).any():
                raise ValueError("Attendance enrollment must be positive.")
            return frame.reset_index(drop=True)

        rng = random.Random(self.seed)
        rows = []
        subjects = ("Algorithms", "Statistics", "Networks", "Databases", "Systems", "Design", "AI", "Security")
        previous_by_subject = {subject: 0.76 for subject in subjects}
        for index in range(2400):
            subject = subjects[index % len(subjects)]
            weekday = rng.randrange(5)
            slot = rng.randrange(8)
            enrollment = rng.randrange(18, 151)
            course_type = "lab" if rng.random() < 0.2 else "lecture"
            previous = previous_by_subject[subject]
            weekday_effect = (0.03, 0.01, 0.025, 0.0, -0.07)[weekday]
            midpoint_boost = 0.08 if slot in (2, 3, 4) else 0.0
            edge_penalty = 0.05 if slot in (0, 7) else 0.0
            expected = (
                previous + weekday_effect + midpoint_boost - edge_penalty
                - 0.0003 * max(0, enrollment - 70)
                + (0.025 if course_type == "lab" else 0)
                + rng.gauss(0, 0.06)
            )
            attendance = min(0.98, max(0.35, expected))
            rows.append(
                {
                    "subject": subject,
                    "weekday": weekday,
                    "time_slot": slot,
                    "enrollment": enrollment,
                    "course_type": course_type,
                    "previous_attendance": previous,
                    "attendance": attendance,
                    "date": index,
                }
            )
            previous_by_subject[subject] = 0.85 * previous + 0.15 * attendance
        return pd.DataFrame(rows)

    def _fit_attendance(self) -> None:
        from sklearn.impute import SimpleImputer

        records = self.attendance_rows
        features = list(ATTENDANCE_FEATURES)
        raw_x = records[features].copy()
        raw_x["subject"] = raw_x["subject"].astype(str)
        raw_x["course_type"] = raw_x["course_type"].astype(str)
        raw_x["weekday"] = raw_x["weekday"].astype(int)
        raw_x["time_slot"] = raw_x["time_slot"].astype(int)
        raw_x["enrollment"] = raw_x["enrollment"].astype(float)
        raw_x["previous_attendance"] = raw_x["previous_attendance"].astype(float)
        targets = records["attendance"].to_numpy(dtype=float)
        split = max(1, min(len(records) - 1, int(len(records) * 0.8)))
        train_x, test_x = raw_x.iloc[:split], raw_x.iloc[split:]
        train_y, test_y = targets[:split], targets[split:]
        self.attendance_categories = ("subject", "course_type")
        encoded_train = _encode_frame(train_x, self.attendance_categories)
        self.attendance_columns = tuple(encoded_train.columns)
        encoded_test = _encode_frame(test_x, self.attendance_categories).reindex(
            columns=self.attendance_columns, fill_value=0
        )
        self.attendance_model.fit(encoded_train, train_y)
        test_prediction = self.attendance_model.predict(encoded_test)
        self.attendance_metrics = _metrics(test_y, test_prediction)
        self.attendance_holdout = pd.DataFrame(
            {"actual": test_y, "predicted": test_prediction}
        )
        importance = dict(zip(self.attendance_columns, self.attendance_model.feature_importances_))
        self.attendance_feature_importance = self._group_importance(importance)

    def _demand_data(self, records: list[dict[str, Any]] | None) -> pd.DataFrame:
        if records:
            frame = pd.DataFrame(records).copy()
            required = {"department", "course_type", "historical_enrollment", "expected_enrollment"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"Enrollment history is missing columns: {', '.join(sorted(missing))}.")
            if len(frame) < 5:
                raise ValueError("Enrollment history needs at least 5 dated/ordered records.")
            if "offered_sections" not in frame:
                frame["offered_sections"] = 1
            if "elective" not in frame:
                frame["elective"] = 0
            if "date" in frame:
                frame = frame.sort_values("date")
            if frame["expected_enrollment"].isna().any() or frame["historical_enrollment"].isna().any():
                raise ValueError("Enrollment history contains missing enrollment values.")
            for column in ("historical_enrollment", "expected_enrollment", "offered_sections", "elective"):
                frame[column] = pd.to_numeric(frame[column], errors="raise")
            if (frame["historical_enrollment"] <= 0).any() or (frame["expected_enrollment"] <= 0).any():
                raise ValueError("Enrollment history values must be positive.")
            return frame.reset_index(drop=True)
        rng = random.Random(self.seed + 11)
        rows = []
        departments = ("Computing", "Engineering", "Business", "Science", "Arts")
        for index in range(900):
            enrollment = rng.randrange(16, 181)
            sections = max(1, round(enrollment / rng.randrange(28, 46)))
            elective = int(rng.random() < 0.28)
            course_type = "lab" if rng.random() < 0.18 else "lecture"
            multiplier = 1.10 + 0.08 * elective + 0.06 * (course_type == "lab")
            expected = max(10, enrollment * multiplier + rng.gauss(0, max(2, enrollment * 0.06)))
            rows.append(
                {
                    "department": departments[index % len(departments)],
                    "course_type": course_type,
                    "historical_enrollment": enrollment,
                    "offered_sections": sections,
                    "elective": elective,
                    "expected_enrollment": expected,
                    "date": index,
                }
            )
        return pd.DataFrame(rows)

    def _fit_demand(self) -> None:
        raw_x = self.demand_rows[list(DEMAND_FEATURES)].copy()
        raw_x["department"] = raw_x["department"].astype(str)
        raw_x["course_type"] = raw_x["course_type"].astype(str)
        raw_x["historical_enrollment"] = raw_x["historical_enrollment"].astype(float)
        raw_x["offered_sections"] = raw_x["offered_sections"].astype(float)
        raw_x["elective"] = raw_x["elective"].astype(float)
        targets = self.demand_rows["expected_enrollment"].to_numpy(dtype=float)
        split = max(1, min(len(raw_x) - 1, int(len(raw_x) * 0.8)))
        encoded_train = _encode_frame(raw_x.iloc[:split], ("department", "course_type"))
        self.demand_columns = tuple(encoded_train.columns)
        encoded_test = _encode_frame(raw_x.iloc[split:], ("department", "course_type")).reindex(
            columns=self.demand_columns, fill_value=0
        )
        self.demand_model.fit(encoded_train, targets[:split])
        demand_prediction = self.demand_model.predict(encoded_test)
        self.demand_metrics = _metrics(targets[split:], demand_prediction)
        self.demand_holdout = pd.DataFrame(
            {"actual": targets[split:], "predicted": demand_prediction}
        )
        importance = dict(zip(self.demand_columns, self.demand_model.feature_importances_))
        self.demand_feature_importance = self._group_importance(importance)

    @staticmethod
    def _group_importance(importance: dict[str, float]) -> dict[str, float]:
        grouped: dict[str, float] = {}
        for name, value in importance.items():
            feature = name.split("_", 1)[0]
            grouped[feature] = grouped.get(feature, 0.0) + float(value)
        return dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))

    def _encode_attendance(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = _encode_frame(frame, self.attendance_categories)
        return encoded.reindex(columns=self.attendance_columns, fill_value=0)

    def predict_attendance(self, meeting: Meeting, day: int, slot: int) -> float:
        key = (meeting.meeting_id, day, slot)
        if key not in self._attendance_cache:
            subject_history = self.attendance_rows.loc[
                self.attendance_rows["subject"].astype(str).str.lower() == meeting.subject.lower(),
                "attendance",
            ]
            previous = float(subject_history.tail(10).mean()) if not subject_history.empty else float(
                self.attendance_rows["attendance"].mean()
            )
            frame = pd.DataFrame(
                [{
                    "subject": meeting.subject,
                    "weekday": day,
                    "time_slot": slot,
                    "enrollment": meeting.enrollment,
                    "course_type": meeting.room_type,
                    "previous_attendance": previous,
                }]
            )
            prediction = float(self.attendance_model.predict(self._encode_attendance(frame))[0])
            self._attendance_cache[key] = min(1.0, max(0.0, prediction))
        return self._attendance_cache[key]

    def best_attendance(self, meeting: Meeting, days: int = 5, slots_per_day: int = 8) -> float:
        if meeting.meeting_id not in self._best_attendance_cache:
            self.precompute((meeting,), days, slots_per_day)
        return self._best_attendance_cache[meeting.meeting_id]

    def predict_demand(self, meeting: Meeting, sessions_per_week: int | None = None) -> float:
        if meeting.meeting_id not in self._demand_cache:
            frame = pd.DataFrame(
                [{
                    "department": meeting.department,
                    "course_type": meeting.room_type,
                    "historical_enrollment": meeting.enrollment,
                    "offered_sections": sessions_per_week or meeting.sessions_per_week,
                    "elective": int(len(meeting.groups) > 1),
                }]
            )
            encoded = _encode_frame(frame, ("department", "course_type")).reindex(
                columns=self.demand_columns, fill_value=0
            )
            prediction = float(self.demand_model.predict(encoded)[0])
            self._demand_cache[meeting.meeting_id] = max(1.0, prediction)
        return self._demand_cache[meeting.meeting_id]

    def fit_preference_feedback(self, records: list[dict[str, Any]]) -> bool:
        """Learn a separate soft-weight model from explicit historical preference ratings."""
        required = {
            "student_gaps", "daily_load_imbalance", "heavy_adjacent",
            "teacher_preference_misses", "teacher_gaps", "preference_rating",
        }
        if len(records) < 8:
            raise ValueError("At least 8 preference-feedback rows are required to fit weights.")
        frame = pd.DataFrame(records)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Preference feedback is missing columns: {', '.join(sorted(missing))}.")
        from sklearn.ensemble import RandomForestRegressor

        features = sorted(required - {"preference_rating"})
        if frame[list(required)].isna().any().any():
            raise ValueError("Preference feedback contains missing feature or rating values.")
        frame[list(required)] = frame[list(required)].apply(pd.to_numeric, errors="raise")
        if not frame["preference_rating"].between(1, 5).all():
            raise ValueError("preference_rating must be between 1 and 5.")
        forest = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=self.seed)
        forest.fit(frame[features].astype(float), frame["preference_rating"].astype(float))
        raw = dict(zip(features, forest.feature_importances_))
        mapping = {
            "student_gaps": "student_gaps",
            "daily_load_imbalance": "daily_load",
            "heavy_adjacent": "heavy_adjacent",
            "teacher_preference_misses": "teacher_preferences",
            "teacher_gaps": "teacher_gaps",
        }
        scale = max(raw.values(), default=1.0)
        for feature, key in mapping.items():
            if feature in raw:
                self.preference_weights[key] = round(0.1 + 4.9 * raw[feature] / max(scale, 1e-9), 3)
        self.preference_training_rows = len(frame)
        self.preference_learning_source = f"Random Forest learned from {len(frame)} explicit feedback rows"
        return True


def create_preference_model(
    seed: int = 42,
    attendance_records: list[dict[str, Any]] | None = None,
    enrollment_records: list[dict[str, Any]] | None = None,
) -> PreferenceModel:
    return PreferenceModel(seed, attendance_records, enrollment_records)


def model_description(problem: Problem) -> str:
    del problem
    return (
        "Attendance and demand Random Forests are trained on the available historical files, "
        "or generated synthetic records when none are uploaded. Synthetic-only scores do not "
        "demonstrate real-world accuracy. Attendance predictions affect soft slot utility; "
        "they do not automatically learn preference weights."
    )
