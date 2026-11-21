"""Generate the 20-page IntelliSchedule AI project documentation PDF."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepInFrame,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "IntelliSchedule_AI_Project_Documentation.pdf"
NAVY = colors.HexColor("#17233b")
INK = colors.HexColor("#26344d")
MUTED = colors.HexColor("#59667d")
INDIGO = colors.HexColor("#5146e5")
PALE = colors.HexColor("#efedff")
PAPER = colors.HexColor("#ffffff")
LINE = colors.HexColor("#dce2ed")
TEAL = colors.HexColor("#0f806d")
PAGE_WIDTH, PAGE_HEIGHT = letter

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=MUTED,
        spaceAfter=13,
    )
)
styles.add(
    ParagraphStyle(
        name="DocBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.15,
        leading=12.4,
        textColor=INK,
        spaceAfter=7,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="DocSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.2,
        textColor=MUTED,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=INDIGO,
        spaceBefore=5,
        spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="DocH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=9.6,
        leading=12,
        textColor=NAVY,
        spaceBefore=3,
        spaceAfter=3,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="TableHead",
        fontName="Helvetica-Bold",
        fontSize=7.7,
        leading=9.3,
        textColor=PAPER,
    )
)
styles.add(
    ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.3,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=31,
        leading=37,
        textColor=PAPER,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverSubtitle",
        fontName="Helvetica",
        fontSize=13,
        leading=19,
        textColor=colors.HexColor("#d9ddff"),
        spaceAfter=15,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverSmall",
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#c4cbe0"),
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="CoverLabel",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#aaa3ff"),
        spaceAfter=9,
    )
)


def para(text: str, style: str = "DocBody") -> Paragraph:
    return Paragraph(escape(text), styles[style])


def markup(text: str, style: str = "DocBody") -> Paragraph:
    return Paragraph(text, styles[style])


def heading(text: str) -> Paragraph:
    return para(text, "DocH2")


def bullet_list(items: list[str]) -> list[Paragraph]:
    return [para(f"- {item}") for item in items]


def table(
    rows: list[list[str]],
    widths: list[float],
    *,
    font_size: float = 7.5,
    repeat_rows: int = 1,
) -> Table:
    cell_style = ParagraphStyle(
        f"Cell{font_size}",
        parent=styles["TableCell"],
        fontSize=font_size,
        leading=font_size + 1.8,
    )
    head_style = ParagraphStyle(
        f"Head{font_size}",
        parent=styles["TableHead"],
        fontSize=font_size,
        leading=font_size + 1.8,
    )
    data = [
        [
            Paragraph(escape(str(value)), head_style if row_index == 0 else cell_style)
            for value in row
        ]
        for row_index, row in enumerate(rows)
    ]
    result = Table(data, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), PAPER),
                ("BACKGROUND", (0, 1), (-1, -1), PAPER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, colors.HexColor("#f5f7fb")]),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return result


def callout(title: str, text: str, color=PALE) -> Table:
    safe_text = escape(text).replace("\n", "<br/>")
    content = Paragraph(
        f'<font color="#3930a8"><b>{escape(title)}</b></font><br/>{safe_text}',
        styles["DocBody"],
    )
    block = Table([[content]], colWidths=[7.15 * inch])
    block.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return block


def flow_row(labels: list[str], widths: list[float] | None = None) -> Table:
    if widths is None:
        widths = [7.15 * inch / len(labels)] * len(labels)
    cells = []
    for label in labels:
        cells.append(
            Paragraph(
                escape(label),
                ParagraphStyle(
                    "FlowCell",
                    parent=styles["TableCell"],
                    fontName="Helvetica-Bold",
                    fontSize=8,
                    leading=10,
                    textColor=NAVY,
                    alignment=TA_CENTER,
                ),
            )
        )
    result = Table([cells], colWidths=widths)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def on_page(canvas, doc) -> None:
    page_number = canvas.getPageNumber()
    canvas.saveState()
    if page_number == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#24284e"))
        canvas.circle(PAGE_WIDTH - 40, PAGE_HEIGHT - 30, 190, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#30345f"))
        canvas.circle(PAGE_WIDTH + 35, PAGE_HEIGHT + 25, 115, stroke=0, fill=1)
        canvas.setFillColor(INDIGO)
        canvas.roundRect(42, 44, 58, 5, 2, stroke=0, fill=1)
    else:
        canvas.setFillColor(colors.HexColor("#f5f7fb"))
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        canvas.setFillColor(INDIGO)
        canvas.roundRect(42, PAGE_HEIGHT - 34, 24, 4, 2, stroke=0, fill=1)
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(73, PAGE_HEIGHT - 36, "INTELLISCHEDULE AI")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.2)
        canvas.drawRightString(
            PAGE_WIDTH - 42,
            PAGE_HEIGHT - 36,
            "PROJECT DOCUMENTATION  |  2026",
        )
        canvas.setStrokeColor(LINE)
        canvas.line(42, PAGE_HEIGHT - 46, PAGE_WIDTH - 42, PAGE_HEIGHT - 46)
        canvas.line(42, 38, PAGE_WIDTH - 42, 38)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(42, 25, "Implementation reference - synthetic data clearly labeled")
        canvas.drawRightString(PAGE_WIDTH - 42, 25, f"{page_number:02d} / 20")
    canvas.restoreState()


def page(title: str, subtitle: str, content: list) -> list:
    flowables = [para(title, "DocTitle"), para(subtitle, "DocSubtitle")]
    flowables.extend(content)
    return [KeepInFrame(7.25 * inch, 9.15 * inch, flowables, mode="shrink")]


def cover_page() -> list:
    content = [
        Spacer(1, 1.2 * inch),
        para("AI COURSE PROJECT  |  TECHNICAL DOCUMENTATION", "CoverLabel"),
        para("IntelliSchedule AI", "CoverTitle"),
        para(
            "AI-Driven University Timetable Optimization and Predictive Scheduling",
            "CoverSubtitle",
        ),
        Spacer(1, 0.18 * inch),
        para(
            "Smarter Scheduling. Better Academic Planning.",
            "CoverSubtitle",
        ),
        para(
            "A full-scale implementation guide to the project as built: its constraint solver, "
            "metaheuristic optimizers, machine-learning layer, interactive Streamlit product, "
            "scenario repair, data contracts, evaluation evidence, and known limitations.",
            "CoverSmall",
        ),
        Spacer(1, 0.4 * inch),
        callout(
            "Implementation snapshot",
            "Python 3.12 project; 8 Streamlit workspaces; 110-section / 220-meeting campus data; "
            "8-section interactive demonstration; Random Forest attendance and demand models; "
            "custom CSP, SA, Tabu, Hill Climbing, and CSP-guided Means-Ends repair.",
            color=colors.HexColor("#e9eaff"),
        ),
        Spacer(1, 0.35 * inch),
        para("Prepared from the implemented source tree and measured evaluation artifacts.", "CoverSmall"),
        para("Versioned project snapshot: 15 November 2026", "CoverSmall"),
        Spacer(1, 0.25 * inch),
        para(
            "Evidence note: bundled training and demo data are synthetic. Reported scores are "
            "instance-specific experiments, not institutional accuracy or service guarantees.",
            "CoverSmall",
        ),
    ]
    return [KeepInFrame(7.2 * inch, 9.2 * inch, content, mode="shrink")]


PAGES: list[tuple[str, str, list]] = [
    (
        "Reading guide and executive summary",
        "What the system does, who this document is for, and how to interpret the evidence.",
        [
            para(
                "IntelliSchedule AI is a university timetable planning prototype. It treats class "
                "placement as a constraint-satisfaction problem, first finding a complete assignment "
                "that respects non-negotiable rules, then searching feasible alternatives to improve "
                "schedule quality. A Random Forest layer estimates attendance by day and period and "
                "course demand; those predictions inform soft scheduling utility but never waive official "
                "room-capacity requirements."
            ),
            heading("Project outcomes"),
            *bullet_list(
                [
                    "A typed scheduling model for teachers, rooms, meetings, groups, and placements.",
                    "Backtracking CSP with dynamic MRV, degree tie-break, forward checking, and bounded search.",
                    "Simulated Annealing as the main optimizer; Tabu Search and Hill Climbing as baselines.",
                    "A CSP-guided local repair process for operational disruptions, with explicit preview/apply.",
                    "A Streamlit interface for generation, timetable exploration, ML analysis, analytics, data import, and exports.",
                ]
            ),
            heading("How to use this document"),
            para(
                "Pages 3-10 explain the scheduling model and algorithms; pages 11-14 cover learning, "
                "repair, and explanations; pages 15-16 describe the user-facing platform and persistence; "
                "pages 17-20 record results, tests, installation, and limitations. The companion "
                "evaluation_report.md contains the detailed benchmark run. The source code remains the "
                "authoritative implementation."
            ),
            callout(
                "Interpretation",
                "Synthetic-only Random Forest metrics describe the generated holdout distribution. "
                "They do not establish predictive accuracy for a real university.",
            ),
        ],
    ),
    (
        "Problem statement, objectives, and scope",
        "The academic scheduling problem and the boundaries of this prototype.",
        [
            para(
                "A timetable assigns each weekly class meeting a weekday, contiguous start period, "
                "and compatible classroom. Meetings consume resources: instructors, student cohorts, "
                "and rooms. Any overlap involving a shared resource is invalid. Capacity, room type, "
                "equipment, teacher availability, daily workload, and lab duration further restrict "
                "which assignments are possible."
            ),
            heading("Functional objectives"),
            *bullet_list(
                [
                    "Generate a complete clash-free week using constraint-first search.",
                    "Improve soft quality: fewer idle periods, more balanced daily loads, fewer undesirable adjacent classes, and better preference alignment.",
                    "Support different calendar projections: student group, teacher, room, department, course search, and weekday.",
                    "Expose useful model evidence, optimization diagnostics, comparisons, and decision explanations.",
                    "Preview a repair after a disruption and require an explicit user action before adopting it.",
                ]
            ),
            heading("Operational assumptions"),
            para(
                "The default calendar is Monday-Friday with eight one-hour periods, indexed 0-7 "
                "from 08:00 to 16:00. A lecture occupies one period by default; a lab occupies "
                "two adjacent periods. Enrollment is attached to each meeting. Electives are represented "
                "by one meeting reserving multiple group identifiers together."
            ),
            heading("Out of scope"),
            para(
                "This is not a production student-information-system integration. It does not model "
                "individual student registration, travel time between campuses, accessibility matching, "
                "variable period lengths, room setup transitions, or live institutional feeds. A feasible "
                "prototype schedule is not a substitute for registrar approval."
            ),
        ],
    ),
    (
        "System architecture",
        "How data, algorithms, models, interface, persistence, and exports fit together.",
        [
            para(
                "The application is a modular Python program. app.py orchestrates Streamlit controls "
                "and calls the domain modules. CSV/Excel tables are validated and converted into "
                "dataclasses. The CSP constructs a feasible seed; optimizers search it without violating "
                "hard constraints. Analytics validate and explain outputs. SQLite records runs and "
                "uploaded tables; exporters produce portable files."
            ),
            flow_row(
                [
                    "CSV / Excel / demo",
                    "Validation + domain model",
                    "CSP feasible seed",
                    "SA / Tabu / Hill",
                    "Calendar + analytics",
                ]
            ),
            Spacer(1, 6),
            flow_row(
                ["Historical or synthetic records", "Random Forest models", "Attendance utility + demand forecast"]
            ),
            Spacer(1, 6),
            flow_row(
                ["Absence / outage / scenario", "Means-Ends + bounded CSP repair", "Preview -> explicit apply"]
            ),
            heading("Module responsibilities"),
            table(
                [
                    ["Module", "Responsibility"],
                    ["app.py", "Streamlit interface, session state, workflows, charts, and integration."],
                    ["timetable/models.py", "Teacher, Room, Meeting, Placement, and Problem types."],
                    ["constraints.py", "Candidate domains, bitset occupancy, validation, and CSP search."],
                    ["optimizer.py", "Soft objective, schedule metrics, SA, Tabu, Hill Climbing."],
                    ["ml.py", "Attendance/demand forests, holdout metrics, prediction caches, feedback weights."],
                    ["reschedule.py", "Disruption transformations and low-change local CSP repair."],
                    ["analytics.py / data.py / storage.py", "Explanation, import/export, persistence."],
                ],
                [1.65 * inch, 5.5 * inch],
                font_size=7.2,
            ),
            para(
                "Dependency direction is intentionally simple: the domain layer is independent of "
                "Streamlit; UI integration consumes typed algorithm APIs rather than reimplementing "
                "constraint rules.",
                "DocSmall",
            ),
        ],
    ),
    (
        "Domain model and timetable representation",
        "The typed entities, assignment format, and representation of resources and sessions.",
        [
            table(
                [
                    ["Entity", "Important fields", "Meaning"],
                    ["Teacher", "ID, display name, max hours/day, available slots, preferred slots", "Resource with hard availability and soft preferences."],
                    ["Room", "ID, capacity, type, equipment", "Physical space with hard compatibility requirements."],
                    ["Meeting", "IDs, course, teacher, groups, enrollment, type, duration, equipment", "One scheduled weekly occurrence."],
                    ["Placement", "day index, start slot, room ID", "Decision value assigned to a meeting."],
                    ["Problem", "teacher map, room map, meeting tuple, days, slots/day, outages", "Immutable-in-practice scheduling instance."],
                ],
                [1.05 * inch, 2.85 * inch, 3.25 * inch],
                font_size=7,
            ),
            heading("Meeting identity and repeated sessions"),
            para(
                "Course rows expand into individual meeting records. A meeting ID distinguishes each "
                "weekly occurrence (for example, course session 1 versus session 2), while course ID "
                "groups the occurrences for display and analysis. Placements are stored by meeting ID. "
                "This allows each repeated session to receive its own day, period, and room."
            ),
            heading("Elective overlap"),
            para(
                "A course can reference more than one student group through group_id or group_ids. "
                "Occupancy reserves every listed group for the full duration. Two meetings sharing any "
                "one of those groups therefore cannot overlap, even when their other groups differ."
            ),
            heading("Bitset occupancy"),
            para(
                "The occupancy engine maps each day/period interval to an integer bit mask. Separate "
                "masks represent teacher, room, and group resources. A bitwise intersection tests "
                "overlap efficiently. Per-teacher, per-day hour counters enforce daily caps separately "
                "from overlap checks."
            ),
            callout(
                "Scenario state",
                "Closed rooms, room-specific unavailable equipment, and teacher slot restrictions are "
                "represented in the active Problem state, so subsequent validation and generation honor them.",
            ),
        ],
    ),
    (
        "Data ingestion, schemas, and validation",
        "Supported tables, field meanings, parsing rules, and import safeguards.",
        [
            para(
                "Data Management accepts separate CSV files or Excel workbooks. CSV files are read "
                "as UTF-8 with optional BOM; Excel import uses the active/first worksheet. Required "
                "schedule tables are courses, teachers, rooms, and groups. Optional files provide "
                "attendance history, enrollment history, or explicit preference feedback."
            ),
            table(
                [
                    ["Table", "Required", "Selected optional fields"],
                    ["courses", "course_id, teacher_id, group_id or group_ids", "course_name, subject, department, enrollment, room_type, duration, sessions_per_week, heavy, required_equipment"],
                    ["teachers", "teacher_id", "name, max_hours_per_day, available_slots, preferred_slots"],
                    ["rooms", "room_id, room_type, capacity", "equipment"],
                    ["groups", "group_id", "name, size"],
                    ["attendance", "subject, weekday, time_slot, enrollment, course_type, attendance", "date, previous_attendance"],
                    ["enrollment", "department, course_type, historical_enrollment, expected_enrollment", "date, offered_sections, elective"],
                    ["preference_feedback", "six numeric features and preference_rating", "At least eight labeled examples."],
                ],
                [1.05 * inch, 2.8 * inch, 3.3 * inch],
                font_size=6.7,
            ),
            heading("Normalization and validation"),
            *bullet_list(
                [
                    "Identifiers are trimmed and duplicates/missing identifiers are reported.",
                    "Group lists and equipment tags accept pipe or comma separators.",
                    "Availability/preference tokens use weekday:zero-based-slot, such as Mon:0|Tue:3.",
                    "Blank availability or * means no restriction; an empty preference set means no declared preference.",
                    "Room capacities, room types, slot bounds, teacher references, enrollment, and equipment compatibility are validated.",
                    "Data-quality warnings are retained in the Data Management workspace; invalid construction errors are shown to the user.",
                ]
            ),
            para(
                "Room capacity always compares against official enrollment, not an ML estimate. Predicted "
                "demand is advisory and does not mutate imported course records.",
                "DocSmall",
            ),
        ],
    ),
    (
        "Hard constraints and the CSP solver",
        "The feasibility layer: legal domains, backtracking, heuristics, propagation, and diagnostics.",
        [
            heading("Hard rules"),
            *bullet_list(
                [
                    "A teacher cannot teach overlapping meetings.",
                    "A room cannot host overlapping meetings.",
                    "A student group cannot attend overlapping meetings; multi-group electives reserve all groups.",
                    "Room capacity must be at least official meeting enrollment.",
                    "Room type and required equipment must match; closed rooms and unavailable equipment are excluded.",
                    "Every occupied period must be within teacher availability and within that teacher's daily teaching-hour limit.",
                    "Laboratory meetings occupy exactly two consecutive periods.",
                ]
            ),
            heading("Constraint-satisfaction formulation"),
            para(
                "For every meeting i, choose one variable value x_i = (day, start, room) from its "
                "legal domain D_i. For any two meetings sharing a teacher, group, or room, their "
                "occupied intervals must be disjoint. Unary constraints prune incompatible rooms, "
                "period windows, availability, lab duration, and capacities before search."
            ),
            heading("Search procedure"),
            para(
                "The solver applies dynamic Minimum Remaining Values (choose the unassigned meeting "
                "with the fewest currently legal values), with a degree tie-break favoring meetings "
                "connected to more unassigned neighbors. After a tentative assignment, forward checking "
                "removes conflicting values from neighboring domains. Empty domains trigger backtracking."
            ),
            heading("Bounded search and outcomes"),
            para(
                "Search is bounded by node count and wall-clock timeout (defaults in the implementation "
                "are 100,000 nodes and 60 seconds). SolverStats reports variable count, modeled constraints, "
                "visited nodes, backtracks, and elapsed seconds. If no solution is found within the bound, "
                "the solver raises a diagnostic error: that means no solution was found within the limit, "
                "not proof that no solution exists."
            ),
        ],
    ),
    (
        "Soft objective and measurable quality",
        "How feasible schedules are scored and how weighted terms are exposed.",
        [
            para(
                "After a feasible schedule exists, objective_cost computes a lower-is-better cost. "
                "Hard constraints are not traded against soft quality: the search neighborhood admits "
                "only placements that preserve hard feasibility. The objective weights are a combination "
                "of configured defaults or explicit preference-feedback learning."
            ),
            table(
                [
                    ["Term", "Implemented contribution", "Default weight"],
                    ["Student gaps", "Empty periods between the first and last occupied group period/day.", "2.00"],
                    ["Daily load", "Squared deviation of each cohort's daily contact hours from its weekly mean.", "0.09"],
                    ["Heavy adjacency", "Penalty for consecutive heavy or same-subject sessions.", "2.00"],
                    ["Teacher preferences", "Occupied periods outside the instructor's declared preferred set.", "1.25"],
                    ["Teacher gaps", "Idle periods between each instructor's first and last occupied period/day.", "1.00"],
                    ["Room balance", "Squared room-load deviation from average scheduled hours.", "0.005"],
                    ["Attendance utility", "Penalty relative to model-predicted best feasible attendance slot.", "10.00"],
                    ["Teacher load balance", "Squared total-hours deviation; fixed coefficient 0.02.", "0.02 fixed"],
                ],
                [1.15 * inch, 4.65 * inch, 1.35 * inch],
                font_size=7.05,
            ),
            heading("Score and metrics"),
            para(
                "The UI also reports soft_score = 100 / (1 + cost / meeting_count), rounded for display. "
                "This is a convenience transformation, not a probability or percentage of satisfied "
                "constraints. Separate schedule metrics report group and teacher idle periods, utilization, "
                "teacher fairness, and preference satisfaction."
            ),
            callout(
                "Objective explanation",
                "The Optimization workspace isolates additive weighted terms and confirms that their sum "
                "matches the measured objective. The ML comparison changes only the attendance weight.",
            ),
        ],
    ),
    (
        "Simulated Annealing: principal optimizer",
        "Feasible single-meeting moves, temperature schedule, adaptive search, reheating, and stops.",
        [
            para(
                "Simulated Annealing starts from a complete, validated timetable. It selects a meeting, "
                "temporarily removes its occupancy, samples candidate day/start/room values, and filters "
                "them through the same occupancy rules used by the CSP. Candidate objective values are "
                "measured on the resulting whole schedule. The best solution is stored separately so "
                "accepted uphill moves cannot erase the best-known result."
            ),
            heading("Acceptance and temperature"),
            para(
                "An improving move is accepted. A worse move may be accepted with Metropolis probability "
                "exp(-delta / T), where delta is the cost increase and T is temperature. The nominal "
                "schedule is T_k = max(T_min, T_0 * alpha^k), with configurable initial temperature "
                "and cooling rate. The default UI exposes T_0 and alpha."
            ),
            heading("Adaptive controls added to the application"),
            *bullet_list(
                [
                    "Recent best-improvement rate adjusts the sampled neighborhood: low improvement increases candidate sampling.",
                    "After a configurable stagnant interval, reheating resets the temperature schedule, up to a bounded number of reheats.",
                    "An early-stop stagnation limit prevents wasting iterations after reheating is exhausted.",
                    "A fixed seed makes experiments reproducible; runtime and iteration bounds remain explicit.",
                    "Convergence rows record current cost, best cost, temperature, acceptance, and reheat events.",
                ]
            ),
            heading("Diagnostics"),
            para(
                "Results include initial and best costs, completed iterations, accepted and rejected "
                "moves, rejection reasons, reheat count, elapsed time, stop reason, and convergence "
                "history. The engine is intentionally an assignment-relocation neighborhood. Dedicated "
                "multi-session swap or room-only exchange operators are not separately modeled as named "
                "move types."
            ),
        ],
    ),
    (
        "Baseline optimizers and algorithm comparison",
        "Why the project compares SA with Tabu Search and Hill Climbing, and how comparison works.",
        [
            table(
                [
                    ["Method", "Candidate choice", "Mechanism", "Typical limitation"],
                    ["Simulated Annealing", "Sampled feasible single-meeting placements.", "Metropolis acceptance for some uphill moves; cooling and reheat.", "Outcome depends on seed, temperature, budget, and instance."],
                    ["Tabu Search", "Best acceptable sampled feasible move.", "Short-term tabu memory blocks reverse assignments; aspiration permits a new global best.", "A small search sample or tenure can miss better regions."],
                    ["Hill Climbing", "Best sampled feasible neighbor.", "Accept only improvement; stop after a sampled local plateau.", "Can get trapped at a local optimum."],
                ],
                [1.15 * inch, 1.85 * inch, 2.55 * inch, 1.6 * inch],
                font_size=6.8,
            ),
            heading("Fair comparison protocol"),
            para(
                "The Optimization Lab creates a shared deterministic CSP seed. Each algorithm receives "
                "that seed and the same objective, iteration selector, and random seed. The UI records "
                "runtime, iterations, accepted moves, final cost, soft score, student idle periods, "
                "fairness, and hard violations. Every result is revalidated."
            ),
            heading("Interpretation"),
            para(
                "No method is universally best. On the recorded 220-meeting synthetic instance, Hill "
                "Climbing attained the lowest cost among these tested runs; that result is specific to "
                "that instance and run configuration. Multi-seed distributions and statistical confidence "
                "intervals are not yet reported."
            ),
            heading("CSP versus optimizer"),
            para(
                "The CSP is responsible for feasibility and may use preferences to order candidate values; "
                "it does not optimize the full soft objective. Metaheuristics start after feasibility is "
                "established and attempt to lower that objective without violating hard rules."
            ),
        ],
    ),
    (
        "Machine learning: models, data, and evaluation",
        "Random Forest training sources, features, chronological holdouts, and honest interpretation.",
        [
            para(
                "PreferenceModel owns two scikit-learn RandomForestRegressor models: attendance and "
                "course demand. If user history is absent, repeatable synthetic records are generated "
                "with a fixed seed. The interface labels synthetic data and warns that its scores do not "
                "demonstrate real institutional accuracy."
            ),
            table(
                [
                    ["Model", "Inputs", "Target", "Reported metrics"],
                    ["Attendance", "Subject, weekday, time slot, enrollment, course type, previous attendance.", "Attendance fraction from 0 to 1 (or percent input).", "MAE, RMSE, R2; actual-versus-predicted and feature importance."],
                    ["Demand", "Department, course type, historical enrollment, offered sections, elective flag.", "Expected enrollment.", "MAE, RMSE, R2; held-out scatter and feature importance."],
                    ["Preference weights", "Student gaps, daily imbalance, adjacent-heavy count, teacher misses/gaps.", "Explicit preference rating, 1-5.", "A separate fit updates soft preference weights; requires at least 8 rows."],
                ],
                [1.1 * inch, 2.75 * inch, 1.8 * inch, 1.5 * inch],
                font_size=6.8,
            ),
            heading("Temporal evaluation and leakage controls"),
            para(
                "When a usable date column is present, attendance/enrollment rows are date-sorted before "
                "a first-80%-train, last-20%-holdout split. Otherwise original row order is treated as "
                "history. The attendance previous-attendance feature is derived using subject-wise prior "
                "rows when not supplied, then the model evaluates the ordered holdout. This is a practical "
                "course-project temporal holdout, not a full rolling-origin backtest."
            ),
            heading("Recorded synthetic-model snapshot"),
            para(
                "The recorded evaluation report gives attendance MAE 0.0386 fraction, RMSE 0.0494, "
                "R2 0.588; demand MAE 5.77 students, RMSE 7.61, R2 0.981. These are synthetic-only "
                "holdout statistics. They must not be presented as real-world accuracy."
            ),
        ],
    ),
    (
        "ML integration and impact exploration",
        "Where predictions enter decisions, what remains fixed, and how alternatives are shown.",
        [
            para(
                "For each meeting and candidate period, the attendance forest predicts a value between "
                "zero and one. The objective computes a penalty relative to the best predicted attendance "
                "slot, scaled by the attendance preference weight and predicted demand relative to official "
                "enrollment. Batch precomputation and caches keep repeated inference practical."
            ),
            heading("Impact Explorer"),
            *bullet_list(
                [
                    "Select a class, weekday, and start period to inspect predicted attendance and expected headcount.",
                    "When an active timetable exists, see assigned room capacity alongside official enrollment.",
                    "Evaluate hard-feasible alternative slots/rooms and show their full-schedule objective and attendance-term cost.",
                    "Review subject-filtered historical attendance trends where matching records exist.",
                    "Compare two seeded SA runs with attendance weight enabled versus set to zero.",
                ]
            ),
            heading("Controlled A/B comparison"),
            para(
                "Both optimizer runs use the same starting schedule, random seed, iteration limit, hard "
                "constraints, and all non-attendance preference weights. Only the attendance coefficient "
                "changes. Final objective, score, student idle periods, runtime, and hard violations are "
                "measured. Either run can win; the UI reports observed outcomes rather than assuming ML "
                "improves a schedule."
            ),
            callout(
                "Capacity safety",
                "Predicted attendance and forecast demand are advisory. Room domains and validation "
                "always compare room capacity with official meeting enrollment.",
            ),
        ],
    ),
    (
        "Dynamic rescheduling and scenario simulation",
        "Disruptions, local CSP repair, minimal-change priorities, and explicit apply semantics.",
        [
            para(
                "The repair API builds an active scenario Problem that encodes changed teacher "
                "availability, a day-long absence, closed rooms, room-specific unavailable equipment, "
                "official enrollment updates, or a new meeting. It finds directly affected meetings and "
                "constructs a shared-resource graph to identify nearby blockers."
            ),
            heading("Repair search"),
            para(
                "Unaffected meetings are fixed when possible. Affected meetings and a bounded cascade "
                "of direct blockers are placed with CSP occupancy validation. Candidate ordering favors "
                "preserving the old day, time, and room, then uses objective cost as a tie-break. Search "
                "has a 25,000-node bound and a maximum cascade of eight additional meetings."
            ),
            heading("Available scenarios in the UI"),
            *bullet_list(
                [
                    "Professor absent for a selected weekday or changed availability periods.",
                    "Classroom closure or a selected piece of lab equipment becoming unavailable in one room.",
                    "Increased official enrollment for a selected meeting, which may force a larger room or another feasible placement.",
                    "Changed professor time preferences, followed by a bounded optimization pass.",
                    "A new lecture or lab meeting with a selected cohort, instructor, and equipment requirements.",
                ]
            ),
            heading("Preview and commit boundary"),
            para(
                "The repair workspace stores a pending preview, displays moved meetings, disruption cost, "
                "soft score, idle-time changes, and hard-violation count. It does not replace the active "
                "Problem or timetable until the user presses Apply repaired timetable. Discard removes "
                "the preview without altering the active schedule."
            ),
            para(
                "This is a bounded local heuristic, not a globally optimal repair guarantee. Infeasibility "
                "within the search bound is reported explicitly.",
                "DocSmall",
            ),
        ],
    ),
    (
        "Explainability, analytics, and decision support",
        "How the platform makes schedule quality and placement rationale inspectable.",
        [
            para(
                "The analytics module reports computed constraint counts and provides a factual "
                "placement explanation. It is not a generative explanation service: messages are derived "
                "from the actual assigned values, model predictions, preference data, and tested "
                "alternatives."
            ),
            heading("Why This Schedule?"),
            *bullet_list(
                [
                    "Validates the current timetable and explains that the placement passes hard checks.",
                    "Shows official enrollment versus assigned room capacity and verifies required equipment.",
                    "Reports teacher preference matches/misses, or that no preference data was supplied.",
                    "Displays attendance and demand model outputs while stating that predictions do not waive capacity.",
                    "Checks up to a bounded sample of feasible local alternatives and compares whole-schedule cost.",
                ]
            ),
            heading("Operational metrics"),
            table(
                [
                    ["Metric", "Interpretation"],
                    ["Student idle periods", "Unoccupied group periods between first and last class within each day."],
                    ["Teacher idle periods", "Equivalent instructor gaps across teaching days."],
                    ["Teacher fairness", "Dispersion-based teaching-load summary; not a labor-equity guarantee."],
                    ["Teacher preference match", "Share of scheduled periods that fall inside declared preferred slots."],
                    ["Room utilization", "Scheduled room-hours divided by all room-hours available in the modeled week."],
                    ["Hard violations", "Validator findings; a valid complete schedule should report zero."],
                ],
                [1.75 * inch, 5.4 * inch],
                font_size=7.2,
            ),
            callout(
                "Metric caution",
                "Room utilization is a time-utilization ratio, not seat occupancy or energy usage. "
                "The fairness figure is a descriptive summary and does not encode contracts or workload preferences.",
            ),
        ],
    ),
    (
        "Streamlit interface and user workflows",
        "The eight workspaces, theme behavior, sample demo, and presentation path.",
        [
            table(
                [
                    ["Workspace", "Main purpose"],
                    ["Overview", "Dataset summary, generation action, workflow, saved-run preview, and live quick demo."],
                    ["Generate", "CSP diagnostics, optimized schedule, and feasibility/quality details."],
                    ["Optimize", "Objective breakdown, convergence, SA diagnostics, and algorithm comparison."],
                    ["ML Insights", "Model holdouts, feature importance, attendance heatmap, demand, ML Impact Explorer."],
                    ["Repair", "Scenario builder, non-destructive repair preview, apply/discard, and timetable exports."],
                    ["Timetable", "Calendar by group/teacher/room, filter/search, and baseline versus optimized view."],
                    ["Analytics", "Load, utilization, idle time, conflict checks, and Why This Schedule explanations."],
                    ["Data", "Import validation, input templates, saved runs, models, exports, and project reports."],
                ],
                [1.25 * inch, 5.9 * inch],
                font_size=7.1,
            ),
            heading("Design and accessibility"),
            para(
                "A visible Light/Dark selector applies separate semantic palettes to app backgrounds, "
                "controls, cards, charts, tables, tabs, and sidebar. The selected mode survives normal "
                "Streamlit interactions through session state. The timetable uses a weekly grid; tables "
                "use explicit palette-aware styles; chart templates follow the selected mode."
            ),
            heading("Interactive and presentation paths"),
            para(
                "The homepage can run the actual solver and optimizer on the 8-section sample without "
                "replacing the active campus schedule. Presentation Mode guides users through the problem, "
                "data, CSP seed, optimization, ML prediction, repair, algorithm comparison, and final "
                "results. Individual stages invoke real project functions or show actual data."
            ),
        ],
    ),
    (
        "Persistence, exports, and project structure",
        "What is stored, what gets exported, and where implementation responsibilities live.",
        [
            heading("SQLite persistence"),
            para(
                "RunStore uses SQLite to persist timetable runs, uploaded dataset tables, and experiment "
                "measurements. Run records contain timestamp, algorithm, soft cost, runtime, and serialized "
                "meeting placements. Dataset rows preserve table data as JSON. Experiment rows retain "
                "initial/final cost, elapsed time, iteration and acceptance counts, plus JSON metrics. "
                "The redesign retains the existing database schema."
            ),
            heading("Exports"),
            *bullet_list(
                [
                    "CSV exports the selected schedule rows for spreadsheet/data workflows.",
                    "Excel creates a styled Timetable sheet with IntelliSchedule AI title metadata, frozen headers, autofilter, and readable widths.",
                    "PDF creates a landscape, print-ready timetable with branded title and repeating column headers.",
                    "Group, professor, or room filters support focused schedule handoffs.",
                    "The app also exposes ML metrics, algorithm comparison, and academic evaluation reports.",
                ]
            ),
            heading("Repository map"),
            table(
                [
                    ["Path", "What it contains"],
                    ["app.py", "Streamlit orchestration, state, themes, workspaces, charts, scenarios, and demo."],
                    ["timetable/constraints.py", "CSP domains, occupancy bitsets, validator, solver diagnostics."],
                    ["timetable/optimizer.py", "Objective, metrics, objective breakdown, SA, Tabu, Hill Climbing."],
                    ["timetable/ml.py", "Random Forest data creation/fitting, prediction and feedback learning."],
                    ["timetable/reschedule.py", "Means-Ends state changes and bounded repair search."],
                    ["timetable/data.py / storage.py", "Imports/exports and SQLite persistence."],
                    ["scripts/ and tests/", "Demo generation, benchmark report, and automated behavior tests."],
                ],
                [2.4 * inch, 4.75 * inch],
                font_size=7.1,
            ),
        ],
    ),
    (
        "Measured evaluation and current demo result",
        "Recorded benchmark values and one measured quick-demo run, each with its limitations.",
        [
            para(
                "The checked-in evaluation report records a deterministic synthetic 110-section, "
                "220-meeting instance with 35 teachers, 16 rooms, and 10 groups. CSP used seed 42, "
                "reported 220 nodes and zero backtracks, and took 24.562 seconds in that run. Each "
                "metaheuristic shared that seed and had a 150-iteration / 12-second budget."
            ),
            table(
                [
                    ["Algorithm", "Initial", "Final", "Improve", "Runtime s", "Idle", "Violations"],
                    ["CSP seed", "941.232", "941.232", "0.00%", "24.562", "78", "0"],
                    ["Simulated Annealing", "941.232", "884.946", "5.98%", "4.693", "58", "0"],
                    ["Tabu Search", "941.232", "805.768", "14.39%", "4.697", "48", "0"],
                    ["Hill Climbing", "941.232", "777.822", "17.36%", "4.542", "36", "0"],
                ],
                [1.7 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.65 * inch, 0.85 * inch],
                font_size=6.8,
            ),
            Spacer(1, 8),
            para(
                "A separately executed quick-demo sanity run contained 8 sections and 13 weekly "
                "meetings. CSP used 13 nodes and zero backtracks. The 160-iteration seeded SA run "
                "changed objective cost from 28.916 to 27.802, ran for 0.473 seconds, accepted 103 "
                "moves, recorded 57 rejections and one reheat, and finished with zero hard violations. "
                "This is a single sample instance, not a universal improvement guarantee."
            ),
            heading("What can and cannot be concluded"),
            *bullet_list(
                [
                    "The validator confirms all compared schedules passed the modeled hard constraints.",
                    "On the recorded full synthetic run, Hill Climbing achieved the lowest final cost.",
                    "Runtime depends on data, machine, Python/library versions, and the time budget.",
                    "The experiment does not show generalization to real university systems or statistical significance across instances.",
                ]
            ),
            callout(
                "Evidence provenance",
                "Full-instance numbers are copied from evaluation_report.md. Quick-demo numbers were measured "
                "from the included quick dataset and current implementation on 15 November 2026.",
            ),
        ],
    ),
    (
        "Testing and quality assurance",
        "Automated coverage, UI smoke checks, scenario verification, and the evidence boundary.",
        [
            para(
                "The current automated suite contains 18 unittest cases. It covers the CSP, feasibility "
                "validation, all three optimizers, objective accounting, adaptive annealing, room/equipment "
                "restrictions, disruption repair, ML training, explanation, data validation, and exports."
            ),
            table(
                [
                    ["Area", "Assertions exercised"],
                    ["CSP / hard constraints", "Valid assignment; labs span two slots; elective groups cannot overlap; missing meetings detected."],
                    ["Optimization", "SA, Tabu, Hill preserve feasibility; traces are populated; best cost is not worse than initial; seed behavior and reheat stop path."],
                    ["Objective / analytics", "Additive breakdown sums to the measured objective; schedule metrics and factual explanations are returned."],
                    ["Repair", "Absence, room/equipment outage, changed availability, added class, enrollment scenario, and minimal cascade."],
                    ["ML / data", "Synthetic and uploaded history fit; metrics/predictions work; feedback learning stays separate; validation catches invalid rows."],
                    ["Exports", "Excel/PDF signatures and product title metadata; filtered export supported."],
                ],
                [1.55 * inch, 5.6 * inch],
                font_size=7.1,
            ),
            heading("Interactive acceptance checks executed"),
            *bullet_list(
                [
                    "App compiles and Streamlit AppTest renders all eight tabs without exceptions.",
                    "Light and Dark selection each render with no Streamlit exception.",
                    "Homepage action toggles methodology; live CSP + SA demo produces computed metrics and zero violations.",
                    "Presentation CSP and optimization stages complete on the quick dataset.",
                    "ML signal-on/off A/B computation completes.",
                    "Absence repair creates a preview while preserving the original active schedule; explicit apply changes it.",
                ]
            ),
            para(
                "Visual contrast and browser screenshots were also inspected. The tests do not replace "
                "assistive-technology audits, real-user usability studies, or performance tests on a real SIS.",
                "DocSmall",
            ),
        ],
    ),
    (
        "Technology stack and setup",
        "Dependencies, local run instructions, demo flow, and operational notes.",
        [
            table(
                [
                    ["Technology", "Use in project", "Environment observed"],
                    ["Python", "Language and runtime.", "3.12.5"],
                    ["Streamlit", "Interactive web application.", "1.67.0"],
                    ["pandas / NumPy", "Tables, transformations, and numeric support.", "2.2.3 / 2.1.2"],
                    ["scikit-learn", "Random Forest regression and metrics.", "1.7.2"],
                    ["Plotly", "Interactive charts.", "7.1.0"],
                    ["SQLite", "Local run, dataset, and experiment persistence.", "Python sqlite3"],
                    ["openpyxl / ReportLab", "Excel and PDF timetable exports.", "3.1.5 / 5.1.0"],
                ],
                [1.45 * inch, 3.7 * inch, 2.0 * inch],
                font_size=7.1,
            ),
            heading("Windows quick start"),
            callout(
                "Commands",
                "py -m venv .venv\n"
                ".\\.venv\\Scripts\\Activate.ps1\n"
                "py -m pip install -r requirements.txt\n"
                "py scripts\\generate_demo_data.py\n"
                "py -m streamlit run app.py",
            ),
            heading("Run tests and evaluation"),
            para(
                "Run py -m unittest discover -s tests -v for the automated suite. Run "
                "py scripts\\evaluate.py to recompute the full synthetic benchmark and write "
                "evaluation_report.md. Evaluation is bounded, can take tens of seconds, and changes "
                "with hardware/runtime; preserve the generated report with its measurement timestamp."
            ),
            heading("Recommended first demonstration"),
            para(
                "Open Overview; choose Quick demo in the sidebar for a small dataset; use the homepage "
                "interactive demo to show CSP versus SA; inspect the timetable by group; open Optimize "
                "to display the objective breakdown and algorithm comparison; open ML Insights to show "
                "the synthetic label and attendance impact; then preview an absence repair and demonstrate "
                "that Apply is explicit."
            ),
        ],
    ),
    (
        "Limitations, responsible use, and next steps",
        "Known boundaries and a practical roadmap for moving from course prototype to stronger evidence.",
        [
            heading("Current limitations"),
            *bullet_list(
                [
                    "Demo attendance and demand rows are synthetic; their metrics do not establish real-world prediction quality.",
                    "Weekly equal-length slots omit travel time, multi-campus distances, student-level registration, accessibility, and room transition rules.",
                    "CSP and repair are bounded; search failure within the configured limit is not a proof of infeasibility.",
                    "Repair is local and bounded, not a globally minimal or globally optimal schedule guarantee.",
                    "Teacher fairness and room utilization are simplified summaries; neither replaces policy or operational review.",
                    "Temporal holdout is a single ordered 80/20 split rather than rolling-origin evaluation.",
                    "No OR-Tools/XGBoost comparison or confidence interval across many datasets is included.",
                ]
            ),
            heading("Responsible deployment checklist"),
            *bullet_list(
                [
                    "Obtain authorization and governance approval before uploading identifiable attendance/enrollment data.",
                    "Minimize data fields, control SQLite file access, and define retention/deletion policy.",
                    "Measure models on representative historical cohorts and assess subgroup performance and drift.",
                    "Keep registrar/faculty review in the loop; communicate that the solver proposes, not authorizes, a timetable.",
                    "Treat constraint configuration, preference weights, and scenario previews as auditable decisions.",
                ]
            ),
            heading("Recommended roadmap"),
            para(
                "Next technical steps are real authorized datasets, rolling temporal validation, multiple "
                "random seeds and confidence intervals, travel/accessibility constraints, richer swap and "
                "room-exchange neighborhoods, benchmark comparison with OR-Tools, production-grade "
                "authentication/storage, and user testing with registrars and faculty."
            ),
            callout(
                "Final summary",
                "The project demonstrates a coherent constraint-first scheduling pipeline with measurable "
                "soft optimization, advisory ML, explainability, local disruption repair, and a working "
                "interactive interface. Its strongest academic presentation is to clearly separate hard "
                "feasibility, objective trade-offs, synthetic model evidence, and observed instance results.",
            ),
        ],
    ),
]


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=42,
        leftMargin=42,
        topMargin=55,
        bottomMargin=48,
        title="IntelliSchedule AI - Project Documentation",
        author="IntelliSchedule AI Project",
        subject="20-page technical and academic documentation",
    )
    story = cover_page()
    for title, subtitle, content in PAGES:
        story.append(PageBreak())
        story.extend(page(title, subtitle, content))
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Generated {OUTPUT}")
    print(f"Documentation sections: {len(PAGES) + 1} pages")


if __name__ == "__main__":
    main()
