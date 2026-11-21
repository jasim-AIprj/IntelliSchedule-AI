"""IntelliSchedule AI: AI-driven university timetable optimization and prediction."""

from __future__ import annotations

import hashlib
import html
import copy
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from timetable.analytics import explain_placement, hard_constraint_count
from timetable.constraints import Occupancy, meeting_domains, solve_csp, validate_schedule
from timetable.data import (
    build_problem,
    export_excel,
    export_pdf,
    load_dataset,
    read_uploaded_tables,
    schedule_rows,
    validate_tables,
)
from timetable.ml import create_preference_model, model_description
from timetable.models import Meeting, Placement, Problem
from timetable.optimizer import (
    objective_cost,
    objective_breakdown,
    optimize_detailed,
    schedule_metrics,
    soft_score,
    teacher_load_fairness,
)
from timetable.reschedule import disruption_cost, repair_schedule
from timetable.storage import RunStore

ROOT = Path(__file__).resolve().parent
COLORS = ("#4F46E5", "#0F9F8F", "#F28C45", "#C45283", "#4085D5", "#8A63C7", "#299E79")
st.session_state.setdefault("theme_mode", "Light")
THEME = st.session_state["theme_mode"]
CHART_TEMPLATE = "plotly_dark" if THEME == "Dark" else "plotly_white"
CHART_TEXT = "#e2e8f0" if THEME == "Dark" else "#46536a"
px.defaults.template = CHART_TEMPLATE
st.set_page_config(
    page_title="IntelliSchedule AI | Timetable Planner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #17233b; --muted: #748099; --line: #e8ebf2; --paper: #ffffff;
      --canvas: #f5f7fb; --violet: #5146e5; --mint: #17a887;
    }
    html, body, [class*="css"] { font-family: 'Aptos', 'Segoe UI', sans-serif; color: var(--ink); }
    .stApp { background: var(--canvas); }
    [data-testid="stHeader"] { background: rgba(245,247,251,.9); }
    [data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 1.65rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #64708a; }
    h1, h2, h3 { font-family: 'Aptos Display', 'Segoe UI', sans-serif !important; color: var(--ink) !important; letter-spacing: -.035em; }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.12rem !important; }
    .brand { display:flex; align-items:center; gap:11px; padding: 5px 3px 22px; }
    .brand-icon { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; color:white; font-size:20px; background:linear-gradient(145deg,#6659f2,#4238cc); box-shadow:0 7px 18px #5146e533; }
    .brand-name { font:800 17px 'Aptos Display','Segoe UI',sans-serif; color:#17233b; line-height:1.15; }
    .brand-sub { font-size:11px; color:#8a94a9; margin-top:3px; letter-spacing:.08em; text-transform:uppercase; }
    .section-label { color:#98a1b3; font-weight:700; font-size:10px; text-transform:uppercase; letter-spacing:.11em; margin:15px 0 7px; }
    .hero { position:relative; overflow:hidden; padding:25px 29px; border:1px solid #e8e8ff; border-radius:22px; background:linear-gradient(115deg,#fff 10%,#f1f0ff 70%,#eeefff 100%); margin-bottom:22px; }
    .hero:after { content:''; position:absolute; width:230px; height:230px; right:6%; top:-140px; border-radius:50%; background:radial-gradient(circle,#d7d4ff 0,rgba(215,212,255,0) 70%); }
    .hero-kicker { color:#6358d9; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.12em; margin-bottom:9px; }
    .hero-title { font:800 27px 'Aptos Display','Segoe UI',sans-serif; color:#17233b; letter-spacing:-.045em; margin:0 0 6px; }
    .hero-sub { color:#747f98; font-size:13px; margin:0; max-width:720px; }
    .hero-badge { position:absolute; right:26px; bottom:23px; z-index:1; padding:8px 13px; border:1px solid #dedcff; border-radius:99px; background:#fff; color:#5b54c9; font-size:11px; font-weight:700; }
    .metric-card { background:#fff; border:1px solid var(--line); border-radius:17px; padding:16px 18px; min-height:104px; box-shadow:0 3px 12px rgba(28,39,69,.025); }
    .metric-top { display:flex; align-items:center; justify-content:space-between; color:#7e899e; font-size:12px; font-weight:600; }
    .metric-icon { display:grid; place-items:center; width:30px; height:30px; border-radius:10px; background:#f0efff; color:#594fe2; font-size:15px; }
    .metric-value { font:800 25px 'Aptos Display','Segoe UI',sans-serif; color:#17233b; margin-top:8px; line-height:1.1; }
    .metric-foot { color:#9aa3b4; font-size:11px; margin-top:4px; }
    .panel { border:1px solid var(--line); border-radius:18px; background:#fff; padding:18px 20px; box-shadow:0 3px 12px rgba(28,39,69,.025); }
    .panel-title { font:700 15px 'Aptos Display','Segoe UI',sans-serif; color:#17233b; margin:0 0 4px; }
    .panel-sub { font-size:12px; color:#8a94a7; margin:0 0 14px; }
    .status-pill { display:inline-flex; align-items:center; gap:7px; padding:7px 11px; border-radius:99px; font-size:11px; font-weight:700; color:#11866e; background:#e9f8f3; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:#18ad88; }
    div[data-testid="stTabs"] button { font-weight:650; color:#758097; border-radius:9px 9px 0 0; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#5146e5; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#5146e5; }
    div.stButton > button[kind="primary"] { border:0; border-radius:11px; color:#fff !important; background:linear-gradient(110deg,#5b50e8,#483ed2); font-weight:700; box-shadow:0 6px 15px #5146e52b; }
    div.stButton > button[kind="primary"] *,
    div.stButton > button[kind="primary"] p { color:#fff !important; }
    div.stButton > button:not([kind="primary"]) { border:1px solid #e4e8f0; border-radius:10px; color:#344057 !important; background:#fff; font-weight:600; }
    div.stButton > button:not([kind="primary"]) * { color:#344057 !important; }
    div.stButton > button:not([kind="primary"]):hover { border-color:#c8c5ff; color:#5146e5; background:#f9f8ff; }
    div[data-baseweb="select"] > div { background:#fff !important; border-color:#e4e8f0; color:#344057 !important; border-radius:10px; }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input { color:#26344d !important; }
    [data-testid="stDownloadButton"] button { border-radius:10px; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
    .calendar-wrap { overflow-x:auto; background:#fff; border:1px solid #e6e9f0; border-radius:17px; box-shadow:0 4px 16px rgba(30,40,70,.035); }
    .calendar-grid { min-width:940px; display:grid; grid-template-columns:66px repeat(5,minmax(150px,1fr)); position:relative; background:white; }
    .cal-corner { position:sticky; left:0; z-index:4; background:#fafbfe; border-bottom:1px solid #edf0f5; }
    .cal-day { padding:13px 10px 11px; text-align:center; border-left:1px solid #edf0f5; border-bottom:1px solid #e8ebf2; background:#fafbfe; }
    .cal-day-name { font-size:12px; font-weight:800; letter-spacing:.04em; color:#35415b; }
    .cal-day-date { color:#9aa3b2; font-size:11px; margin-top:3px; }
    .cal-time { position:sticky; left:0; z-index:2; height:72px; padding:8px 8px 0 0; text-align:right; color:#929bad; background:#fff; border-bottom:1px solid #f0f2f6; font-size:11px; }
    .cal-cell { height:72px; border-left:1px solid #edf0f5; border-bottom:1px solid #f0f2f6; }
    .cal-event { z-index:3; min-width:0; margin:4px 4px; padding:8px 8px 6px 10px; border-left:3px solid var(--event-color); border-radius:8px; background:var(--event-bg); overflow:hidden; color:#23304b; }
    .cal-event-title { font-size:11px; line-height:1.3; font-weight:800; overflow-wrap:anywhere; }
    .cal-event.lab { border-left-style:double; border-left-width:5px; }
    .cal-event-time { margin-top:4px; color:#59677f; font-size:10px; font-weight:600; }
    .cal-event-meta { margin-top:3px; color:#78849a; font-size:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .empty-day { display:flex; align-items:center; justify-content:center; height:100%; color:#c0c6d1; font-size:10px; }
    .legend-item { display:inline-flex; align-items:center; gap:6px; margin-right:14px; color:#738097; font-size:11px; }
    .legend-dot { width:8px; height:8px; border-radius:3px; }
    .soft-note { padding:12px 14px; border-radius:12px; color:#665b9c; background:#f4f1ff; font-size:12px; line-height:1.55; }
    /* Streamlit's active theme can override text colors while the app uses a light canvas. */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stMetricLabel"],
    [data-testid="stMain"] [data-testid="stMetricLabel"] *,
    [data-testid="stMain"] [data-testid="stMetricValue"],
    [data-testid="stMain"] [data-testid="stMetricValue"] *,
    [data-testid="stMain"] [data-testid="stMetricDelta"],
    [data-testid="stMain"] [data-testid="stMetricDelta"] * {
      color:#26344d !important;
      opacity:1 !important;
    }
    [data-testid="stMain"] [data-testid="stMetric"] {
      min-height:104px;
      padding:15px 17px;
      border:1px solid #e1e6ef;
      border-radius:15px;
      background:#fff;
      box-shadow:0 3px 12px rgba(28,39,69,.035);
    }
    [data-testid="stMain"] [data-testid="stMetricLabel"] {
      color:#59667d !important;
      font-size:13px;
      font-weight:650;
    }
    [data-testid="stMain"] [data-testid="stMetricValue"] {
      color:#17233b !important;
      font-size:30px;
      font-weight:800;
    }
    [data-testid="stMain"] [data-testid="stMetricDelta"] { color:#526078 !important; }
    div[data-testid="stTabs"] [role="tablist"] { gap:5px; }
    div[data-testid="stTabs"] button {
      min-height:44px;
      padding:7px 9px;
      color:#46536a !important;
      font-size:12px;
      font-weight:700;
      opacity:1 !important;
    }
    div[data-testid="stTabs"] button *,
    div[data-testid="stTabs"] button p { color:inherit !important; opacity:1 !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
      color:#3930a8 !important;
      background:#efedff;
    }
    .section-label { color:#59667d; font-size:11px; }
    .metric-top { color:#59667d; }
    .metric-foot, .panel-sub { color:#59667d; }
    .hero-sub { color:#59667d; }
    .cal-day-date, .cal-time, .cal-event-meta { color:#59667d; }
    .legend-item { color:#59667d; }
    :root {
      color-scheme: light;
      --ink:#17233b;
      --muted:#59667d;
      --line:#e1e6ef;
      --paper:#ffffff;
      --canvas:#f5f7fb;
      --accent:#5146e5;
      --accent-soft:#efedff;
      --control:#ffffff;
    }
    body .stApp { background:var(--canvas) !important; color:var(--ink); }
    [data-testid="stMainBlockContainer"] { color:var(--ink); }
    [data-testid="stSidebar"] { background:var(--paper) !important; border-color:var(--line); }
    [data-testid="stHeader"] { background:var(--canvas) !important; }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stMetricLabel"],
    [data-testid="stMain"] [data-testid="stMetricLabel"] *,
    [data-testid="stMain"] [data-testid="stMetricValue"],
    [data-testid="stMain"] [data-testid="stMetricValue"] *,
    [data-testid="stMain"] [data-testid="stMetricDelta"] {
      color:var(--ink) !important;
      opacity:1 !important;
    }
    [data-testid="stMain"] [data-testid="stMetric"] {
      background:var(--paper) !important;
      border:1px solid var(--line);
      border-radius:14px;
      padding:14px 16px;
    }
    [data-testid="stMain"] [data-testid="stMetricLabel"] { color:var(--muted) !important; }
    [data-testid="stMain"] [data-testid="stMetricValue"] { color:var(--ink) !important; }
    div[data-testid="stTabs"] button { color:var(--muted) !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
      color:var(--accent) !important;
      background:var(--accent-soft);
    }
    div[data-testid="stTabs"] button *,
    div[data-testid="stTabs"] button p { color:inherit !important; }
    [data-testid="stMain"] div[data-baseweb="select"] > div,
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="input"] > div {
      color:var(--ink) !important;
      background:var(--control) !important;
      border-color:var(--line) !important;
    }
    [data-testid="stMain"] div[data-baseweb="select"] span,
    [data-testid="stMain"] div[data-baseweb="select"] input { color:var(--ink) !important; }
    div.stButton > button:not([kind="primary"]) {
      color:var(--ink) !important;
      background:var(--paper) !important;
      border-color:var(--line) !important;
    }
    div.stButton > button:not([kind="primary"]) *,
    div.stButton > button:not([kind="primary"]) p { color:var(--ink) !important; }
    div.stButton > button[kind="primary"] { color:#fff !important; }
    div.stButton > button[kind="primary"] * { color:#fff !important; }
    .metric-card, .panel, .calendar-wrap { color:var(--ink); background:var(--paper); border-color:var(--line); }
    .metric-card { background:var(--paper); border-color:var(--line); }
    .metric-value, .panel-title, .cal-day-name { color:var(--ink); }
    .metric-top, .metric-foot, .panel-sub, .hero-sub, .cal-time, .cal-event-meta,
    .cal-day-date, .legend-item { color:var(--muted); }
    [data-testid="stDataFrame"] { border-color:var(--line) !important; }
    [data-baseweb="popover"] [role="option"] { color:var(--ink) !important; background:var(--paper); }
    .hero {
      color:var(--ink);
      background:linear-gradient(115deg,var(--paper),var(--accent-soft));
      border-color:var(--line);
    }
    .hero-title, .brand-name { color:var(--ink); }
    .hero-sub { color:var(--muted); }
    @media (max-width: 700px) {
      [data-testid="stMainBlockContainer"] { padding:1rem .75rem 2rem; }
      .hero { padding:20px; } .hero-title { font-size:22px; }
      .hero-badge { position:static; display:inline-block; margin-top:13px; }
      div[data-testid="stTabs"] button { min-height:48px; padding:7px 9px; font-size:12px; }
      [data-testid="stMain"] [data-testid="stMetric"] { min-height:92px; padding:12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if THEME == "Dark":
    st.markdown(
        """
        <style>
        :root {
          color-scheme:dark;
          --ink:#eef2ff;
          --muted:#b3bfd4;
          --line:#34415a;
          --paper:#1b2639;
          --canvas:#111827;
          --accent:#a9a0ff;
          --accent-soft:#302d54;
          --control:#202c40;
        }
        [data-testid="stSidebar"] { border-right:1px solid var(--line) !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stRadio"] label * {
          color:var(--ink) !important;
          opacity:1 !important;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stMain"] [data-testid="stCaptionContainer"],
        [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
        [data-testid="stMain"] [data-testid="stMetricLabel"],
        [data-testid="stMain"] [data-testid="stMetricValue"] { color:var(--ink) !important; }
        [data-testid="stMain"] [data-testid="stMetricDelta"] { color:var(--muted) !important; }
        .hero { border-color:var(--line); background:linear-gradient(115deg,#1b2639,#282744); }
        .hero-kicker, .hero-badge { color:#c2bbff; }
        .hero-badge, .metric-icon { background:#302d54; border-color:#4a447c; }
        .section-label { color:var(--muted); }
        .metric-card, .panel, .calendar-wrap { background:var(--paper); border-color:var(--line); }
        .calendar-grid, .cal-corner, .cal-day { background:var(--paper); }
        .cal-time, .cal-cell { background:var(--paper); border-color:var(--line); }
        .cal-event { color:#f3f5ff; }
        .cal-event-time { color:#d5dcf0; }
        .soft-note { color:#d7d0ff; background:#302d54; }
        .status-pill { color:#8ee0c8; background:#173b38; }
        div.stButton > button:not([kind="primary"]):hover { color:var(--ink) !important; background:#29364d !important; }
        [data-testid="stHeader"] { background:var(--canvas) !important; }
        [data-testid="stMain"] [data-baseweb="tag"] { color:var(--ink); background:#302d54; }
        [data-testid="stMain"] [data-baseweb="popover"] { background:var(--paper); }
        [data-testid="stMain"] [data-baseweb="popover"] * { color:var(--ink); }
        [data-testid="stMain"] [data-testid="stExpander"] summary,
        [data-testid="stMain"] [data-testid="stExpander"] summary * { color:var(--ink) !important; }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary * { color:var(--ink) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot: str, icon: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-top"><span>{html.escape(label)}</span>'
        f'<span class="metric-icon">{icon}</span></div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-foot">{html.escape(foot)}</div></div>'
    )


def render_table(data: pd.DataFrame, *args, **kwargs) -> None:
    """Render a dataframe with explicit palette colors inside Streamlit's table frame."""
    data = data.copy()
    active_problem = st.session_state.get("problem")
    if active_problem and "teacher_id" in data.columns:
        data["teacher_id"] = data["teacher_id"].map(
            lambda teacher_id: (
                f"{active_problem.teachers[teacher_id].name} ({teacher_id})"
                if teacher_id in active_problem.teachers
                else teacher_id
            )
        )
        data = data.rename(columns={"teacher_id": "Instructor"})
    if active_problem and "room_id" in data.columns:
        data["room_id"] = data["room_id"].map(
            lambda room_id: (
                f"{room_id} · {active_problem.rooms[room_id].room_type} · "
                f"{active_problem.rooms[room_id].capacity} seats"
                if room_id in active_problem.rooms
                else room_id
            )
        )
        data = data.rename(columns={"room_id": "Classroom"})
    header_background = "#302d54" if THEME == "Dark" else "#e9eaff"
    header_color = "#f3f2ff" if THEME == "Dark" else "#26344d"
    cell_background = "#1b2639" if THEME == "Dark" else "#ffffff"
    cell_color = "#eef2ff" if THEME == "Dark" else "#26344d"
    alternate_background = "#202c40" if THEME == "Dark" else "#f5f7fb"
    styled = data.style.set_properties(
        **{
            "background-color": cell_background,
            "color": cell_color,
            "font-size": "13px",
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", header_background),
                    ("color", header_color),
                    ("font-weight", "700"),
                    ("border-color", "#34415a" if THEME == "Dark" else "#e1e6ef"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even) td",
                "props": [
                    ("background-color", alternate_background),
                    ("color", cell_color),
                ],
            },
            {
                "selector": "td",
                "props": [("border-color", "#34415a" if THEME == "Dark" else "#e1e6ef")],
            },
        ]
    )
    st.dataframe(styled, *args, **kwargs)


@st.cache_resource
def quick_demo_problem() -> Problem:
    data_root = ROOT / "demo_data" / "quick"
    return load_dataset(
        {
            name: data_root / f"{name}.csv"
            for name in ("courses", "teachers", "rooms", "groups")
        }
    )


def render_calendar(
    problem: Problem,
    schedule: dict[str, Placement],
    rows: list[dict[str, object]],
) -> None:
    meeting_by_id = {meeting.meeting_id: meeting for meeting in problem.meetings}
    filtered_ids = {str(row["meeting_id"]) for row in rows}
    columns = len(problem.days) + 1
    html_parts = [
        '<div class="calendar-wrap"><div class="calendar-grid" style="grid-template-columns:66px repeat('
        f'{len(problem.days)},minmax(150px,1fr));grid-template-rows:48px repeat({problem.slots_per_day},72px)">',
        '<div class="cal-corner" style="grid-column:1;grid-row:1"></div>',
    ]
    for day, day_name in enumerate(problem.days):
        html_parts.append(
            f'<div class="cal-day" style="grid-column:{day + 2};grid-row:1">'
            f'<div class="cal-day-name">{html.escape(day_name.upper())}</div>'
            f'<div class="cal-day-date">SCHEDULE</div></div>'
        )
    for slot in range(problem.slots_per_day):
        grid_row = slot + 2
        hour = 8 + slot
        html_parts.append(
            f'<div class="cal-time" style="grid-column:1;grid-row:{grid_row}">'
            f'{hour:02d}:00</div>'
        )
        for day in range(len(problem.days)):
            html_parts.append(
                f'<div class="cal-cell" style="grid-column:{day + 2};grid-row:{grid_row}"></div>'
            )
    for meeting_id in filtered_ids:
        meeting = meeting_by_id[meeting_id]
        placement = schedule[meeting_id]
        color_hash = hashlib.sha1(meeting.subject.encode("utf-8")).digest()[0]
        color = COLORS[color_hash % len(COLORS)]
        background = f"{color}18"
        course = html.escape(
            meeting.course_name + (" · LAB" if meeting.room_type == "lab" else "")
        )
        time_range = f"{8 + placement.start:02d}:00–{8 + placement.start + meeting.duration:02d}:00"
        teacher_name = problem.teachers[meeting.teacher_id].name
        location = f'{teacher_name} · {placement.room_id} · {", ".join(meeting.groups)}'
        title = f"{meeting.subject} · {teacher_name} · {meeting.room_type}"
        html_parts.append(
            f'<div class="cal-event{" lab" if meeting.room_type == "lab" else ""}" title="{html.escape(title)}" '
            f'style="grid-column:{placement.day + 2};grid-row:{placement.start + 2} / span {meeting.duration};'
            f'--event-color:{color};--event-bg:{background}">'
            f'<div class="cal-event-title">{course}</div>'
            f'<div class="cal-event-time">{time_range}</div>'
            f'<div class="cal-event-meta">{html.escape(location)}</div></div>'
        )
    if not filtered_ids:
        html_parts.append(
            f'<div style="grid-column:2 / {columns + 1};grid-row:2 / span {problem.slots_per_day};'
            'display:grid;place-items:center;color:#8b95a8;font-size:13px">'
            "No classes match these filters.</div>"
        )
    html_parts.append("</div></div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def load_problem(quick: bool = False) -> None:
    data_root = ROOT / "demo_data" / "quick" if quick else ROOT / "demo_data"
    paths = {name: data_root / f"{name}.csv" for name in ("courses", "teachers", "rooms", "groups")}
    st.session_state["problem"] = load_dataset(paths)
    st.session_state["ml_model"] = create_preference_model()
    st.session_state["dataset_name"] = "Quick demo" if quick else "Campus demo · synthetic"
    st.session_state["uploaded_tables"] = None
    st.session_state["table_issues"] = {}
    for key in (
        "schedule", "baseline_schedule", "optimization_result", "comparison",
        "comparison_traces", "home_demo_result", "pending_repair",
        "ml_signal_comparison",
    ):
        st.session_state.pop(key, None)


if "problem" not in st.session_state:
    load_problem()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">◷</div><div>'
        '<div class="brand-name">IntelliSchedule AI</div><div class="brand-sub">Timetable planner</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.radio(
        "Appearance",
        ("Light", "Dark"),
        horizontal=True,
        key="theme_mode",
        help="Choose a high-contrast color palette. Your choice is retained while using the app.",
    )
    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    source = st.radio(
        "Dataset source",
        ("Demo campus", "Quick demo", "Upload my data"),
        label_visibility="collapsed",
    )
    if source == "Upload my data":
        uploads = st.file_uploader(
            "Add your four data files",
            type=("csv", "xlsx", "xlsm"),
            accept_multiple_files=True,
            help="Required: courses, teachers, rooms, groups. Optional: attendance, enrollment, preference_feedback.",
        )
        if st.button("Import dataset", use_container_width=True, disabled=not uploads):
            tables = {}
            try:
                tables = read_uploaded_tables(uploads or [])
                table_issues = validate_tables(tables)
                st.session_state["table_issues"] = table_issues
                imported = build_problem({
                    name: rows
                    for name, rows in tables.items()
                    if name in {"courses", "teachers", "rooms", "groups"}
                })
                attendance_records = tables.get("attendance")
                enrollment_records = tables.get("enrollment")
                imported_model = create_preference_model(
                    attendance_records=attendance_records,
                    enrollment_records=enrollment_records,
                )
                if tables.get("preference_feedback"):
                    imported_model.fit_preference_feedback(tables["preference_feedback"])
                st.session_state["problem"] = imported
                st.session_state["ml_model"] = imported_model
                st.session_state["dataset_name"] = "Uploaded dataset"
                st.session_state["uploaded_tables"] = tables
                for key in (
                    "schedule", "baseline_schedule", "optimization_result", "comparison",
                    "comparison_traces", "home_demo_result", "pending_repair",
                    "ml_signal_comparison",
                ):
                    st.session_state.pop(key, None)
                RunStore(ROOT / "timetable.db").save_dataset("Uploaded dataset", tables)
                if table_issues:
                    st.warning("Imported with data-quality warnings; review them in Data Management.")
                else:
                    st.success(f"Imported {len(imported.meetings)} meetings.")
            except (ValueError, RuntimeError) as exc:
                issues = st.session_state.get("table_issues", {})
                category = "ML history" if any(
                    key in tables for key in ("attendance", "enrollment", "preference_feedback")
                ) else "invalid_constraints"
                issues.setdefault(category, []).append(str(exc))
                st.session_state["table_issues"] = issues
                st.error(str(exc))
    else:
        quick = source == "Quick demo"
        st.markdown(
            '<div class="soft-note">Quick demo · 8 sections</div>' if quick else
            '<div class="soft-note">Demo campus · 110 sections · 35 faculty · 10 cohorts</div>',
            unsafe_allow_html=True,
        )
        if st.button("Load selected demo", use_container_width=True):
            load_problem(quick=quick)
            st.session_state.pop("schedule", None)
            st.session_state.pop("baseline_schedule", None)
            st.rerun()

    st.markdown('<div class="section-label">Generation settings</div>', unsafe_allow_html=True)
    optimizer_name = st.selectbox("Optimization method", ("Simulated Annealing", "Tabu Search", "Hill Climbing"))
    iterations = st.slider("Search iterations", 100, 2000, 600, 100)
    with st.expander("Simulated Annealing controls"):
        initial_temperature = st.slider("Initial temperature", 1.0, 20.0, 8.0, 0.5)
        cooling_rate = st.slider("Cooling rate", 0.90, 0.999, 0.99, 0.001, format="%.3f")
        adaptive_neighborhood = st.checkbox("Adapt search neighborhood", value=True)
        reheat_after = st.slider("Reheat after stagnant moves", 20, 200, 80, 10)
        early_stopping_rounds = st.slider("Early-stop stagnation limit", 100, 1000, 500, 50)
    st.markdown("---")
    st.caption("Constraint engine · ML-assisted preferences")
    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span>Ready to schedule</div>',
        unsafe_allow_html=True,
    )

problem: Problem = st.session_state["problem"]
model = st.session_state["ml_model"]
meetings_count = len(problem.meetings)
section_count = len({meeting.course_id for meeting in problem.meetings})
schedule_exists = "schedule" in st.session_state

st.markdown(
    '<div class="hero"><div class="hero-kicker">AI-Driven University Timetable Optimization</div>'
    '<div class="hero-title">IntelliSchedule AI</div>'
    '<p class="hero-sub"><strong>Smarter Scheduling. Better Academic Planning.</strong><br>'
    "An intelligent university scheduling platform that combines constraint-based AI, machine learning, "
    "and advanced optimization to generate efficient academic timetables.</p>"
    f'<div class="hero-badge">✦ &nbsp;{html.escape(st.session_state.get("dataset_name", "Campus demo"))}</div>'
    "</div>",
    unsafe_allow_html=True,
)

(
    overview_tab,
    generation_tab,
    insights_tab,
    ml_tab,
    reschedule_tab,
    timetable_tab,
    analytics_tab,
    manage_tab,
) = st.tabs(
    [
        "✦ Overview",
        "◉ Generate",
        "↗ Optimize",
        "◎ ML Insights",
        "⟳ Repair",
        "▦ Timetable",
        "◫ Analytics",
        "⚙ Data",
    ]
)

with overview_tab:
    home_actions = st.columns([1, 1, 3])
    home_generate = home_actions[0].button(
        "Regenerate timetable" if schedule_exists else "Generate timetable",
        type="primary",
        use_container_width=True,
        key="home_generate",
    )
    explore_capabilities = home_actions[1].button(
        "Explore AI capabilities",
        use_container_width=True,
        key="home_explore_ai",
    )
    if explore_capabilities:
        st.session_state["show_methodology"] = not st.session_state.get("show_methodology", False)
    status_text = "Schedule ready" if schedule_exists else "Dataset ready to schedule"
    metric_html = (
        '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px">'
        + metric_card("Course sections", f"{section_count}", "Across all departments", "▤")
        + metric_card("Weekly sessions", f"{meetings_count}", "Meetings to place", "◷")
        + metric_card("Teaching staff", f"{len(problem.teachers)}", "Available faculty", "♙")
        + metric_card("Rooms", f"{len(problem.rooms)}", status_text, "⌂")
        + "</div>"
    )
    st.markdown(metric_html, unsafe_allow_html=True)
    if schedule_exists:
        live_metrics = schedule_metrics(problem, st.session_state["schedule"])
        live_errors = validate_schedule(problem, st.session_state["schedule"])
        model_r2 = model.attendance_metrics["r2"]
        row_one = st.columns(4)
        row_one[0].metric("Hard constraints", f"{hard_constraint_count(problem):,}")
        row_one[1].metric("Current violations", len(live_errors))
        row_one[2].metric(
            "Optimization score",
            f'{soft_score(objective_cost(problem, st.session_state["schedule"], model), len(problem.meetings)):.1f}',
        )
        row_one[3].metric("Teacher fairness", f'{live_metrics["teacher_fairness_pct"]:.1f}%')
        row_two = st.columns(4)
        row_two[0].metric("Student idle periods", int(live_metrics["student_idle_periods"]))
        row_two[1].metric("Classroom utilization", f'{live_metrics["classroom_utilization_pct"]:.1f}%')
        row_two[2].metric("Attendance model R²", f"{model_r2:.3f}", "temporal holdout")
        row_two[3].metric(
            "Generation time",
            f'{st.session_state.get("generation", {}).get("runtime", 0):.2f}s',
        )
        row_three = st.columns(3)
        row_three[0].metric("Teacher idle periods", int(live_metrics["teacher_idle_periods"]))
        row_three[1].metric(
            "Teacher preferences",
            (
                f'{live_metrics["teacher_preference_satisfaction_pct"]:.1f}%'
                if live_metrics["teacher_preference_periods"]
                else "Not supplied"
            ),
            "matched preferred periods" if live_metrics["teacher_preference_periods"] else "no preference data",
        )
        row_three[2].metric(
            "Modeled constraints",
            f"{hard_constraint_count(problem):,}",
            "unary + unique pairwise",
        )
    st.write("")

    if schedule_exists:
        current_errors = validate_schedule(problem, st.session_state["schedule"])
        if current_errors:
            st.error(f"{len(current_errors)} hard conflicts need attention.")
        else:
            st.markdown(
                '<div class="status-pill"><span class="status-dot"></span> '
                "Conflict-free schedule · ready to explore</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Your course data is loaded. Generate a schedule to unlock the week view and reports.")
    if home_generate:
        with st.spinner("Finding a feasible timetable, then balancing the week…"):
            started = time.perf_counter()
            try:
                seed_schedule, solver_stats = solve_csp(problem, return_stats=True)
                method = {
                    "Simulated Annealing": "annealing",
                    "Tabu Search": "tabu",
                    "Hill Climbing": "hill-climbing",
                }[optimizer_name]
                result = optimize_detailed(
                    problem,
                    seed_schedule,
                    method,
                    model,
                    iterations=iterations,
                    initial_temperature=initial_temperature,
                    cooling_rate=cooling_rate,
                    adaptive_neighborhood=adaptive_neighborhood,
                    reheat_after=reheat_after,
                    early_stopping_rounds=early_stopping_rounds,
                )
                schedule = result.schedule
                cost = result.best_cost
                elapsed = time.perf_counter() - started
                errors = validate_schedule(problem, schedule)
                if errors:
                    raise RuntimeError("Generated schedule failed validation: " + "; ".join(errors[:5]))
                st.session_state["schedule"] = schedule
                st.session_state["baseline_schedule"] = dict(seed_schedule)
                st.session_state["optimization_result"] = result
                st.session_state["generation"] = {
                    "algorithm": f"CSP + {optimizer_name}",
                    "cost": cost,
                    "runtime": elapsed,
                    "nodes": solver_stats.nodes,
                    "solver_stats": solver_stats,
                }
                RunStore(ROOT / "timetable.db").save(
                    f"CSP + {optimizer_name}", cost, elapsed, schedule
                )
                st.success(
                    f"Your timetable is ready in {elapsed:.1f}s. "
                    f"All hard constraints passed ({solver_stats.nodes:,} CSP search steps)."
                )
                RunStore(ROOT / "timetable.db").save_experiment(
                    result.method,
                    result.initial_cost,
                    result.best_cost,
                    result.runtime_seconds,
                    result.iterations,
                    result.accepted_moves,
                    schedule_metrics(problem, schedule),
                )
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

    st.write("")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">Your scheduling workspace</div>'
            '<div class="panel-sub">A quick look at the data behind this timetable.</div></div>',
            unsafe_allow_html=True,
        )
        group_counts: dict[str, int] = {}
        for meeting in problem.meetings:
            for group_id in meeting.groups:
                group_counts[group_id] = group_counts.get(group_id, 0) + meeting.duration
        load_frame = pd.DataFrame(
            [{"Cohort": group_id, "Contact hours": hours} for group_id, hours in sorted(group_counts.items())]
        )
        chart = px.bar(
            load_frame,
            x="Cohort",
            y="Contact hours",
            color="Contact hours",
            color_continuous_scale=["#c9c6ff", "#5146e5"],
            template=CHART_TEMPLATE,
        )
        chart.update_layout(
            height=235,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            xaxis_title=None,
            yaxis_title=None,
            font=dict(family="Segoe UI", color=CHART_TEXT, size=11),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        chart.update_traces(marker_line_width=0, marker_cornerradius=5)
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">How your timetable is built</div>'
            '<div class="panel-sub">Constraint-first, preference-aware scheduling.</div>'
            '<div style="display:grid;gap:12px">'
            '<div><span class="metric-icon">01</span> &nbsp;<b>Find a feasible week</b>'
            '<div class="panel-sub" style="margin:5px 0 0 42px">Place courses without clashes, respecting rooms and availability.</div></div>'
            '<div><span class="metric-icon">02</span> &nbsp;<b>Balance the schedule</b>'
            '<div class="panel-sub" style="margin:5px 0 0 42px">Reduce gaps and use faculty preferences to improve the week.</div></div>'
            '<div><span class="metric-icon">03</span> &nbsp;<b>Explore and adjust</b>'
            '<div class="panel-sub" style="margin:5px 0 0 42px">Browse by cohort, teacher, or room; repair disruptions.</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown("### From university data to a feasible week")
    workflow_cols = st.columns(5)
    for column, step, title in zip(
        workflow_cols,
        ("01", "02", "03", "04", "05"),
        ("University data", "CSP search", "AI optimization", "ML insights", "Final timetable"),
    ):
        column.markdown(
            f'<div class="panel"><div class="metric-icon">{step}</div>'
            f'<div class="panel-title" style="margin-top:10px">{title}</div></div>',
            unsafe_allow_html=True,
        )
    try:
        saved_runs = RunStore(ROOT / "timetable.db").recent(limit=5)
        if saved_runs:
            st.markdown("#### Recent measured timetable runs")
            render_table(
                pd.DataFrame(saved_runs)[
                    ["created_at", "algorithm", "soft_cost", "runtime_seconds"]
                ],
                hide_index=True,
                use_container_width=True,
            )
    except (OSError, sqlite3.Error) as exc:
        st.warning(f"Saved experiment preview is unavailable: {exc}")
    with st.expander("Interactive AI scheduling demonstration"):
        st.caption(
            "Runs the real CSP and Simulated Annealing on the included 8-section sample. "
            "No fabricated progress or performance figures; the original campus schedule is not changed."
        )
        try:
            demo_problem = quick_demo_problem()
            demo_meetings = demo_problem.meetings
            departments = sorted({meeting.department for meeting in demo_meetings})
            selected_department = st.selectbox(
                "Sample department",
                departments,
                key="home_demo_department",
            )
            department_courses = {
                meeting.course_id: meeting.course_name
                for meeting in demo_meetings
                if meeting.department == selected_department
            }
            selected_course = st.selectbox(
                "Section",
                list(department_courses),
                format_func=lambda course_id: department_courses[course_id],
                key="home_demo_section",
            )
            if st.button(
                "Run CSP + Simulated Annealing",
                type="primary",
                key="run_home_demo",
            ):
                with st.spinner("Solving the sample timetable with the real CSP and optimizer…"):
                    demo_started = time.perf_counter()
                    seed_schedule, demo_stats = solve_csp(
                        demo_problem, seed=42, return_stats=True
                    )
                    demo_optimization = optimize_detailed(
                        demo_problem,
                        seed_schedule,
                        "annealing",
                        model,
                        iterations=160,
                        seed=42,
                        max_runtime_seconds=5,
                    )
                    st.session_state["home_demo_result"] = {
                        "seed_schedule": seed_schedule,
                        "schedule": demo_optimization.schedule,
                        "result": demo_optimization,
                        "solver_nodes": demo_stats.nodes,
                        "runtime": time.perf_counter() - demo_started,
                        "department": selected_department,
                        "course_id": selected_course,
                    }
            demo_result = st.session_state.get("home_demo_result")
            if demo_result and (
                demo_result["department"] != selected_department
                or demo_result["course_id"] != selected_course
            ):
                demo_result = None
            if demo_result:
                before = demo_result["seed_schedule"]
                after = demo_result["schedule"]
                before_metrics = schedule_metrics(demo_problem, before)
                after_metrics = schedule_metrics(demo_problem, after)
                optimization_result = demo_result["result"]
                demo_metrics = st.columns(4)
                demo_metrics[0].metric(
                    "Initial cost",
                    f'{optimization_result.initial_cost:.2f}',
                )
                demo_metrics[1].metric(
                    "Final cost",
                    f'{optimization_result.best_cost:.2f}',
                    f'{optimization_result.initial_cost - optimization_result.best_cost:+.2f}',
                )
                demo_metrics[2].metric(
                    "Student idle periods",
                    int(after_metrics["student_idle_periods"]),
                    f'{int(before_metrics["student_idle_periods"]) - int(after_metrics["student_idle_periods"]):+d} vs CSP',
                )
                demo_metrics[3].metric(
                    "Hard violations",
                    len(validate_schedule(demo_problem, after)),
                    help="Computed by the existing hard-constraint validator.",
                )
                demo_metrics = st.columns(4)
                demo_metrics[0].metric(
                    "Teacher idle periods",
                    int(after_metrics["teacher_idle_periods"]),
                    f'{int(before_metrics["teacher_idle_periods"]) - int(after_metrics["teacher_idle_periods"]):+d} vs CSP',
                )
                demo_metrics[1].metric(
                    "Teacher preference match",
                    f'{after_metrics["teacher_preference_satisfaction_pct"]:.1f}%',
                )
                demo_metrics[2].metric(
                    "Optimizer runtime",
                    f'{optimization_result.runtime_seconds:.3f}s',
                )
                demo_metrics[3].metric(
                    "CSP search nodes",
                    f'{demo_result["solver_nodes"]:,}',
                )
                if optimization_result.best_cost < optimization_result.initial_cost:
                    st.success(
                        f"Measured objective improved by "
                        f"{optimization_result.initial_cost - optimization_result.best_cost:.2f}."
                    )
                else:
                    st.info("This run did not improve the seed objective; the measured result is unchanged.")
                sample_meeting_ids = {
                    meeting.meeting_id
                    for meeting in demo_meetings
                    if meeting.course_id == demo_result["course_id"]
                }
                comparison_rows = []
                meeting_map = {meeting.meeting_id: meeting for meeting in demo_meetings}
                for meeting_id in sorted(sample_meeting_ids):
                    meeting = meeting_map[meeting_id]
                    old, new = before[meeting_id], after[meeting_id]
                    comparison_rows.append(
                        {
                            "Section": meeting.course_name,
                            "Groups": ", ".join(meeting.groups),
                            "CSP placement": (
                                f"{demo_problem.days[old.day]} {8 + old.start:02d}:00 · {old.room_id}"
                            ),
                            "Optimized placement": (
                                f"{demo_problem.days[new.day]} {8 + new.start:02d}:00 · {new.room_id}"
                            ),
                            "Changed": old != new,
                        }
                    )
                st.markdown("**Selected section: CSP seed vs optimized timetable**")
                render_table(
                    pd.DataFrame(comparison_rows),
                    hide_index=True,
                    use_container_width=True,
                )
                moved_rows = []
                for meeting_id, new in after.items():
                    old = before[meeting_id]
                    if old != new:
                        meeting = meeting_map[meeting_id]
                        moved_rows.append(
                            {
                                "Class": meeting.course_name,
                                "Groups": ", ".join(meeting.groups),
                                "From": f"{demo_problem.days[old.day]} {8 + old.start:02d}:00 · {old.room_id}",
                                "To": f"{demo_problem.days[new.day]} {8 + new.start:02d}:00 · {new.room_id}",
                            }
                        )
                if moved_rows:
                    st.markdown("**Classes moved by the optimizer**")
                    render_table(pd.DataFrame(moved_rows), hide_index=True, use_container_width=True)
                else:
                    st.caption("The optimizer made no placement changes in this run.")
            elif st.session_state.get("home_demo_result"):
                st.caption("Select Run again to compute results for this department and section.")
        except (ValueError, RuntimeError) as exc:
            st.error(f"The interactive sample could not be solved: {exc}")

    with st.expander("AI methodology and guided presentation", expanded=st.session_state.get("show_methodology", False)):
        st.markdown("**Unit 2 — AI Search Techniques**")
        st.markdown(
            "- **Constraint Satisfaction:** Inputs are classes, rooms, staff, and groups; "
            "backtracking with MRV, degree tie-breaks, and forward checking produces a feasible seed.\n"
            "- **Simulated Annealing:** Starts from a feasible schedule and samples feasible class placements "
            "to reduce the weighted soft-constraint cost.\n"
            "- **Tabu Search:** Compares feasible alternatives while using short-term move memory to avoid cycles.\n"
            "- **Hill Climbing:** Selects improving feasible moves and stops at a sampled local optimum.\n"
            "- **Means–Ends Analysis:** Compares a disruption's desired state with the current schedule and "
            "uses bounded CSP repair to minimize changed placements."
        )
        comparison_by_algorithm = {
            row["Algorithm"]: row for row in st.session_state.get("comparison", [])
        }
        live_demo = st.session_state.get("home_demo_result")
        sa_measurement = (
            f'Cost {live_demo["result"].initial_cost:.2f} → '
            f'{live_demo["result"].best_cost:.2f}; '
            f'{live_demo["result"].runtime_seconds:.3f}s'
            if live_demo
            else (
                f'Cost {comparison_by_algorithm["Simulated Annealing"]["Initial cost"]:.2f} → '
                f'{comparison_by_algorithm["Simulated Annealing"]["Soft cost"]:.2f}'
                if "Simulated Annealing" in comparison_by_algorithm
                else "Run the demo or comparison to measure"
            )
        )
        methodology = pd.DataFrame(
            [
                {
                    "Method": "CSP backtracking",
                    "Purpose": "Build a hard-feasible seed",
                    "Inputs → outputs": "Classes/resources → valid assignments",
                    "Measured contribution": (
                        f'{live_demo["solver_nodes"]:,} search nodes in live sample'
                        if live_demo else "Run live sample to measure"
                    ),
                },
                {
                    "Method": "Simulated Annealing",
                    "Purpose": "Improve weighted soft quality",
                    "Inputs → outputs": "Feasible seed → lower-cost schedule",
                    "Measured contribution": sa_measurement,
                },
                {
                    "Method": "Tabu Search",
                    "Purpose": "Comparison optimizer with move memory",
                    "Inputs → outputs": "Same CSP seed → feasible schedule",
                    "Measured contribution": (
                        f'Cost {comparison_by_algorithm["Tabu Search"]["Initial cost"]:.2f} → '
                        f'{comparison_by_algorithm["Tabu Search"]["Soft cost"]:.2f}'
                        if "Tabu Search" in comparison_by_algorithm
                        else "Run algorithm comparison to measure"
                    ),
                },
                {
                    "Method": "Hill Climbing",
                    "Purpose": "Greedy local-search baseline",
                    "Inputs → outputs": "Same CSP seed → feasible local optimum",
                    "Measured contribution": (
                        f'Cost {comparison_by_algorithm["Hill Climbing"]["Initial cost"]:.2f} → '
                        f'{comparison_by_algorithm["Hill Climbing"]["Soft cost"]:.2f}'
                        if "Hill Climbing" in comparison_by_algorithm
                        else "Run algorithm comparison to measure"
                    ),
                },
                {
                    "Method": "Means–Ends Analysis",
                    "Purpose": "Repair disruptions with minimal changes",
                    "Inputs → outputs": "Disruption → bounded repaired schedule",
                    "Measured contribution": (
                        "A repair preview is available"
                        if st.session_state.get("pending_repair")
                        else "Run a repair scenario to measure"
                    ),
                },
                {
                    "Method": "Random Forest ML",
                    "Purpose": "Forecast attendance and demand",
                    "Inputs → outputs": "Historical/synthetic records → forecasts",
                    "Measured contribution": (
                        f'Attendance MAE {model.attendance_metrics["mae"]:.3f}; '
                        f'demand MAE {model.demand_metrics["mae"]:.1f}'
                    ),
                },
            ]
        )
        render_table(methodology, hide_index=True, use_container_width=True)
        st.markdown("**Unit 5 — Learning**")
        st.markdown(
            "Random Forest regressors predict attendance by subject/day/time and course demand from "
            "enrollment features. Model holdouts preserve date order when dates are provided; synthetic "
            "scores are explicitly illustrative. Predictions influence the attendance soft-cost term, "
            "never official capacity constraints."
        )
        presentation_mode = st.toggle(
            "Presentation / demo mode",
            key="presentation_mode",
            help="Show a guided sequence through the real application workflows.",
        )
        if presentation_mode:
            presentation_steps = (
                "1 · Problem statement",
                "2 · University dataset",
                "3 · CSP timetable generation",
                "4 · AI optimization",
                "5 · ML predictions",
                "6 · Dynamic rescheduling",
                "7 · Algorithm comparison",
                "8 · Final results",
            )
            selected_step = st.selectbox("Guided workflow stage", presentation_steps)
            st.info(
                f"Stage {selected_step}: the actions below use the included quick dataset and real "
                "scheduler/model calls; timetable repairs remain previews until explicitly applied."
            )
            if selected_step.startswith("1"):
                st.markdown(
                    "The task is to place university classes in time and rooms while preventing "
                    "teacher, room, and cohort clashes, then reduce soft schedule costs."
                )
            elif selected_step.startswith("2"):
                guided_problem = quick_demo_problem()
                render_table(
                    pd.DataFrame(
                        [
                            {
                                "Course": meeting.course_name,
                                "Department": meeting.department,
                                "Professor": guided_problem.teachers[meeting.teacher_id].name,
                                "Groups": ", ".join(meeting.groups),
                                "Enrollment": meeting.enrollment,
                                "Room type": meeting.room_type,
                            }
                            for meeting in guided_problem.meetings
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            elif selected_step.startswith("3"):
                if st.button("Run presentation CSP seed", key="presentation_csp"):
                    guided_problem = quick_demo_problem()
                    try:
                        guided_seed, guided_stats = solve_csp(
                            guided_problem, seed=42, return_stats=True
                        )
                        st.session_state["presentation_seed"] = guided_seed
                        st.session_state["presentation_csp_nodes"] = guided_stats.nodes
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"Presentation CSP failed: {exc}")
                if st.session_state.get("presentation_seed"):
                    st.success(
                        f"Generated a feasible quick-demo CSP seed in "
                        f'{st.session_state.get("presentation_csp_nodes", 0):,} search nodes.'
                    )
            elif selected_step.startswith("4"):
                if not st.session_state.get("presentation_seed"):
                    st.info("Run the CSP seed in stage 3 before optimizing it.")
                elif st.button("Optimize presentation seed", key="presentation_optimize"):
                    try:
                        presentation_result = optimize_detailed(
                            quick_demo_problem(),
                            st.session_state["presentation_seed"],
                            "annealing",
                            model,
                            iterations=160,
                            seed=42,
                            max_runtime_seconds=5,
                        )
                        st.session_state["presentation_optimization"] = presentation_result
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"Presentation optimization failed: {exc}")
                if st.session_state.get("presentation_optimization"):
                    presentation_result = st.session_state["presentation_optimization"]
                    st.metric(
                        "Measured objective cost",
                        f"{presentation_result.best_cost:.2f}",
                        f"{presentation_result.initial_cost - presentation_result.best_cost:+.2f}",
                    )
            elif selected_step.startswith("5"):
                guided_problem = quick_demo_problem()
                guided_meeting = st.selectbox(
                    "Sample class for ML prediction",
                    guided_problem.meetings,
                    format_func=lambda meeting: meeting.course_name,
                    key="presentation_ml_meeting",
                )
                guided_day = st.selectbox(
                    "Predicted weekday",
                    range(len(guided_problem.days)),
                    format_func=lambda day: guided_problem.days[day],
                    key="presentation_ml_day",
                )
                st.metric(
                    "Predicted attendance",
                    f"{model.predict_attendance(guided_meeting, guided_day, 2):.1%}",
                )
                st.caption(
                    f"Model source: {model.attendance_source}; training holdout MAE "
                    f'{model.attendance_metrics["mae"]:.3f}.'
                )
            elif selected_step.startswith("6"):
                st.info(
                    "Open the Repair workspace to preview a professor absence, room/equipment outage, "
                    "enrollment increase, preference change, or added class. The active timetable "
                    "remains unchanged until Apply is selected."
                )
            elif selected_step.startswith("7"):
                if st.button("Compare quick-demo algorithms", key="presentation_compare"):
                    guided_problem = quick_demo_problem()
                    try:
                        guided_seed = st.session_state.get("presentation_seed")
                        if not guided_seed:
                            guided_seed, _stats = solve_csp(
                                guided_problem, seed=42, return_stats=True
                            )
                        guided_results = []
                        for label, method in (
                            ("Simulated Annealing", "annealing"),
                            ("Tabu Search", "tabu"),
                            ("Hill Climbing", "hill-climbing"),
                        ):
                            measured = optimize_detailed(
                                guided_problem,
                                guided_seed,
                                method,
                                model,
                                iterations=100,
                                seed=42,
                                max_runtime_seconds=4,
                            )
                            guided_results.append(
                                {
                                    "Algorithm": label,
                                    "Initial cost": round(measured.initial_cost, 3),
                                    "Final cost": round(measured.best_cost, 3),
                                    "Runtime (s)": round(measured.runtime_seconds, 3),
                                    "Accepted moves": measured.accepted_moves,
                                    "Hard violations": len(
                                        validate_schedule(guided_problem, measured.schedule)
                                    ),
                                }
                            )
                        st.session_state["presentation_comparison"] = guided_results
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"Presentation comparison failed: {exc}")
                if st.session_state.get("presentation_comparison"):
                    render_table(
                        pd.DataFrame(st.session_state["presentation_comparison"]),
                        hide_index=True,
                        use_container_width=True,
                    )
            elif selected_step.startswith("8"):
                if schedule_exists:
                    final_metrics = schedule_metrics(problem, st.session_state["schedule"])
                    final_cols = st.columns(3)
                    final_cols[0].metric(
                        "Hard violations",
                        len(validate_schedule(problem, st.session_state["schedule"])),
                    )
                    final_cols[1].metric(
                        "Student idle periods",
                        int(final_metrics["student_idle_periods"]),
                    )
                    final_cols[2].metric(
                        "Soft score",
                        soft_score(
                            objective_cost(problem, st.session_state["schedule"], model),
                            len(problem.meetings),
                        ),
                    )
                else:
                    st.info("Generate the active timetable to show final measured results.")
    st.markdown(
        f'<div class="soft-note">Model note: {html.escape(model_description(problem))}</div>',
        unsafe_allow_html=True,
    )

with generation_tab:
    st.markdown("## Intelligent timetable generation")
    st.markdown(
        '<div class="panel-sub">Traditional scheduling finds a feasible timetable. AI optimization then explores hard-constraint-safe alternatives to improve its soft score.</div>',
        unsafe_allow_html=True,
    )
    before_after = st.columns(2)
    with before_after[0]:
        st.markdown(
            '<div class="panel"><div class="panel-title">1 · CSP feasible seed</div>'
            '<div class="panel-sub">Backtracking · MRV · Degree heuristic · Forward checking</div>'
            '<div>Respects teacher and cohort conflicts, room type/capacity, labs, equipment, '
            "teacher availability, and daily teaching limits.</div></div>",
            unsafe_allow_html=True,
        )
    with before_after[1]:
        st.markdown(
            '<div class="panel"><div class="panel-title">2 · AI soft-constraint search</div>'
            '<div class="panel-sub">Simulated Annealing / Tabu Search / Hill Climbing</div>'
            '<div>Improves idle gaps, daily load balance, teacher preferences, consecutive heavy classes, '
            "room balance, and attendance-weighted slot utility without breaking hard constraints.</div></div>",
            unsafe_allow_html=True,
        )
    if schedule_exists:
        generation = st.session_state.get("generation", {})
        solver_stats = generation.get("solver_stats")
        if solver_stats:
            cols = st.columns(5)
            cols[0].metric("CSP variables", solver_stats.variables)
            cols[1].metric("Hard constraints", solver_stats.constraints)
            cols[2].metric("Search nodes", f"{solver_stats.nodes:,}")
            cols[3].metric("Backtracks", f"{solver_stats.backtracks:,}")
            cols[4].metric("CSP time", f"{solver_stats.elapsed_seconds:.2f}s")
        seed_schedule = st.session_state.get("baseline_schedule")
        if seed_schedule:
            seed_cost = objective_cost(problem, seed_schedule, model)
            final_cost = objective_cost(problem, st.session_state["schedule"], model)
            improvement = 100.0 * (seed_cost - final_cost) / max(abs(seed_cost), 1e-9)
            st.metric("Optimization improvement over CSP seed", f"{improvement:.2f}%", f"{seed_cost:.2f} → {final_cost:.2f} cost")
        trace = st.session_state.get("optimization_result")
        if trace:
            trace_frame = pd.DataFrame(trace.convergence)
            trace_long = trace_frame.melt(
                id_vars="iteration",
                value_vars=["current_cost", "best_cost"],
                var_name="Trace",
                value_name="Objective cost",
            )
            convergence_chart = px.line(
                trace_long,
                x="iteration",
                y="Objective cost",
                color="Trace",
                color_discrete_sequence=["#8a84ee", "#4338ca"],
                template=CHART_TEMPLATE,
            )
            convergence_chart.update_layout(
                height=330,
                xaxis_title="Search iteration",
                yaxis_title="Soft objective (lower is better)",
                margin=dict(l=0, r=0, t=15, b=0),
                font=dict(family="Segoe UI", color=CHART_TEXT),
            )
            st.plotly_chart(convergence_chart, use_container_width=True, config={"displayModeBar": False})
            temperature_chart = px.line(
                trace_frame,
                x="iteration",
                y="temperature",
                template=CHART_TEMPLATE,
                labels={
                    "iteration": "Iteration",
                    "temperature": "Annealing temperature",
                },
            )
            temperature_chart.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.markdown("**Temperature and reheating trace**")
            st.plotly_chart(temperature_chart, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                f"Rejected moves: {trace.rejected_moves} · reheats: {trace.reheats} · "
                f"adaptive neighborhood: {'on' if adaptive_neighborhood else 'off'}."
            )
            st.caption(
                f"Executed {trace.method}; {trace.iterations} iterations, "
                f"{trace.accepted_moves} accepted moves, {trace.runtime_seconds:.2f}s; "
                f"stop reason: {trace.stop_reason}."
            )
    else:
        st.info("Generate the timetable from the AI Command Center to see the CSP diagnostics and optimization trace.")

with timetable_tab:
    st.markdown("## Weekly timetable")
    st.markdown(
        '<div class="panel-sub">A time-grid view of your schedule. Switch perspectives and search courses to focus on what matters.</div>',
        unsafe_allow_html=True,
    )
    if not schedule_exists:
        st.info("Generate a timetable from Overview to see your weekly calendar.")
    else:
        versions = ["Optimized timetable"]
        if st.session_state.get("baseline_schedule"):
            versions.insert(0, "Original CSP seed")
        version = st.selectbox("Compare schedule version", versions, key="calendar_version")
        schedule: dict[str, Placement] = (
            st.session_state["baseline_schedule"]
            if version == "Original CSP seed"
            else st.session_state["schedule"]
        )
        rows = schedule_rows(problem, schedule)
        filters = st.columns([1, 1.15, 1.2, 1.1, 1.45], gap="small")
        with filters[0]:
            view_type = st.selectbox("View", ("Student group", "Teacher", "Room"), key="calendar_view")
        if view_type == "Student group":
            choices = sorted({group_id for meeting in problem.meetings for group_id in meeting.groups})
            selected = filters[1].selectbox("Cohort", choices, key="calendar_group")
            rows = [row for row in rows if selected in str(row["groups"]).split(", ")]
        elif view_type == "Teacher":
            choices = sorted(problem.teachers)
            selected = filters[1].selectbox(
                "Instructor",
                choices,
                format_func=lambda teacher_id: (
                    f"{problem.teachers[teacher_id].name} ({teacher_id})"
                ),
                key="calendar_teacher",
            )
            rows = [row for row in rows if row["teacher_id"] == selected]
        else:
            choices = sorted(problem.rooms)
            selected = filters[1].selectbox(
                "Room",
                choices,
                format_func=lambda room_id: (
                    f"{room_id} · {problem.rooms[room_id].room_type} · "
                    f"{problem.rooms[room_id].capacity} seats"
                ),
                key="calendar_room",
            )
            rows = [row for row in rows if row["room_id"] == selected]
        departments = sorted({meeting.department for meeting in problem.meetings})
        department_filter = filters[2].selectbox(
            "Department", ["All", *departments], key="calendar_department"
        )
        if department_filter != "All":
            rows = [row for row in rows if row["department"] == department_filter]
        with filters[3]:
            day_filter = st.selectbox("Day", ("All days", *problem.days), key="calendar_day")
        if day_filter != "All days":
            rows = [row for row in rows if row["day"] == day_filter]
        with filters[4]:
            search = st.text_input("Find a course", placeholder="Search course name…", key="calendar_search")
        if search.strip():
            rows = [
                row for row in rows
                if search.lower() in str(row["course_name"]).lower()
                or search.lower() in str(row["subject"]).lower()
                or search.lower() in str(row["course_id"]).lower()
            ]
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin:5px 0 12px">'
            f'<span class="status-pill"><span class="status-dot"></span>{len(rows)} classes shown</span>'
            '<span style="font-size:11px;color:#8a94a7">Scroll horizontally to see the full week</span></div>',
            unsafe_allow_html=True,
        )
        render_calendar(problem, schedule, rows)
        st.write("")
        subjects = sorted({str(row["subject"]) for row in rows})
        if subjects:
            legend = "".join(
                f'<span class="legend-item"><span class="legend-dot" style="background:{COLORS[hashlib.sha1(subject.encode()).digest()[0] % len(COLORS)]}"></span>{html.escape(subject)}</span>'
                for subject in subjects[:9]
            )
            st.markdown(legend, unsafe_allow_html=True)
        if rows:
            row_lookup = {str(row["meeting_id"]): row for row in rows}
            detail_id = st.selectbox(
                "Inspect a session",
                list(row_lookup),
                format_func=lambda meeting_id: (
                    f'{row_lookup[meeting_id]["course_name"]} · {row_lookup[meeting_id]["day"]} '
                    f'{row_lookup[meeting_id]["start_time"]}'
                ),
                key="session_details",
            )
            detail_row = row_lookup[detail_id]
            detail_room = problem.rooms[str(detail_row["room_id"])]
            detail_meeting = next(meeting for meeting in problem.meetings if meeting.meeting_id == detail_id)
            detail_prediction = model.predict_attendance(
                detail_meeting,
                problem.days.index(str(detail_row["day"])),
                int(detail_row["start_slot"]),
            )
            st.markdown(
                f'<div class="panel"><div class="panel-title">{html.escape(str(detail_row["course_name"]))}</div>'
                f'<div class="panel-sub">{html.escape(str(detail_row["department"]))} · '
                f'{html.escape(str(detail_row["room_type"]).title())} session · '
                f'{html.escape(str(detail_row["start_time"]))}–{html.escape(str(detail_row["end_time"]))}</div>'
                f'<div><b>Teacher:</b> {html.escape(str(detail_row["teacher_id"]))} &nbsp; '
                f'<b>Room:</b> {html.escape(str(detail_room.room_id))} '
                f'({detail_room.capacity} seats) &nbsp; '
                f'<b>Section(s):</b> {html.escape(str(detail_row["groups"]))} &nbsp; '
                f'<b>Official enrollment:</b> {detail_meeting.enrollment} &nbsp; '
                f'<b>Predicted attendance:</b> {detail_prediction:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with st.expander("See schedule as a detailed list"):
            if rows:
                render_table(
                    pd.DataFrame(rows)[
                    ["day", "start_time", "end_time", "course_name", "department", "groups", "teacher_id", "room_id", "room_type"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.caption("No classes match the selected filters.")

with insights_tab:
    st.markdown("## Schedule insights")
    st.markdown(
        '<div class="panel-sub">Compare optimizers using the same CSP-feasible starting timetable.</div>',
        unsafe_allow_html=True,
    )
    if not schedule_exists:
        st.info("Generate a timetable to unlock comparisons and schedule quality metrics.")
    else:
        schedule = st.session_state["schedule"]
        generation = st.session_state.get("generation", {})
        errors = validate_schedule(problem, schedule)
        cost = objective_cost(problem, schedule, model)
        cards = (
            '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px">'
            + metric_card("Soft score", f"{soft_score(cost, len(problem.meetings)):.1f}", "Higher is better", "✦")
            + metric_card("Teacher fairness", f"{teacher_load_fairness(problem, schedule):.1f}%", "Teaching load balance", "♙")
            + metric_card("Hard conflicts", str(len(errors)), "Should remain at zero", "✓")
            + metric_card("Generation time", f'{generation.get("runtime", 0):.1f}s', generation.get("algorithm", "Current schedule"), "◷")
            + "</div>"
        )
        st.markdown(cards, unsafe_allow_html=True)
        st.write("")
        if errors:
            st.error("Conflict detector reported hard-constraint violations.")
            render_table(pd.DataFrame({"Conflict": errors}), use_container_width=True)
        else:
            st.success("All teacher, room, group, capacity, type, availability, lab-duration, and daily-hours checks passed.")
        st.markdown("### Active objective breakdown")
        breakdown = objective_breakdown(problem, schedule, model)
        breakdown_frame = pd.DataFrame(
            [
                {"Objective term": key.replace("_", " ").title(), "Weighted cost": value}
                for key, value in breakdown.items()
            ]
        ).sort_values("Weighted cost", ascending=False)
        render_table(breakdown_frame, hide_index=True, use_container_width=True)
        st.caption(
            f"Additive components total {sum(breakdown.values()):.3f}; "
            f"full objective {objective_cost(problem, schedule, model):.3f}. "
            "Larger cost contributions are the most influential terms for this schedule."
        )
        optimization_result = st.session_state.get("optimization_result")
        if optimization_result:
            diagnostic_cols = st.columns(4)
            diagnostic_cols[0].metric("Accepted moves", optimization_result.accepted_moves)
            diagnostic_cols[1].metric("Rejected moves", optimization_result.rejected_moves)
            diagnostic_cols[2].metric("Reheats", optimization_result.reheats)
            diagnostic_cols[3].metric("Iterations", optimization_result.iterations)
            st.caption(f"Optimizer stop reason: {optimization_result.stop_reason}")
            if optimization_result.rejection_reasons:
                st.caption(
                    "Rejected-move diagnostics: "
                    + " · ".join(
                        f"{reason}: {count}"
                        for reason, count in optimization_result.rejection_reasons
                    )
                )
            convergence_data = pd.DataFrame(optimization_result.convergence)
            convergence_chart = px.line(
                convergence_data,
                x="iteration",
                y=["current_cost", "best_cost"],
                template=CHART_TEMPLATE,
                labels={
                    "iteration": "Iteration",
                    "value": "Objective cost",
                    "variable": "Trace",
                },
            )
            convergence_chart.update_layout(
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                legend_title=None,
            )
            st.plotly_chart(convergence_chart, use_container_width=True, config={"displayModeBar": False})
        st.markdown("### Compare optimization methods")
        compare_col, button_col = st.columns([1, 1], vertical_alignment="bottom")
        compare_iterations = compare_col.select_slider(
            "Iterations per method", options=(100, 200, 300, 400, 600, 800), value=300
        )
        run_comparison = button_col.button("Run comparison", use_container_width=True)
        if run_comparison:
            started = time.perf_counter()
            try:
                seed_schedule, solver_stats = solve_csp(problem, seed=42, return_stats=True)
                results = []
                traces = {}
                with st.spinner("Running the three optimizer baselines…"):
                    for label, method in (
                        ("Simulated Annealing", "annealing"),
                        ("Tabu Search", "tabu"),
                        ("Hill Climbing", "hill-climbing"),
                    ):
                        run_result = optimize_detailed(
                            problem, seed_schedule, method, model, int(compare_iterations), seed=42
                        )
                        result_schedule = run_result.schedule
                        result_cost = run_result.best_cost
                        final_metrics = schedule_metrics(problem, result_schedule)
                        violations = len(validate_schedule(problem, result_schedule))
                        improvement_pct = 100.0 * (
                            run_result.initial_cost - result_cost
                        ) / max(abs(run_result.initial_cost), 1e-9)
                        results.append(
                            {
                                "Algorithm": label,
                                "Initial cost": round(run_result.initial_cost, 3),
                                "Soft cost": round(result_cost, 3),
                                "Soft score": soft_score(result_cost, len(problem.meetings)),
                                "Improvement (%)": round(improvement_pct, 2),
                                "Student idle periods": int(final_metrics["student_idle_periods"]),
                                "Teacher-load fairness (%)": round(final_metrics["teacher_fairness_pct"], 2),
                                "Runtime (s)": round(run_result.runtime_seconds, 3),
                                "Iterations": run_result.iterations,
                                "Accepted moves": run_result.accepted_moves,
                                "Hard violations": violations,
                            }
                        )
                        traces[label] = pd.DataFrame(run_result.convergence)
                        RunStore(ROOT / "timetable.db").save_experiment(
                            method,
                            run_result.initial_cost,
                            result_cost,
                            run_result.runtime_seconds,
                            run_result.iterations,
                            run_result.accepted_moves,
                            final_metrics,
                        )
                st.session_state["comparison"] = results
                st.session_state["comparison_traces"] = traces
                st.session_state["comparison_seed_cost"] = objective_cost(problem, seed_schedule, model)
                st.caption(
                    f"Shared CSP seed: {solver_stats.nodes:,} search steps · "
                    f"total comparison: {time.perf_counter() - started:.1f}s"
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
        if "comparison" in st.session_state:
            result_frame = pd.DataFrame(st.session_state["comparison"])
            render_table(result_frame, hide_index=True, use_container_width=True)
            valid_rows = result_frame[result_frame["Hard violations"] == 0]
            if not valid_rows.empty:
                winner = valid_rows.sort_values("Soft cost").iloc[0]
                st.success(
                    f"Best measured result for this run: {winner['Algorithm']} "
                    f"(cost {winner['Soft cost']:.2f}, improvement {winner['Improvement (%)']:.2f}%). "
                    "This is an instance-specific measurement, not a universal ranking."
                )
            chart_data = result_frame.melt(
                id_vars="Algorithm",
                value_vars=["Soft score", "Teacher-load fairness (%)", "Improvement (%)"],
                var_name="Measure",
                value_name="Score",
            )
            comparison_chart = px.bar(
                chart_data,
                x="Algorithm",
                y="Score",
                color="Measure",
                barmode="group",
                color_discrete_sequence=["#5146e5", "#20a486"],
                template=CHART_TEMPLATE,
            )
            comparison_chart.update_layout(
                height=330,
                margin=dict(l=0, r=0, t=25, b=0),
                font=dict(family="Segoe UI", color=CHART_TEXT),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend_title=None,
            )
            comparison_chart.update_traces(marker_line_width=0, marker_cornerradius=6)
            st.plotly_chart(comparison_chart, use_container_width=True, config={"displayModeBar": False})
            if st.session_state.get("comparison_traces"):
                convergence_frames = []
                for algorithm, trace_frame in st.session_state["comparison_traces"].items():
                    part = trace_frame[["iteration", "best_cost"]].copy()
                    part["Algorithm"] = algorithm
                    convergence_frames.append(part)
                convergence = pd.concat(convergence_frames, ignore_index=True)
                convergence_chart = px.line(
                    convergence,
                    x="iteration",
                    y="best_cost",
                    color="Algorithm",
                    template=CHART_TEMPLATE,
                    color_discrete_sequence=["#5146e5", "#0f9f8f", "#f28c45"],
                )
                convergence_chart.update_layout(
                    height=320,
                    xaxis_title="Iteration",
                    yaxis_title="Best cost so far (lower is better)",
                    margin=dict(l=0, r=0, t=15, b=0),
                    font=dict(family="Segoe UI", color=CHART_TEXT),
                )
                st.plotly_chart(convergence_chart, use_container_width=True, config={"displayModeBar": False})

with ml_tab:
    st.markdown("## Machine learning prediction center")
    st.markdown(
        '<div class="panel-sub">Two trained Random Forest regressors support planning. Predictions influence soft slot utility and demand forecasts; mandatory room capacity still uses official course enrollment.</div>',
        unsafe_allow_html=True,
    )
    ml_cols = st.columns(2)
    ml_cols[0].metric(
        "Attendance model · MAE",
        f'{model.attendance_metrics["mae"] * 100:.2f} pp',
        f'RMSE {model.attendance_metrics["rmse"] * 100:.2f} pp · R² {model.attendance_metrics["r2"]:.3f}',
    )
    ml_cols[1].metric(
        "Demand model · MAE",
        f'{model.demand_metrics["mae"]:.1f} students',
        f'RMSE {model.demand_metrics["rmse"]:.1f} · R² {model.demand_metrics["r2"]:.3f}',
    )
    st.caption(
        f"Attendance training: {model.attendance_source} ({len(model.attendance_rows):,} records; "
        "last 20% held out in chronological/row order). "
        f"Demand training: {model.demand_source} ({len(model.demand_rows):,} records). "
        "Synthetic holdout metrics are not evidence of real-world accuracy."
    )
    attendance_col, demand_col = st.columns(2, gap="large")
    with attendance_col:
        st.markdown("### Attendance: actual vs predicted")
        holdout = model.attendance_holdout.head(300).copy()
        holdout["actual"] *= 100
        holdout["predicted"] *= 100
        holdout["Record"] = range(1, len(holdout) + 1)
        actual_chart = px.line(
            holdout,
            x="Record",
            y=["actual", "predicted"],
            color_discrete_sequence=["#5146e5", "#0f9f8f"],
            template=CHART_TEMPLATE,
        )
        actual_chart.update_layout(
            height=320,
            yaxis_title="Attendance (%)",
            legend_title=None,
            margin=dict(l=0, r=0, t=15, b=0),
        )
        st.plotly_chart(actual_chart, use_container_width=True, config={"displayModeBar": False})
        importance = pd.DataFrame(
            [{"Feature": key, "Importance": value} for key, value in model.attendance_feature_importance.items()]
        )
        importance_chart = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#c9c6ff", "#5146e5"],
            template=CHART_TEMPLATE,
        )
        importance_chart.update_layout(
            height=280,
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(importance_chart, use_container_width=True, config={"displayModeBar": False})
    with demand_col:
        st.markdown("### Demand: held-out enrollment")
        demand_holdout = model.demand_holdout.head(300).copy()
        demand_chart = px.scatter(
            demand_holdout,
            x="actual",
            y="predicted",
            template=CHART_TEMPLATE,
            color_discrete_sequence=["#0f9f8f"],
            labels={"actual": "Observed demand", "predicted": "Predicted demand"},
        )
        max_demand = max(
            float(demand_holdout["actual"].max()),
            float(demand_holdout["predicted"].max()),
            1.0,
        )
        demand_chart.add_shape(
            type="line", x0=0, y0=0, x1=max_demand, y1=max_demand,
            line=dict(color="#9da5b3", dash="dash"),
        )
        demand_chart.update_layout(height=320, margin=dict(l=0, r=0, t=15, b=0))
        st.plotly_chart(demand_chart, use_container_width=True, config={"displayModeBar": False})
        demand_importance = pd.DataFrame(
            [{"Feature": key, "Importance": value} for key, value in model.demand_feature_importance.items()]
        )
        render_table(demand_importance, hide_index=True, use_container_width=True)
        unique_courses = {}
        for meeting in problem.meetings:
            unique_courses.setdefault(meeting.course_id, meeting)
        forecasts = []
        for meeting in unique_courses.values():
            predicted_demand = model.predict_demand(meeting)
            forecasts.append(
                {
                    "Course": meeting.course_name,
                    "Department": meeting.department,
                    "Section enrollment": meeting.enrollment,
                    "Elective": len(meeting.groups) > 1,
                    "Forecast demand": round(predicted_demand, 1),
                    "Advisory capacity": max(meeting.enrollment, int(round(predicted_demand))),
                }
            )
        st.markdown("#### Course demand and capacity-planning signals")
        st.caption(
            "Advisory only: these forecasts do not change official enrollment or hard room-capacity checks."
        )
        render_table(
            pd.DataFrame(forecasts).sort_values("Forecast demand", ascending=False).head(15),
            hide_index=True,
            use_container_width=True,
        )
    st.markdown("### Predicted attendance by weekday and period")
    selected_meeting = st.selectbox(
        "Course for attendance heatmap",
        problem.meetings,
        format_func=lambda meeting: f"{meeting.course_name} · {meeting.meeting_id}",
        key="attendance_course",
    )
    heatmap = [
        [
            100 * model.predict_attendance(selected_meeting, day, slot)
            for slot in range(problem.slots_per_day - selected_meeting.duration + 1)
        ]
        for day in range(len(problem.days))
    ]
    heatmap_chart = px.imshow(
        heatmap,
        x=[f"{8 + slot:02d}:00" for slot in range(problem.slots_per_day - selected_meeting.duration + 1)],
        y=problem.days,
        color_continuous_scale=["#e9e7ff", "#6559e8", "#24224e"],
        text_auto=".0f",
        aspect="auto",
        labels={"x": "Start time", "y": "Weekday", "color": "Attendance %"},
    )
    heatmap_chart.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(heatmap_chart, use_container_width=True, config={"displayModeBar": False})
    with st.expander("ML Impact Explorer"):
        st.caption(
            "Predictions are model outputs from the labeled source above. Expected attendance is "
            "advisory; room capacity is still checked against official enrollment."
        )
        selected_attendance = model.predict_attendance(
            selected_meeting,
            st.selectbox("Prediction weekday", list(range(len(problem.days))),
                         format_func=lambda day: problem.days[day], key="ml_impact_day"),
            st.selectbox("Prediction start period", list(range(problem.slots_per_day - selected_meeting.duration + 1)),
                         format_func=lambda slot: f"{8 + slot:02d}:00", key="ml_impact_slot"),
        )
        predicted_headcount = selected_meeting.enrollment * selected_attendance
        ml_summary = st.columns(3)
        ml_summary[0].metric("Predicted attendance", f"{selected_attendance:.1%}")
        ml_summary[1].metric("Expected attendees", f"{predicted_headcount:.1f}")
        if schedule_exists:
            impact_schedule = st.session_state["schedule"]
            impact_placement = impact_schedule[selected_meeting.meeting_id]
            assigned_room = problem.rooms[impact_placement.room_id]
            ml_summary[2].metric(
                "Assigned room capacity",
                assigned_room.capacity,
                f"official enrollment {selected_meeting.enrollment}",
            )
            occupancy = Occupancy(problem)
            for other_meeting in problem.meetings:
                if other_meeting.meeting_id != selected_meeting.meeting_id:
                    occupancy.add(other_meeting, impact_schedule[other_meeting.meeting_id])
            alternatives = []
            best_attendance = model.best_attendance(
                selected_meeting, len(problem.days), problem.slots_per_day
            )
            for candidate in meeting_domains(problem)[selected_meeting.meeting_id]:
                if not occupancy.can_place(selected_meeting, candidate):
                    continue
                candidate_schedule = {
                    **impact_schedule,
                    selected_meeting.meeting_id: candidate,
                }
                candidate_attendance = model.predict_attendance(
                    selected_meeting, candidate.day, candidate.start
                )
                attendance_term = (
                    model.preference_weights["attendance"]
                    * model.predict_demand(selected_meeting)
                    / max(1.0, selected_meeting.enrollment)
                    * max(0.0, best_attendance - candidate_attendance)
                )
                alternatives.append(
                    {
                        "Day": problem.days[candidate.day],
                        "Start": f"{8 + candidate.start:02d}:00",
                        "Room": candidate.room_id,
                        "Capacity": problem.rooms[candidate.room_id].capacity,
                        "Predicted attendance": f"{candidate_attendance:.1%}",
                        "Attendance cost": round(attendance_term, 3),
                        "Whole-schedule objective": round(
                            objective_cost(problem, candidate_schedule, model), 3
                        ),
                        "Objective delta vs assigned": round(
                            objective_cost(problem, candidate_schedule, model)
                            - objective_cost(problem, impact_schedule, model),
                            3,
                        ),
                    }
                )
            st.markdown("**Feasible alternatives (lowest full objective first)**")
            render_table(
                pd.DataFrame(alternatives).sort_values("Whole-schedule objective").head(8),
                hide_index=True,
                use_container_width=True,
            )
        else:
            ml_summary[2].metric("Official enrollment", selected_meeting.enrollment)
            st.info("Generate the active schedule to inspect feasible room and time alternatives.")
        history = model.attendance_rows
        course_history = history.loc[
            history["subject"].astype(str).str.lower() == selected_meeting.subject.lower()
        ].copy()
        if course_history.empty:
            course_history = history.copy()
        course_history["record_order"] = range(1, len(course_history) + 1)
        course_history["attendance_pct"] = course_history["attendance"] * 100
        history_chart = px.line(
            course_history.tail(100),
            x="record_order",
            y="attendance_pct",
            template=CHART_TEMPLATE,
            labels={
                "record_order": "Historical record order",
                "attendance_pct": "Observed attendance (%)",
            },
        )
        history_chart.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(history_chart, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Trend source: {model.attendance_source}; filtered to subject when matching records exist.")
        if schedule_exists and st.button(
            "Compare optimization with vs without attendance signal",
            key="compare_ml_signal",
        ):
            try:
                comparison_seed = st.session_state.get("baseline_schedule")
                if not comparison_seed:
                    comparison_seed, _stats = solve_csp(problem, seed=42, return_stats=True)
                no_signal_model = copy.copy(model)
                no_signal_model.preference_weights = dict(model.preference_weights)
                no_signal_model.preference_weights["attendance"] = 0.0
                with st.spinner("Running the same seeded optimizer under both objective settings…"):
                    no_signal = optimize_detailed(
                        problem, comparison_seed, "annealing", no_signal_model,
                        iterations=160, seed=42, max_runtime_seconds=5,
                    )
                    with_signal = optimize_detailed(
                        problem, comparison_seed, "annealing", model,
                        iterations=160, seed=42, max_runtime_seconds=5,
                    )
                st.session_state["ml_signal_comparison"] = {
                    "without": no_signal,
                    "with": with_signal,
                }
            except (ValueError, RuntimeError) as exc:
                st.error(f"ML signal comparison failed: {exc}")
        signal_comparison = st.session_state.get("ml_signal_comparison")
        if signal_comparison:
            rows = []
            for label, key in (
                ("Attendance signal off", "without"),
                ("Attendance signal on", "with"),
            ):
                result = signal_comparison[key]
                result_schedule = result.schedule
                rows.append(
                    {
                        "Run": label,
                        "Final objective": round(result.best_cost, 3),
                        "Soft score": soft_score(result.best_cost, len(problem.meetings)),
                        "Student idle periods": int(schedule_metrics(problem, result_schedule)["student_idle_periods"]),
                        "Runtime (s)": round(result.runtime_seconds, 3),
                        "Hard violations": len(validate_schedule(problem, result_schedule)),
                    }
                )
            render_table(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                f"Only the attendance objective weight changed: {model.preference_weights['attendance']:.3f} "
                "with signal versus 0.000 without. All hard constraints and other learned weights are held fixed; "
                "measured outcomes can vary and are not assumed to improve."
            )
    st.markdown("### Preference feedback (separate from attendance learning)")
    st.caption(
        "Upload explicit timetable ratings in the Data Management section to learn soft-objective weights. "
        "Attendance estimates never silently train preference weights."
    )
    render_table(
        pd.DataFrame(
            [{"Soft objective": key.replace("_", " ").title(), "Weight": value}
             for key, value in model.preference_weights.items()]
        ),
        hide_index=True,
        use_container_width=True,
    )

with analytics_tab:
    st.markdown("## Analytics and explainable decisions")
    if not schedule_exists:
        st.info("Generate a timetable to see computed workload, utilization, idle-time, and constraint analytics.")
    else:
        current_schedule = st.session_state["schedule"]
        metrics = schedule_metrics(problem, current_schedule)
        hard_errors = validate_schedule(problem, current_schedule)
        constraint_cols = st.columns(4)
        constraint_cols[0].metric("Hard constraints", f"{hard_constraint_count(problem):,}")
        constraint_cols[1].metric("Violations", len(hard_errors))
        constraint_cols[2].metric("Student idle periods", int(metrics["student_idle_periods"]))
        constraint_cols[3].metric("Room utilization", f'{metrics["classroom_utilization_pct"]:.1f}%')
        extra_metric_cols = st.columns(3)
        extra_metric_cols[0].metric("Teacher idle periods", int(metrics["teacher_idle_periods"]))
        extra_metric_cols[1].metric("Teacher fairness", f'{metrics["teacher_fairness_pct"]:.1f}%')
        extra_metric_cols[2].metric(
            "Teacher preference satisfaction",
            (
                f'{metrics["teacher_preference_satisfaction_pct"]:.1f}%'
                if metrics["teacher_preference_periods"]
                else "Not supplied"
            ),
        )
        teacher_rows = []
        room_hours: dict[str, int] = {room_id: 0 for room_id in problem.rooms}
        group_day_slots: dict[str, dict[int, set[int]]] = {}
        meeting_map = {meeting.meeting_id: meeting for meeting in problem.meetings}
        for meeting_id, placement in current_schedule.items():
            meeting = meeting_map[meeting_id]
            for group_id in meeting.groups:
                group_day_slots.setdefault(group_id, {}).setdefault(placement.day, set()).update(
                    range(placement.start, placement.start + meeting.duration)
                )
            room_hours[placement.room_id] += meeting.duration
        teacher_hours: dict[str, int] = {teacher_id: 0 for teacher_id in problem.teachers}
        for meeting in problem.meetings:
            if meeting.meeting_id in current_schedule:
                teacher_hours[meeting.teacher_id] += meeting.duration
        teacher_rows = [
            {"Teacher": teacher_id, "Name": teacher.name, "Teaching hours": teacher_hours[teacher_id],
             "Daily limit": teacher.max_hours_per_day}
            for teacher_id, teacher in problem.teachers.items()
        ]
        group_rows = []
        for group_id, days in group_day_slots.items():
            idle = sum(max(slots) - min(slots) + 1 - len(slots) for slots in days.values() if slots)
            group_rows.append({"Student group": group_id, "Idle periods": idle})
        room_capacity_hours = len(problem.days) * problem.slots_per_day
        room_frame = pd.DataFrame([
            {
                "Room": room_id,
                "Type": room.room_type.title(),
                "Capacity": room.capacity,
                "Scheduled hours": room_hours[room_id],
                "Utilization (%)": 100 * room_hours[room_id] / max(1, room_capacity_hours),
            }
            for room_id, room in problem.rooms.items()
        ])
        chart_a, chart_b = st.columns(2, gap="large")
        with chart_a:
            st.markdown("### Teacher workload")
            workload_chart = px.bar(
                pd.DataFrame(teacher_rows),
                x="Teacher",
                y="Teaching hours",
                color="Teaching hours",
                color_continuous_scale=["#d9d6ff", "#5146e5"],
                hover_data=["Name", "Daily limit"],
                template=CHART_TEMPLATE,
            )
            workload_chart.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=0, r=0, t=12, b=0))
            st.plotly_chart(workload_chart, use_container_width=True, config={"displayModeBar": False})
        with chart_b:
            st.markdown("### Classroom utilization")
            utilization_chart = px.bar(
                room_frame,
                x="Room",
                y="Utilization (%)",
                color="Type",
                color_discrete_sequence=["#5146e5", "#0f9f8f"],
                template=CHART_TEMPLATE,
            )
            utilization_chart.update_layout(height=320, margin=dict(l=0, r=0, t=12, b=0))
            st.plotly_chart(utilization_chart, use_container_width=True, config={"displayModeBar": False})
        student_idle_chart = px.bar(
            pd.DataFrame(group_rows),
            x="Student group",
            y="Idle periods",
            color="Idle periods",
            color_continuous_scale=["#bdeee2", "#0f9f8f"],
            template=CHART_TEMPLATE,
        )
        student_idle_chart.update_layout(height=280, coloraxis_showscale=False, margin=dict(l=0, r=0, t=12, b=0))
        st.plotly_chart(student_idle_chart, use_container_width=True, config={"displayModeBar": False})
        st.markdown("### Constraint satisfaction report")
        render_table(
            pd.DataFrame(
                [{"Constraint": "Hard constraints", "Status": "PASS" if not hard_errors else "FAIL",
                  "Details": f"{hard_constraint_count(problem):,} modeled CSP restrictions; {len(hard_errors)} current violation(s)."}]
                + [{"Constraint": f"Violation {index + 1}", "Status": "FAIL", "Details": error}
                   for index, error in enumerate(hard_errors)]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("### Explain an AI scheduling decision")
        selected_id = st.selectbox(
            "Choose a scheduled course",
            list(current_schedule),
            format_func=lambda meeting_id: f"{meeting_map[meeting_id].course_name} · {meeting_id}",
            key="explain_meeting",
        )
        if st.button("Explain this placement"):
            try:
                explanation = explain_placement(problem, current_schedule, selected_id, model)
                meeting = explanation["meeting"]
                placement = explanation["placement"]
                st.markdown(
                    f"**{meeting.course_name}** is scheduled **{problem.days[placement.day]} at "
                    f"{8 + placement.start:02d}:00** in **{placement.room_id}**."
                )
                for reason in explanation["reasons"]:
                    st.markdown(f"- {reason}")
                if explanation["attendance_prediction"] is not None:
                    st.caption(
                        f"Model evidence: predicted attendance {explanation['attendance_prediction']:.1%}; "
                        f"forecast demand {explanation['demand_prediction']:.1f} students. "
                        "Predicted attendance is not used to waive official room capacity."
                    )
            except ValueError as exc:
                st.error(str(exc))

with reschedule_tab:
    st.markdown("## Smart AI rescheduling")
    st.markdown(
        '<div class="panel-sub">Means–Ends Analysis identifies what differs from the desired state, then a bounded CSP local repair searches for low-disruption alternatives. It is not a guarantee of globally optimal repair.</div>',
        unsafe_allow_html=True,
    )
    repair_col, export_col = st.columns([1.15, 1], gap="large")
    with repair_col:
        st.markdown("### Reschedule a disruption")
        st.markdown(
            '<div class="panel-sub">Preserve unaffected placements where feasible; cascade to direct blockers only when the local repair needs them.</div>',
            unsafe_allow_html=True,
        )
        if not schedule_exists:
            st.info("Generate a timetable before repairing disruptions.")
        else:
            absent_teacher = st.selectbox(
                "Professor unavailable for a day",
                ["None", *sorted(problem.teachers)],
                format_func=lambda teacher_id: (
                    "No absence" if teacher_id == "None"
                    else f"{problem.teachers[teacher_id].name} ({teacher_id})"
                ),
            )
            absent_day_name = st.selectbox("Absence day", problem.days, key="absence_day")
            unavailable_room = st.selectbox(
                "Classroom unavailable",
                ["None", *sorted(problem.rooms)],
                format_func=lambda room_id: (
                    "No room outage" if room_id == "None"
                    else f"{room_id} · {problem.rooms[room_id].room_type} · "
                    f"{problem.rooms[room_id].capacity} seats"
                ),
            )
            availability_teacher = st.selectbox(
                "Professor availability changed",
                ["None", *sorted(problem.teachers)],
                format_func=lambda teacher_id: (
                    "No availability change" if teacher_id == "None"
                    else f"{problem.teachers[teacher_id].name} ({teacher_id})"
                ),
            )
            availability_day = st.selectbox(
                "Changed availability day", problem.days, key="availability_day"
            )
            unavailable_slots = st.multiselect(
                "Slots now unavailable",
                list(range(problem.slots_per_day)),
                format_func=lambda slot: f"{8 + slot:02d}:00",
                disabled=availability_teacher == "None",
            )
            equipment_rooms = [
                room_id for room_id, room in problem.rooms.items() if room.equipment
            ]
            equipment_room = st.selectbox(
                "Lab equipment outage — affected room", ["None", *equipment_rooms]
            )
            equipment_options = (
                ["None", *sorted(problem.rooms[equipment_room].equipment)]
                if equipment_room != "None"
                else []
            )
            equipment_choice = (
                st.selectbox("Unavailable equipment", equipment_options)
                if equipment_options
                else "None"
            )
            unavailable_equipment = None if equipment_choice == "None" else equipment_choice
            increase_enrollment = st.checkbox("Simulate increased section enrollment")
            enrollment_updates: dict[str, int] = {}
            if increase_enrollment:
                enrollment_meeting = st.selectbox(
                    "Section with increased enrollment",
                    problem.meetings,
                    format_func=lambda meeting: (
                        f"{meeting.course_name} · {', '.join(meeting.groups)} · "
                        f"{meeting.enrollment} enrolled"
                    ),
                    key="enrollment_scenario_meeting",
                )
                updated_enrollment_value = st.number_input(
                    "New official enrollment",
                    min_value=enrollment_meeting.enrollment + 1,
                    max_value=max(500, enrollment_meeting.enrollment + 10),
                    value=enrollment_meeting.enrollment + 10,
                    key="enrollment_scenario_value",
                )
                enrollment_updates[enrollment_meeting.meeting_id] = int(updated_enrollment_value)
            change_preferences = st.checkbox("Simulate changed professor time preferences")
            preference_teacher = "None"
            preference_slots: list[tuple[int, int]] = []
            if change_preferences:
                preference_teacher = st.selectbox(
                    "Professor whose preferences changed",
                    sorted(problem.teachers),
                    format_func=lambda teacher_id: (
                        f"{problem.teachers[teacher_id].name} ({teacher_id})"
                    ),
                    key="preference_scenario_teacher",
                )
                all_preference_slots = [
                    (day, slot)
                    for day in range(len(problem.days))
                    for slot in range(problem.slots_per_day)
                ]
                preference_slots = st.multiselect(
                    "Preferred periods",
                    all_preference_slots,
                    default=sorted(problem.teachers[preference_teacher].preferred_slots),
                    format_func=lambda item: (
                        f"{problem.days[item[0]]} {8 + item[1]:02d}:00"
                    ),
                    key="preference_scenario_slots",
                )
            add_new_class = st.checkbox("Include a newly added class")
            added_meeting = None
            if add_new_class:
                course_fields = st.columns(3)
                new_course_id = course_fields[0].text_input("New course ID", "NEW101")
                new_course_name = course_fields[1].text_input("Course name", "Special Topics")
                new_teacher = course_fields[2].selectbox(
                    "Professor",
                    sorted(problem.teachers),
                    format_func=lambda teacher_id: (
                        f"{problem.teachers[teacher_id].name} ({teacher_id})"
                    ),
                    key="new_teacher",
                )
                group_options = sorted({group for meeting in problem.meetings for group in meeting.groups})
                class_fields = st.columns(3)
                new_group = class_fields[0].selectbox("Student section", group_options, key="new_group")
                new_department = class_fields[1].selectbox(
                    "Department",
                    sorted({meeting.department for meeting in problem.meetings}),
                    key="new_department",
                )
                new_room_type = class_fields[2].selectbox(
                    "Room type", ("lecture", "lab"), key="new_room_type"
                )
                group_enrollment = max(
                    (
                        meeting.enrollment
                        for meeting in problem.meetings
                        if new_group in meeting.groups
                    ),
                    default=30,
                )
                max_capacity = max((room.capacity for room in problem.rooms.values()), default=500)
                new_enrollment = st.number_input(
                    "Official enrollment",
                    min_value=1,
                    max_value=max_capacity,
                    value=min(group_enrollment, max_capacity),
                )
                equipment_choices = sorted({
                    equipment for room in problem.rooms.values() for equipment in room.equipment
                })
                new_equipment = (
                    st.multiselect("Required lab equipment", equipment_choices)
                    if new_room_type == "lab"
                    else []
                )
                if new_course_id.strip():
                    added_meeting = Meeting(
                        meeting_id=f"{new_course_id.strip()}-S1",
                        course_id=new_course_id.strip(),
                        course_name=new_course_name.strip() or new_course_id.strip(),
                        subject=new_course_name.strip() or new_course_id.strip(),
                        teacher_id=new_teacher,
                        groups=(new_group,),
                        enrollment=int(new_enrollment),
                        room_type=new_room_type,
                        duration=2 if new_room_type == "lab" else 1,
                        session_index=1,
                        sessions_per_week=1,
                        department=new_department,
                        required_equipment=frozenset(new_equipment),
                    )
            if st.button("Repair affected classes", type="primary", use_container_width=True):
                try:
                    if (
                        absent_teacher == "None"
                        and unavailable_room == "None"
                        and not unavailable_equipment
                        and not unavailable_slots
                        and not enrollment_updates
                        and not change_preferences
                        and not added_meeting
                    ):
                        raise ValueError(
                            "Select a disruption, scenario change, or new class to repair."
                        )
                    if added_meeting and any(
                        meeting.course_id == added_meeting.course_id for meeting in problem.meetings
                    ):
                        raise ValueError(f"Course ID {added_meeting.course_id} already exists.")
                    availability_changes = (
                        {
                            availability_teacher: {
                                (problem.days.index(availability_day), slot)
                                for slot in unavailable_slots
                            }
                        }
                        if availability_teacher != "None" and unavailable_slots
                        else None
                    )
                    original_schedule = dict(st.session_state["schedule"])
                    repaired, moved = repair_schedule(
                        problem,
                        original_schedule,
                        absent_teacher=None if absent_teacher == "None" else absent_teacher,
                        absent_day=problem.days.index(absent_day_name) if absent_teacher != "None" else None,
                        unavailable_room=None if unavailable_room == "None" else unavailable_room,
                        unavailable_equipment=unavailable_equipment,
                        equipment_room=equipment_room if unavailable_equipment else None,
                        teacher_unavailable_slots=availability_changes,
                        enrollment_updates=enrollment_updates,
                        added_meeting=added_meeting,
                        model=model,
                    )
                    persistent_blocks = {
                        teacher_id: set(slots)
                        for teacher_id, slots in (availability_changes or {}).items()
                    }
                    if absent_teacher != "None":
                        persistent_blocks.setdefault(absent_teacher, set()).update(
                            (problem.days.index(absent_day_name), slot)
                            for slot in range(problem.slots_per_day)
                        )
                    updated_teachers = dict(problem.teachers)
                    all_week_slots = {
                        (day, slot)
                        for day in range(len(problem.days))
                        for slot in range(problem.slots_per_day)
                    }
                    for teacher_id, blocked_slots in persistent_blocks.items():
                        teacher = updated_teachers[teacher_id]
                        available = (
                            set(teacher.available_slots)
                            if teacher.available_slots is not None
                            else set(all_week_slots)
                        )
                        available.difference_update(blocked_slots)
                        updated_teachers[teacher_id] = replace(
                            teacher,
                            available_slots=frozenset(available),
                        )
                    if change_preferences:
                        updated_teachers[preference_teacher] = replace(
                            updated_teachers[preference_teacher],
                            preferred_slots=frozenset(preference_slots),
                        )
                    closed_rooms = set(problem.closed_rooms)
                    if unavailable_room != "None":
                        closed_rooms.add(unavailable_room)
                    unavailable_equipment_by_room = {
                        room_id: set(tags)
                        for room_id, tags in problem.unavailable_equipment_by_room.items()
                    }
                    if unavailable_equipment and equipment_room != "None":
                        unavailable_equipment_by_room.setdefault(equipment_room, set()).add(
                            unavailable_equipment.lower()
                        )
                    updated_problem = replace(
                        problem,
                        teachers=updated_teachers,
                        closed_rooms=frozenset(closed_rooms),
                        unavailable_equipment_by_room={
                            room_id: frozenset(tags)
                            for room_id, tags in unavailable_equipment_by_room.items()
                        },
                        meetings=tuple(
                            replace(
                                meeting,
                                enrollment=enrollment_updates.get(
                                    meeting.meeting_id, meeting.enrollment
                                ),
                            )
                            for meeting in problem.meetings
                        ) + ((added_meeting,) if added_meeting else ()),
                    )
                    preference_optimization = None
                    if change_preferences:
                        preference_optimization = optimize_detailed(
                            updated_problem,
                            repaired,
                            "annealing",
                            model,
                            iterations=160,
                            seed=42,
                            max_runtime_seconds=5,
                        )
                        repaired = preference_optimization.schedule
                        moved = sorted(
                            meeting_id
                            for meeting_id, placement in repaired.items()
                            if original_schedule.get(meeting_id) != placement
                        )
                    errors = validate_schedule(updated_problem, repaired)
                    if errors:
                        raise RuntimeError("Repair failed validation: " + "; ".join(errors[:5]))
                    repair_score = disruption_cost(original_schedule, repaired, updated_problem)
                    st.session_state["pending_repair"] = {
                        "original_schedule": original_schedule,
                        "updated_problem": updated_problem,
                        "repaired": repaired,
                        "moved": moved,
                        "added_meeting": added_meeting,
                        "disruption_cost": repair_score,
                        "preference_optimization": preference_optimization,
                    }
                    st.success("A feasible repair preview is ready. The active timetable has not been changed.")
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
            pending_repair = st.session_state.get("pending_repair")
            if pending_repair:
                preview_problem = pending_repair["updated_problem"]
                preview_schedule = pending_repair["repaired"]
                preview_original = pending_repair["original_schedule"]
                preview_moved = pending_repair["moved"]
                st.markdown("### Repair preview · not yet applied")
                before_metrics = schedule_metrics(preview_problem, preview_original)
                after_metrics = schedule_metrics(preview_problem, preview_schedule)
                preview_metrics = st.columns(5)
                preview_metrics[0].metric(
                    "Disruption cost",
                    f'{pending_repair["disruption_cost"]:.2f}',
                    "lower means fewer/smaller changes",
                )
                preview_metrics[1].metric(
                    "Soft score",
                    soft_score(
                        objective_cost(preview_problem, preview_schedule, model),
                        len(preview_problem.meetings),
                    ),
                    f'from {soft_score(objective_cost(preview_problem, preview_original, model), len(preview_problem.meetings)):.1f}',
                )
                preview_metrics[2].metric(
                    "Student idle periods",
                    int(after_metrics["student_idle_periods"]),
                    f'{int(after_metrics["student_idle_periods"] - before_metrics["student_idle_periods"]):+d} change',
                )
                preview_metrics[3].metric(
                    "Teacher idle periods",
                    int(after_metrics["teacher_idle_periods"]),
                    f'{int(after_metrics["teacher_idle_periods"] - before_metrics["teacher_idle_periods"]):+d} change',
                )
                preview_metrics[4].metric(
                    "Hard violations",
                    len(validate_schedule(preview_problem, preview_schedule)),
                )
                preference_result = pending_repair.get("preference_optimization")
                if preference_result:
                    st.caption(
                        f"Preference-change optimization: objective "
                        f"{preference_result.initial_cost:.2f} → {preference_result.best_cost:.2f}; "
                        f"{preference_result.runtime_seconds:.3f}s measured."
                    )
                meeting_lookup = {
                    meeting.meeting_id: meeting for meeting in preview_problem.meetings
                }
                change_rows = []
                for meeting_id in preview_moved:
                    old = preview_original.get(meeting_id)
                    new = preview_schedule[meeting_id]
                    change_rows.append(
                        {
                            "Class": meeting_lookup[meeting_id].course_name,
                            "Groups": ", ".join(meeting_lookup[meeting_id].groups),
                            "Before": (
                                f"{problem.days[old.day]} {8 + old.start:02d}:00 · {old.room_id}"
                                if old is not None
                                else "New class"
                            ),
                            "After": (
                                f"{preview_problem.days[new.day]} {8 + new.start:02d}:00 · {new.room_id}"
                            ),
                        }
                    )
                if change_rows:
                    render_table(
                        pd.DataFrame(change_rows),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.caption("The repair found no placement changes.")
                apply_col, discard_col, _ = st.columns([1, 1, 2])
                if apply_col.button(
                    "Apply repaired timetable",
                    type="primary",
                    key="apply_pending_repair",
                ):
                    st.session_state["problem"] = preview_problem
                    st.session_state["schedule"] = preview_schedule
                    if pending_repair["added_meeting"]:
                        st.session_state.pop("baseline_schedule", None)
                    st.session_state["generation"] = {
                        "algorithm": "Means-Ends + CSP local repair",
                        "cost": objective_cost(preview_problem, preview_schedule, model),
                        "runtime": 0.0,
                        "nodes": 0,
                    }
                    st.session_state.pop("optimization_result", None)
                    st.session_state.pop("pending_repair", None)
                    st.success("Repaired timetable applied.")
                    st.rerun()
                if discard_col.button("Discard preview", key="discard_pending_repair"):
                    st.session_state.pop("pending_repair", None)
                    st.rerun()
                st.caption(
                    "The preview uses Means–Ends Analysis and bounded CSP repair. "
                    "It is not a guarantee of globally optimal repair."
                )
    with export_col:
        st.markdown("### Share your timetable")
        st.markdown(
            '<div class="panel-sub">Export a clean copy for printing, sharing, or further analysis.</div>',
            unsafe_allow_html=True,
        )
        if not schedule_exists:
            st.info("Generate a timetable to enable exports.")
        else:
            current_schedule = st.session_state["schedule"]
            export_scope = st.selectbox(
                "Schedule to export",
                ("Entire university", "Student group", "Teacher", "Room"),
            )
            export_by = {
                "Entire university": None,
                "Student group": "group",
                "Teacher": "teacher",
                "Room": "room",
            }[export_scope]
            export_options = (
                sorted({group for meeting in problem.meetings for group in meeting.groups})
                if export_by == "group"
                else sorted(problem.teachers)
                if export_by == "teacher"
                else sorted(problem.rooms)
                if export_by == "room"
                else ["All"]
            )
            export_value = (
                st.selectbox(
                    "Select schedule",
                    export_options,
                    format_func=lambda value: (
                        f"{problem.teachers[value].name} ({value})"
                        if export_by == "teacher"
                        else (
                            f"{value} · {problem.rooms[value].room_type} · "
                            f"{problem.rooms[value].capacity} seats"
                            if export_by == "room"
                            else value
                        )
                    ),
                )
                if export_by
                else "All"
            )
            export_suffix = "" if export_by is None else f"_{export_value}"
            rows = schedule_rows(problem, current_schedule)
            if export_by == "group":
                rows = [row for row in rows if export_value in str(row["groups"]).split(", ")]
            elif export_by == "teacher":
                rows = [row for row in rows if row["teacher_id"] == export_value]
            elif export_by == "room":
                rows = [row for row in rows if row["room_id"] == export_value]
            st.download_button(
                "Download Excel workbook",
                export_excel(
                    problem,
                    current_schedule,
                    filter_by=export_by,
                    filter_value=export_value,
                ),
                f"intellischedule_ai_timetable{export_suffix}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.download_button(
                "Download print-ready PDF",
                export_pdf(
                    problem,
                    current_schedule,
                    filter_by=export_by,
                    filter_value=export_value,
                ),
                f"intellischedule_ai_timetable{export_suffix}.pdf",
                "application/pdf",
                use_container_width=True,
            )
            st.download_button(
                "Download CSV data",
                pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                "intellischedule_ai_timetable.csv",
                "text/csv",
                use_container_width=True,
            )
            st.markdown("#### Saved runs")
            try:
                recent = RunStore(ROOT / "timetable.db").recent()
                if recent:
                    render_table(pd.DataFrame(recent), hide_index=True, use_container_width=True)
                else:
                    st.caption("No previous generation runs saved yet.")
            except OSError as exc:
                st.error(f"Could not read the run history: {exc}")

with manage_tab:
    st.markdown("## University data management")
    st.markdown(
        '<div class="panel-sub">Upload validated university tables and optional history using the sidebar. Download templates or inspect the current import below.</div>',
        unsafe_allow_html=True,
    )
    table_issues = st.session_state.get("table_issues", {})
    if table_issues:
        issue_rows = [
            {"Table": table_name, "Issue": issue}
            for table_name, messages in table_issues.items()
            for issue in messages
        ]
        st.warning("Data-quality checks found items to review.")
        render_table(pd.DataFrame(issue_rows), hide_index=True, use_container_width=True)
    elif st.session_state.get("uploaded_tables"):
        st.success("Uploaded required tables passed duplicate-ID and required-field checks.")
    st.markdown("### Input templates")
    template_defs = {
        "courses": ["course_id", "course_name", "subject", "department", "teacher_id", "group_ids", "enrollment", "room_type", "duration", "sessions_per_week", "heavy", "required_equipment"],
        "teachers": ["teacher_id", "name", "max_hours_per_day", "available_slots", "preferred_slots"],
        "rooms": ["room_id", "capacity", "room_type", "equipment"],
        "groups": ["group_id", "name", "size"],
        "attendance": ["date", "subject", "weekday", "time_slot", "enrollment", "course_type", "previous_attendance", "attendance"],
        "enrollment": ["date", "department", "course_type", "historical_enrollment", "offered_sections", "elective", "expected_enrollment"],
        "preference_feedback": ["student_gaps", "daily_load_imbalance", "heavy_adjacent", "teacher_preference_misses", "teacher_gaps", "preference_rating"],
    }
    template_cols = st.columns(4)
    for index, (name, headers) in enumerate(template_defs.items()):
        template_cols[index % 4].download_button(
            f"{name.title()} CSV template",
            pd.DataFrame(columns=headers).to_csv(index=False).encode("utf-8"),
            f"{name}_template.csv",
            "text/csv",
            use_container_width=True,
        )
    st.markdown("### Active dataset")
    st.write(
        f"**{st.session_state.get('dataset_name', 'Campus demo')}** · "
        f"{len(problem.meetings)} meeting variables · "
        f"{len(problem.teachers)} teachers · {len(problem.rooms)} rooms"
    )
    st.caption(
        "Optional attendance history fields: subject, weekday (Mon-Fri), time_slot (0-based), "
        "enrollment, course_type, and attendance (0-1 or 0-100). Optional enrollment history "
        "fields: department, course_type, historical_enrollment, expected_enrollment; date is "
        "used for chronological holdout if supplied."
    )
    ml_report = pd.DataFrame([
        {"Model": "Attendance Random Forest", "Source": model.attendance_source, **model.attendance_metrics},
        {"Model": "Demand Random Forest", "Source": model.demand_source, **model.demand_metrics},
    ])
    report_cols = st.columns(3)
    if schedule_exists:
        constraint_rows = [
            {
                "Constraint": "Hard constraints",
                "Status": "PASS" if not validate_schedule(problem, st.session_state["schedule"]) else "FAIL",
                "Details": f"{hard_constraint_count(problem)} modeled restrictions",
            }
        ] + [
            {"Constraint": f"Violation {index + 1}", "Status": "FAIL", "Details": error}
            for index, error in enumerate(validate_schedule(problem, st.session_state["schedule"]))
        ]
        report_cols[0].download_button(
            "Constraint report CSV",
            pd.DataFrame(constraint_rows).to_csv(index=False).encode("utf-8"),
            "constraint_report.csv",
            "text/csv",
            use_container_width=True,
        )
    report_cols[1].download_button(
        "ML evaluation CSV",
        ml_report.to_csv(index=False).encode("utf-8"),
        "ml_evaluation_metrics.csv",
        "text/csv",
        use_container_width=True,
    )
    evaluation_path = ROOT / "evaluation_report.md"
    if evaluation_path.exists():
        report_cols[2].download_button(
            "Academic evaluation report",
            evaluation_path.read_bytes(),
            "intellischedule_ai_evaluation_report.md",
            "text/markdown",
            use_container_width=True,
        )
    if st.session_state.get("comparison"):
        report_cols[2].download_button(
            "Algorithm results CSV",
            pd.DataFrame(st.session_state["comparison"]).to_csv(index=False).encode("utf-8"),
            "algorithm_comparison.csv",
            "text/csv",
            use_container_width=True,
        )
    if st.session_state.get("uploaded_tables"):
        with st.expander("Inspect imported table row counts"):
            render_table(
                pd.DataFrame([
                    {"Table": table_name, "Rows": len(rows)}
                    for table_name, rows in st.session_state["uploaded_tables"].items()
                ]),
                hide_index=True,
                use_container_width=True,
            )
    store = RunStore(ROOT / "timetable.db")
    runs = store.recent()
    st.markdown("### Recent experiment records")
    render_table(pd.DataFrame(runs), hide_index=True, use_container_width=True)

st.markdown(
    '<div style="text-align:center;color:#a0a8b7;font-size:10px;padding-top:22px">'
    "INTELLISCHEDULE AI · SMART TIMETABLE PLANNER · Synthetic-data ML estimates are illustrative"
    "</div>",
    unsafe_allow_html=True,
)
