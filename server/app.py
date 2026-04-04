"""FastAPI ASGI application: Hugging Face Spaces + OpenEnv HTTP API."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

import asyncio
import json
import logging
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from models import Action, EnvironmentState, Observation, RCAReport, ServiceMetrics
from server.environment import RCAEnvironment
from server.grader import grade
from server.llm import ensure_gemini_configured

logger = logging.getLogger(__name__)

SCENARIOS: dict[str, dict] = {}
sessions: dict[str, RCAEnvironment] = {}


def _load_all_scenarios() -> None:
    global SCENARIOS
    SCENARIOS.clear()
    base = _PROJECT_ROOT / "scenarios"
    for key, name in (("easy", "easy.json"), ("medium", "medium.json"), ("hard", "hard.json")):
        with (base / name).open(encoding="utf-8") as f:
            SCENARIOS[key] = json.load(f)


def _services_from_scenario(scenario: dict) -> dict[str, ServiceMetrics]:
    out: dict[str, ServiceMetrics] = {}
    for name, data in scenario.get("service_metrics", {}).items():
        out[name] = ServiceMetrics(
            latency_ms=float(data["latency_ms"]),
            error_rate=float(data["error_rate"]),
            cpu_percent=float(data["cpu_percent"]),
            memory_percent=float(data["memory_percent"]),
            status=data["status"],
        )
    return out


def _environment_state_for_grader(
    scenario: dict,
    report: RCAReport,
    queries_used: int,
) -> EnvironmentState:
    return EnvironmentState(
        scenario_id=scenario["scenario_id"],
        difficulty=scenario["difficulty"],
        alert=scenario["alert"],
        services=_services_from_scenario(scenario),
        queries_used=queries_used,
        max_queries=int(scenario.get("max_queries", 25)),
        hypotheses=[],
        rca_submitted=True,
        submitted_report=report,
    )


app = FastAPI(title="RCAAgent-Env", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    _load_all_scenarios()
    ensure_gemini_configured()
    logger.info(
        "RCAAgent-Env ready (scenarios=%s, GOOGLE_API_KEY set=%s)",
        list(SCENARIOS.keys()),
        bool(os.getenv("GOOGLE_API_KEY")),
    )


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    ui_path = _PROJECT_ROOT / "server" / "ui.html"
    html = ui_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


class TaskMeta(BaseModel):
    id: str
    name: str
    difficulty: str
    max_steps: int
    optimal_steps: int


class TasksResponse(BaseModel):
    tasks: list[TaskMeta]
    action_schema: dict


class GraderRequest(BaseModel):
    difficulty: str
    report: RCAReport
    queries_used: int | None = Field(default=None)


@app.get("/tasks", response_model=TasksResponse)
def list_tasks() -> TasksResponse:
    out: list[TaskMeta] = []
    for diff in ("easy", "medium", "hard"):
        sc = SCENARIOS[diff]
        gt = sc.get("ground_truth", {})
        out.append(
            TaskMeta(
                id=sc["scenario_id"],
                name=f"RCA {sc['scenario_id']}",
                difficulty=sc["difficulty"],
                max_steps=int(sc.get("max_queries", 25)),
                optimal_steps=int(gt.get("optimal_queries", 0)),
            )
        )
    return TasksResponse(tasks=out, action_schema=Action.model_json_schema())


@app.post("/reset/{difficulty}", response_model=EnvironmentState)
def reset_session(difficulty: str) -> EnvironmentState:
    d = difficulty.lower()
    if d not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown difficulty: {difficulty}")
    env = RCAEnvironment(d)
    sessions[d] = env
    return env.reset()


def _get_session(difficulty: str) -> RCAEnvironment:
    d = difficulty.lower()
    env = sessions.get(d)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"No session for {difficulty}; call POST /reset/{difficulty} first",
        )
    return env


@app.post("/step/{difficulty}", response_model=Observation)
def take_step(difficulty: str, body: Action) -> Observation:
    env = _get_session(difficulty)
    return env.step(body)


@app.get("/state/{difficulty}", response_model=EnvironmentState)
def get_state(difficulty: str) -> EnvironmentState:
    env = _get_session(difficulty)
    return env.state()


def _run_grade(body: GraderRequest) -> dict:
    d = body.difficulty.lower()
    if d not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown difficulty: {body.difficulty}")
    scenario = SCENARIOS[d]
    qu = body.queries_used if body.queries_used is not None else 0
    state = _environment_state_for_grader(scenario, body.report, qu)
    return grade(body.report, state, scenario)


@app.post("/grader")
def post_grader(body: GraderRequest) -> dict:
    return _run_grade(body)


@app.get("/baseline")
async def get_baseline(
    difficulty: str = Query(..., description="Task difficulty: easy, medium, or hard"),
) -> dict:
    from baseline.agent import run_baseline

    d = difficulty.lower()
    if d not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown difficulty: {difficulty}")
    return await asyncio.to_thread(run_baseline, d)


class BaselineBody(BaseModel):
    difficulty: str = Field(description="easy, medium, or hard")


@app.post("/baseline")
async def post_baseline(body: BaselineBody) -> dict:
    from baseline.agent import run_baseline

    d = body.difficulty.lower()
    if d not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown difficulty: {body.difficulty}")
    return await asyncio.to_thread(run_baseline, d)
