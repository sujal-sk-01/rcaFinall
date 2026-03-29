"""Google Gemini baseline agent: drives env.step() in a reproducible loop."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

import json
import os
import re
from typing import Any

import google.generativeai as genai

from models import Action, ActionType
from server.environment import RCAEnvironment
from server.grader import grade
from server.llm import get_baseline_model


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found in response")


def _system_prompt(services: list[str]) -> str:
    svc = ", ".join(services)
    return f"""You are an on-call SRE debugging a production incident in a microservice system.

Available services: {svc}

You may take ONE action per turn. Respond with a single JSON object (no markdown) matching this schema:
{{
  "action_type": one of [
    "query_metrics",
    "query_logs",
    "pull_traces",
    "query_dependencies",
    "form_hypothesis",
    "submit_rca"
  ],
  "target_service": "<service name or null>",
  "hypothesis": "<string or null; required when action_type is form_hypothesis>",
  "rca_report": null OR {{
    "root_cause_service": "<string>",
    "root_cause_type": "<string>",
    "affected_services": ["<service>", "..."],
    "causal_chain": ["<root first, then downstream, ...>"],
    "summary": "<string>",
    "fix_recommendation": "<string>",
    "confidence": <float 0.0-1.0>
  }}
}}

Rules:
- For query_metrics, query_logs, pull_traces, query_dependencies: set target_service to a valid service name.
- Start by investigating api_gateway, then follow dependencies and anomalies.
- Be strategic: correlate metrics, logs, and traces before forming conclusions.
- Use form_hypothesis to record intermediate theories.
- Only use submit_rca when you are confident; rca_report must be complete.
- Your entire reply MUST be valid JSON parsable as an Action.
"""


def _parse_action(content: str) -> Action:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    data = extract_json(text)
    return Action.model_validate(data)


def _generate_action(model: Any, messages: list[dict[str, str]], prompt: str) -> str:
    try:
        gc = genai.types.GenerationConfig(temperature=0)
        resp = model.generate_content(prompt, generation_config=gc)
    except Exception:
        resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def run_baseline(difficulty: str) -> dict[str, Any]:
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "difficulty": difficulty,
            "steps": 0,
            "history": [],
            "report": None,
            "scores": None,
            "error": "GOOGLE_API_KEY is not set",
        }

    model = get_baseline_model()
    if model is None:
        return {
            "difficulty": difficulty,
            "steps": 0,
            "history": [],
            "report": None,
            "scores": None,
            "error": "Gemini client not configured",
        }

    env = RCAEnvironment(difficulty)
    st = env.reset()
    services = list(st.services.keys())

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt(services)},
        {
            "role": "user",
            "content": f"Incident alert:\n{st.alert}\n\nBegin investigation. Output only JSON Action.",
        },
    ]

    history: list[dict[str, Any]] = []
    steps = 0
    last_report = None

    for _ in range(25):
        steps += 1
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        try:
            raw = _generate_action(model, messages, prompt)
        except Exception as exc:
            return {
                "difficulty": difficulty,
                "steps": steps,
                "history": history,
                "report": None,
                "scores": None,
                "error": f"Gemini API error: {exc!s}",
            }

        try:
            action = _parse_action(raw)
        except Exception:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid JSON or schema. Reply again with ONLY one JSON object matching the Action schema.",
                },
            )
            continue

        messages.append({"role": "assistant", "content": raw})
        obs = env.step(action)
        history.append(
            {
                "action": action.model_dump(mode="json"),
                "observation": obs.model_dump(mode="json"),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"Observation:\n{obs.model_dump_json()}",
            }
        )

        if action.action_type == ActionType.submit_rca and action.rca_report is not None:
            last_report = action.rca_report
            break

    scores: dict[str, Any] | None = None
    if last_report is not None:
        scores = grade(last_report, env.state(), env.raw_scenario)

    return {
        "difficulty": difficulty,
        "steps": steps,
        "history": history,
        "report": last_report.model_dump(mode="json") if last_report else None,
        "scores": scores,
    }
