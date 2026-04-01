# RCAAgent-Env

OpenEnv-compatible FastAPI environment for simulated microservice **root cause analysis (RCA)**. Agents call `reset` / `step` / `state`, then `POST /grader`. A **Gemini** baseline is exposed as **`GET /baseline`**.

## Layout

```text
models.py           # Pydantic: Action, Observation, EnvironmentState, RCAReport, …
client.py           # httpx client for local/Space API
server/
  app.py            # FastAPI app (uvicorn server.app:app)
  environment.py    # RCAEnvironment: reset(), step(), state()
  grader.py         # Deterministic + Gemini rubric
  llm.py            # One-time Gemini configure + shared GenerativeModel handles
baseline/agent.py   # Baseline loop (env.step + Gemini)
scenarios/*.json
openenv.yaml
Dockerfile
```

## Secrets

Use **`GOOGLE_API_KEY`** (Google AI Studio / Gemini API). Do not commit keys.

```text
cp .env.example .env
# edit .env → GOOGLE_API_KEY=...
```

Loaded from the **project root** via `python-dotenv` in `server/app.py`.

## API (OpenEnv)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/tasks` | Tasks list + **JSON Schema** for `Action` |
| POST | `/reset/{difficulty}` | New session (`easy` / `medium` / `hard`) |
| POST | `/step/{difficulty}` | Body: `Action` JSON |
| GET | `/state/{difficulty}` | Current `EnvironmentState` |
| POST | `/grader` | Body: `{ difficulty, report, queries_used? }` → scores + **`final_score`** in `[0,1]` |
| GET | `/baseline?difficulty=easy` | Runs baseline agent (Gemini + `step()`); **async** |
| POST | `/baseline` | Same as GET, JSON body `{ "difficulty": "easy" }` |

## Run locally

```bash
cd rcaagent-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

Windows: if you hit socket errors, try `py -3.11 run_dev.py` (Selector loop + optional `fd` bind) or another port.

## Hugging Face Spaces

1. Create a **Docker** Space pointing at this repo.
2. **Settings → Repository secrets**: add **`GOOGLE_API_KEY`** (value = your Gemini API key). Spaces inject secrets as environment variables.
3. The **Dockerfile** exposes **8000** and runs:

   `uvicorn server.app:app --host 0.0.0.0 --port 8000`

4. In Space **Settings → Dev mode**, set the **container port** to **8000** if the UI asks for it.
5. Open the public Space URL; use **`/docs`** for Swagger.

CPU-only; no GPU required.

## OpenEnv validation

```bash
openenv validate
# (from repo root, with openenv CLI installed per hackathon instructions)
```

Ensure you have called **`POST /reset/{difficulty}`** before **`/step`** or **`/state`**.

## Client smoke test

```python
from client import RCAAgentEnvClient
c = RCAAgentEnvClient("http://127.0.0.1:8000")
print(c.tasks())
print(c.reset("easy"))
```

## Use Cases
- AI Agent Evaluation
- Research in Autonomous Debugging
- Hackathon Projects
- LLM + Systems Integration Experiments
