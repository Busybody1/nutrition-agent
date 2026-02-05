# Nutrition Agent

**BB AI Agents** — Nutrition, meal planning, food logging, and nutrition advice. Generates meal plans, handles food logs and nutrition targets, and provides daily/weekly summaries with AI (OpenAI).

**Part of the Busybody (BB) ecosystem.** Invoked by the **Supervisor Agent**; uses user DB and optionally a nutrition reference DB (e.g. **food-nutrition-backend** or internal nutrition DB).

---

## Features

- **Meal plans** — AI-generated meal plans
- **Food logging** — Log meals and track intake
- **Nutrition targets** — Set and manage nutrition goals
- **Summaries** — Daily and weekly nutrition summaries
- **AI** — OpenAI for meal planning and advice
- **Batching** — Nutrition-specific batch manager
- **Background jobs** — Celery for longer-running tasks

---

## Tech Stack

- **FastAPI** — REST API
- **PostgreSQL** — User nutrition data; optional nutrition reference DB
- **OpenAI** — Meal plans and nutrition AI
- **Redis** — Caching (optional)
- **Celery** — Background tasks

---

## Quick Start

```bash
cd bb-ai-agents/nutrition-agent
cp config.example.env .env
# Set USER_DATABASE_URI, NUTRITION_DB_URI (optional), OPENAI_API_KEY, REDIS_URL
pip install -r requirements.txt
python main.py
```

Default port: **8001** (override with `PORT` in `.env`).

---

## Main Endpoints

- `POST /execute-tool` — Tool execution (e.g. `generate_meal_plan`, `log_meal`, `get_nutrition_advice`, `analyze_nutrition`) — called by Supervisor
- `GET /health` — Health check
- `GET /test-db` — Database connection test

**Full API docs** (request/response, cURL, parameters): [docs/README.md](docs/README.md)

---

## Configuration

Copy `config.example.env` to `.env`. Key variables:

- `USER_DATABASE_URI` — User DB (meal plans, food logs, targets)
- `NUTRITION_DB_URI` — Optional nutrition reference DB
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TIMEOUT`
- `REDIS_URL` — Optional
- `PORT` — Server port (default 8001)

---

## Project Structure

```
nutrition-agent/
├── main.py              # FastAPI app and routes
├── tasks.py             # Celery tasks
├── models.py            # User, meal plans, food logs, targets
├── schemas.py           # Pydantic schemas
├── utils/
│   ├── batching/        # Nutrition batch manager, cache, optimizer
│   ├── config/          # Settings
│   ├── database/        # DB connections
│   └── nutrition_database_service.py
├── config.example.env
├── requirements.txt
└── Procfile / Dockerfile
```

---

## Documentation

- [Nutrition Agent documentation](../docs/NUTRITION_AGENT_DOCUMENTATION.md) — Full API and tools
- [BB AI Agents overview](../README.md) — System architecture and other agents
