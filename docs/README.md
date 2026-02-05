# Nutrition Agent API Documentation

Per-endpoint documentation with **request body**, **parameters**, **sample request/response JSON**, and **cURL** examples.

**Base URL:** `{BASE_URL}` (e.g. `http://localhost:8001`). Default port: **8001**.

This agent is typically called by the **Supervisor Agent** via `POST /execute-tool`. Direct CRUD and health endpoints are also available.

---

## Endpoint docs

| Domain | File | Description |
|--------|------|-------------|
| **Health** | [health.md](health.md) | GET /, GET /health, GET /test-db |
| **Execute tool** | [execute-tool.md](execute-tool.md) | POST /execute-tool (log_meal, get_nutrition_summary, create_meal_plan, etc.) |
| **Meal plans & food** | [meal-plans-and-food.md](meal-plans-and-food.md) | Meal plans, food logs, nutrition targets, summaries, async jobs |

---

## Authentication

Endpoints do not require auth headers; the Supervisor or client passes `user_id` in the request body or query where needed.

---

## Interactive docs

- **Swagger UI:** `{BASE_URL}/docs` (if enabled)
- **OpenAPI:** `{BASE_URL}/openapi.json`
