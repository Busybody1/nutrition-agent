# Execute Tool API

The Supervisor Agent calls `POST /execute-tool` with a tool name and parameters. All tools accept `user_id` inside `parameters`.

---

## POST /execute-tool

**Summary:** Execute a nutrition tool by name.

### Request body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| tool_name | string | Yes | One of: `log_meal`, `get_nutrition_summary`, `create_meal_plan`, `create_meal`, `create_recipe` |
| parameters | object | Yes | Tool-specific parameters; include `user_id` |

### Sample request (log_meal)

```json
{
  "tool_name": "log_meal",
  "parameters": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "meal_type": "lunch",
    "food_name": "Grilled chicken salad",
    "calories": 450,
    "protein_g": 35,
    "carbs_g": 20,
    "fat_g": 25,
    "logged_at": "2025-02-05T13:00:00Z"
  }
}
```

### Sample request (get_nutrition_summary)

```json
{
  "tool_name": "get_nutrition_summary",
  "parameters": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "date": "2025-02-05"
  }
}
```

### Sample request (create_meal_plan)

```json
{
  "tool_name": "create_meal_plan",
  "parameters": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "High protein week",
    "start_date": "2025-02-05",
    "end_date": "2025-02-11",
    "days": []
  }
}
```

### cURL example

```bash
curl -X POST "http://localhost:8001/execute-tool" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "log_meal",
    "parameters": {
      "user_id": "550e8400-e29b-41d4-a716-446655440000",
      "meal_type": "breakfast",
      "food_name": "Oatmeal with berries",
      "calories": 320
    }
  }'
```

### Sample response (log_meal)

**Status:** `200 OK`

```json
{
  "log_id": "770e8400-e29b-41d4-a716-446655440002",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "meal_type": "breakfast",
  "food_name": "Oatmeal with berries",
  "calories": 320,
  "logged_at": "2025-02-05T08:00:00Z",
  "message": "Meal logged successfully"
}
```

### Errors

- **400** — Unknown tool name or invalid parameters.
- **500** — Tool execution failed.
