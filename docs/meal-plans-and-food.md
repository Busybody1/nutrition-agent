# Meal Plans & Food API

CRUD for meal plans, food logs, nutrition targets, and summaries. All endpoints use `user_id` (query or path as documented).

---

## Meal plans

### POST /api/meal-plans

**Summary:** Create a meal plan. Query: `user_id`. Body: `name`, `description`, `start_date`, `end_date`, `is_active`, `is_template`, optional `days` (with `meals` and `items`).

### GET /api/meal-plans/{user_id}

**Summary:** List meal plans for a user.

### PUT /api/meal-plans/{meal_plan_id}

**Summary:** Update a meal plan. Body: partial meal plan fields.

### DELETE /api/meal-plans/{meal_plan_id}

**Summary:** Delete a meal plan.

---

## Food logs

### POST /api/food-logs

**Summary:** Create a food log. Query: `user_id`. Body: meal and food fields.

### GET /api/food-logs/{user_id}

**Summary:** List food logs for a user.

### PUT /api/food-logs/{log_id}

**Summary:** Update a food log.

### DELETE /api/food-logs/{log_id}

**Summary:** Delete a food log.

---

## Nutrition targets

### POST /api/nutrition-targets

**Summary:** Create a nutrition target. Query: `user_id`.

### GET /api/nutrition-targets/{user_id}

**Summary:** List nutrition targets for a user.

### PUT /api/nutrition-targets/{target_id}

**Summary:** Update a target.

### DELETE /api/nutrition-targets/{target_id}

**Summary:** Delete a target.

---

## Summaries and stats

### GET /api/nutrition-summary/{user_id}

**Summary:** Get nutrition summary for a user. Query: optional `date` or date range.

### GET /api/nutrition-stats/{user_id}/weekly

**Summary:** Weekly nutrition statistics.

---

## Nutrition database (reference)

### GET /nutrition-database/status

**Summary:** Nutrition reference database connection status.

### GET /nutrition-database/foods

**Summary:** List foods from reference DB (query params for filter/pagination).

### GET /nutrition-database/search

**Summary:** Search reference foods. Query: `q`, `limit`, etc.

### GET /nutrition-database/ingredients/verify

**Summary:** Verify ingredients against reference DB.

---

## Async jobs

### POST /generate-meal-plan-async

**Summary:** Start async meal plan generation. Returns job ID.

### POST /analyze-nutrition-async

**Summary:** Start async nutrition analysis. Returns job ID.

### POST /generate-nutrition-recommendations-async

**Summary:** Start async nutrition recommendations. Returns job ID.

### GET /meal-plan/job/{job_id}

### GET /nutrition-analysis/job/{job_id}

### GET /nutrition-recommendations/job/{job_id}

**Summary:** Poll async job result.
