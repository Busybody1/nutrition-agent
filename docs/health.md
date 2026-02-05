# Health API

Base path: `/`. No authentication required.

---

## GET /

**Summary:** Root endpoint; service info.

### Sample response

**Status:** `200 OK`

```json
{
  "message": "Nutrition Agent API",
  "version": "1.0.0",
  "status": "running"
}
```

---

## GET /health

**Summary:** Health check.

### Sample response

**Status:** `200 OK`

```json
{
  "status": "healthy",
  "timestamp": "2025-02-05T12:00:00Z",
  "database": "connected",
  "version": "1.0.0"
}
```

---

## GET /test-db

**Summary:** Test database connectivity (user DB, nutrition reference DB, etc.).

### Sample response

**Status:** `200 OK`

```json
{
  "user_db": "connected",
  "nutrition_db": "connected",
  "workout_db": "disconnected",
  "timestamp": "2025-02-05T12:00:00Z"
}
```
