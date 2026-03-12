import asyncio

from main import _normalize_estimated_calories, log_meal


def test_normalize_estimated_calories_none():
    assert _normalize_estimated_calories(None) == 0


def test_normalize_estimated_calories_empty_string():
    assert _normalize_estimated_calories("") == 0


def test_normalize_estimated_calories_string_number():
    assert _normalize_estimated_calories("250") == 250


def test_normalize_estimated_calories_float_string():
    assert _normalize_estimated_calories("123.4") == 123


def test_normalize_estimated_calories_int():
    assert _normalize_estimated_calories(180) == 180


def test_log_meal_handles_none_estimated_calories_event_loop():
    """
    Smoke test: ensure log_meal does not raise when estimated_calories is None.
    """

    async def _run():
        result = await log_meal(
            {
                "description": "I ate a caramel layered croissant, give me macros",
                "meal_type": "snack",
                "estimated_calories": None,
            },
            user_id="test-user",
        )
        assert "response" in result

    asyncio.run(_run())

