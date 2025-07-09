import logging
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import datetime
import difflib
import math
from contextlib import asynccontextmanager

# Import models and utilities
from shared import FoodLogEntry, DataValidator

# Set up logger - reduced for faster startup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()
from fastapi import FastAPI, Depends, HTTPException, Body

# Lazy database initialization
_nutrition_engine = None
_fitness_engine = None
_NutritionSessionLocal = None
_FitnessSessionLocal = None
_SessionLocal = None

def get_nutrition_engine():
    global _nutrition_engine
    if _nutrition_engine is None:
        try:
            from shared.database import get_nutrition_db_engine
            _nutrition_engine = get_nutrition_db_engine()
        except Exception as e:
            logger.error(f"Failed to create nutrition database engine: {e}")
            from sqlalchemy import create_engine
            _nutrition_engine = create_engine("sqlite:///:memory:")
    return _nutrition_engine

def get_fitness_engine():
    global _fitness_engine
    if _fitness_engine is None:
        try:
            from shared.database import get_fitness_db_engine
            _fitness_engine = get_fitness_db_engine()
        except Exception as e:
            logger.error(f"Failed to create fitness database engine: {e}")
            from sqlalchemy import create_engine
            _fitness_engine = create_engine("sqlite:///:memory:")
    return _fitness_engine

def get_nutrition_session_local():
    global _NutritionSessionLocal
    if _NutritionSessionLocal is None:
        _NutritionSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_nutrition_engine())
    return _NutritionSessionLocal

def get_fitness_session_local():
    global _FitnessSessionLocal
    if _FitnessSessionLocal is None:
        _FitnessSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_fitness_engine())
    return _FitnessSessionLocal

def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_nutrition_engine())
    return _SessionLocal

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup - minimal initialization to avoid boot timeout
    logger.info("Starting nutrition-agent...")
    
    yield
    
    # Shutdown
    logger.info("Nutrition agent shutting down")

app = FastAPI(title="Nutrition Agent", lifespan=lifespan)


def get_db():
    try:
        session_local = get_session_local()
        db = session_local()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")


# --- Dependency functions ---
# Use nutrition DB for food reference data only

def get_nutrition_db():
    try:
        session_local = get_nutrition_session_local()
        db = session_local()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Nutrition database connection error: {e}")
        raise HTTPException(status_code=503, detail="Nutrition database service unavailable")

# Use shared DB for all user-specific data (logs, goals, history, etc.)
def get_shared_db():
    try:
        from shared.database import get_fitness_db_engine
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=get_fitness_db_engine())
        db = session_local()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Shared database connection error: {e}")
        raise HTTPException(status_code=503, detail="Shared database service unavailable")


@app.get("/health")
def health_check():
    """Fast health check that doesn't block startup."""
    return {
        "status": "healthy", 
        "agent": "nutrition",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.get("/health/detailed")
def detailed_health_check():
    """Detailed health check with database testing."""
    try:
        # Test main database connection (for writes)
        fitness_engine = get_fitness_engine()
        with fitness_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Test nutrition database connection (for reads) - optional
        nutrition_status = "not_configured"
        try:
            nutrition_engine = get_nutrition_engine()
            with nutrition_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            nutrition_status = "connected"
        except Exception as e:
            logger.warning(f"Nutrition database not accessible: {e}")
            nutrition_status = "unavailable"
        
        return {
            "status": "healthy", 
            "agent": "nutrition",
            "main_database": "connected",
            "nutrition_database": nutrition_status,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "degraded", 
            "agent": "nutrition",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }


# --- Update all queries to use 'foods' instead of 'food_items' ---
@app.get("/foods/count")
def food_count(db=Depends(get_nutrition_db)):
    try:
        result = db.execute(text("SELECT COUNT(*) FROM foods")).scalar()
        return {"food_count": result}
    except Exception as e:
        logger.error(f"Error getting food count: {e}")
        return {"food_count": 0, "error": "Database temporarily unavailable"}


@app.get("/foods/{food_id}")
def get_food_full_view(food_id: int, db=Depends(get_nutrition_db)):
    rows = db.execute(
        """
        SELECT
          f.id AS food_id,
          f.name AS food_name,
          f.description,
          f.created_at,
          b.id AS brand_id,
          b.name AS brand_name,
          c.id AS category_id,
          c.name AS category_name,
          n.id AS nutrient_id,
          n.name AS nutrient_name,
          n.unit AS nutrient_unit,
          fn.amount
        FROM foods f
        LEFT JOIN brands b ON f.brand_id = b.id
        LEFT JOIN categories c ON f.category_id = c.id
        LEFT JOIN food_nutrients fn ON f.id = fn.food_id
        LEFT JOIN nutrients n ON fn.nutrient_id = n.id
        WHERE f.id = :food_id
        """,
        {"food_id": food_id},
    ).fetchall()

    if not rows:
        return {"error": "Food not found"}

    food_row = rows[0]
    nutrients = [
        {
            "id": r["nutrient_id"],
            "name": r["nutrient_name"],
            "unit": r["nutrient_unit"],
            "amount": r["amount"],
        }
        for r in rows
        if r["nutrient_id"] is not None
    ]

    return {
        "id": food_row["food_id"],
        "name": food_row["food_name"],
        "description": food_row["description"],
        "created_at": food_row["created_at"],
        "brand": {"id": food_row["brand_id"], "name": food_row["brand_name"]}
        if food_row["brand_id"]
        else None,
        "category": {"id": food_row["category_id"], "name": food_row["category_name"]}
        if food_row["category_id"]
        else None,
        "nutrients": nutrients,
    }


# Import models when needed
def get_models():
    from shared.models import FoodItem, FoodLogEntry, NutritionInfo
    return FoodItem, FoodLogEntry, NutritionInfo

class ExecuteToolRequest(BaseModel):
    tool: str
    params: Dict[str, Any]


@app.post("/execute-tool")
def execute_tool(
    request: ExecuteToolRequest,
    db_nutrition=Depends(get_nutrition_db),
    db_shared=Depends(get_shared_db),
):
    tool = request.tool
    params = request.params

    if tool == "search_food_by_name":
        name = params.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing 'name' parameter.")
        return search_food_by_name(db_nutrition, name)

    elif tool == "get_food_nutrition":
        food_id = params.get("food_id")
        if not food_id:
            raise HTTPException(status_code=400, detail="Missing 'food_id' parameter.")
        return get_food_nutrition(db_nutrition, food_id)

    elif tool == "log_food_to_calorie_log":
        try:
            FoodItem, FoodLogEntry, NutritionInfo = get_models()
            entry = FoodLogEntry(**params)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid log entry: {e}")
        return log_food_to_calorie_log(db_shared, entry)

    elif tool == "get_user_calorie_history":
        user_id = params.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="Missing 'user_id' parameter.")
        return get_user_calorie_history(db_shared, user_id)

    # Advanced Features
    elif tool == "search_food_fuzzy":
        name = params.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Missing 'name' parameter.")
        return search_food_fuzzy(db_nutrition, name)

    elif tool == "calculate_calories":
        food_id = params.get("food_id")
        quantity_g = params.get("quantity_g")
        if not food_id or not quantity_g:
            raise HTTPException(
                status_code=400, detail="Missing 'food_id' or 'quantity_g' parameter."
            )
        return calculate_calories(db_nutrition, food_id, quantity_g)

    elif tool == "get_meal_suggestions":
        user_id = params.get("user_id")
        meal_type = params.get("meal_type")
        target_calories = params.get("target_calories")
        if not user_id or not meal_type:
            raise HTTPException(status_code=400, detail="Missing required parameters.")
        return get_meal_suggestions(db_nutrition, user_id, meal_type, target_calories)

    elif tool == "get_nutrition_recommendations":
        user_id = params.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="Missing 'user_id' parameter.")
        return get_nutrition_recommendations(db_shared, user_id)

    elif tool == "track_nutrition_goals":
        user_id = params.get("user_id")
        goal_type = params.get("goal_type")
        target_value = params.get("target_value")
        if not user_id or not goal_type or not target_value:
            raise HTTPException(status_code=400, detail="Missing required parameters.")
        return track_nutrition_goals(db_shared, user_id, goal_type, target_value)

    elif tool == "meal-plan":
        return create_meal_plan(params, db_nutrition)

    elif tool == "calculate-calories":
        return calculate_calories(params)

    elif tool == "nutrition-recommendations":
        return get_nutrition_recommendations(params, db_nutrition)

    elif tool == "fuzzy-search":
        return fuzzy_search_food(params, db_nutrition)

    elif tool == "nutrition-goals":
        return track_nutrition_goals(params, db_shared)

    else:
        raise HTTPException(status_code=400, detail="Unknown tool.")


# --- Core Nutrition Functions ---
def search_food_by_name(db, name: str):
    name = DataValidator.sanitize_string(name)
    rows = db.execute(
        text(
            """
        SELECT id, name, brand_id, calories, category_id, created_at
        FROM foods
        WHERE LOWER(name) LIKE :name
        ORDER BY name
        LIMIT 20
        """
        ),
        {"name": f"%{name.lower()}%"},
    ).fetchall()
    return [dict(row) for row in rows]


def get_food_nutrition(db, food_id: Any):
    row = db.execute(
        text(
            """
        SELECT * FROM foods WHERE id = :food_id
        """
        ),
        {"food_id": food_id},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Food not found")
    return dict(row)


def log_food_to_calorie_log(db, entry: FoodLogEntry):
    # Validate nutrition data
    if not DataValidator.validate_nutrition_data(entry.actual_nutrition.model_dump()):
        raise HTTPException(status_code=400, detail="Invalid nutrition data")
    # Insert log
    db.execute(
        text(
            """
        INSERT INTO food_log_entries (id, user_id, food_item_id, quantity_g, meal_type, consumed_at, calories, protein_g, carbs_g, fat_g, notes, created_at)
        VALUES (:id, :user_id, :food_item_id, :quantity_g, :meal_type, :consumed_at, :calories, :protein_g, :carbs_g, :fat_g, :notes, :created_at)
        """
        ),
        {
            "id": str(entry.id),
            "user_id": str(entry.user_id),
            "food_item_id": str(entry.food_item_id),
            "quantity_g": entry.quantity_g,
            "meal_type": entry.meal_type.value
            if hasattr(entry.meal_type, "value")
            else entry.meal_type,
            "consumed_at": entry.consumed_at,
            "calories": entry.actual_nutrition.calories,
            "protein_g": entry.actual_nutrition.protein_g,
            "carbs_g": entry.actual_nutrition.carbs_g,
            "fat_g": entry.actual_nutrition.fat_g,
            "notes": entry.notes,
            "created_at": entry.created_at or datetime.datetime.utcnow(),
        },
    )
    db.commit()
    return {"status": "success"}


def get_user_calorie_history(db, user_id: Any):
    rows = db.execute(
        text(
            """
        SELECT * FROM food_log_entries WHERE user_id = :user_id ORDER BY consumed_at DESC LIMIT 100
        """
        ),
        {"user_id": user_id},
    ).fetchall()
    return [dict(row) for row in rows]


# --- Advanced Nutrition Features ---


def search_food_fuzzy(db, name: str):
    """Search food with fuzzy matching for better results."""
    name = DataValidator.sanitize_string(name)

    # Get all foods for fuzzy matching
    rows = db.execute(
        text(
            """
        SELECT id, name, brand_id, calories, category_id, created_at
        FROM foods
        ORDER BY name
        LIMIT 1000
        """
        )
    ).fetchall()

    foods = [dict(row) for row in rows]

    # Use difflib for fuzzy matching
    matches = []
    for food in foods:
        similarity = difflib.SequenceMatcher(
            None, name.lower(), food["name"].lower()
        ).ratio()
        if similarity > 0.3:  # Threshold for similarity
            food["similarity"] = similarity
            matches.append(food)

    # Sort by similarity and return top 20
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches[:20]


@app.post("/meal-plan")
def create_meal_plan(request: Dict[str, Any], db=Depends(get_nutrition_db)):
    """Create a personalized meal plan based on user goals and preferences."""
    try:
        user_id = request.get("user_id")
        daily_calories = request.get("daily_calories", 2000)
        meal_count = request.get("meal_count", 3)
        dietary_restrictions = request.get("dietary_restrictions", [])

        # Get user's food preferences and history
        user_history = get_user_calorie_history(db, user_id)

        # Simple meal planning algorithm
        meal_plan = generate_meal_plan(
            db, daily_calories, meal_count, dietary_restrictions, user_history
        )

        return {
            "status": "success",
            "meal_plan": meal_plan,
            "total_calories": sum(meal["calories"] for meal in meal_plan),
            "meals": len(meal_plan),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create meal plan: {e}")


def generate_meal_plan(
    db,
    daily_calories: int,
    meal_count: int,
    dietary_restrictions: List[str],
    user_history: List[Dict],
):
    """Generate a meal plan based on user preferences and history."""
    calories_per_meal = daily_calories // meal_count

    # Get popular foods from user history
    popular_foods = get_popular_foods_from_history(user_history)

    meal_plan = []
    for i in range(meal_count):
        meal_type = ["breakfast", "lunch", "dinner"][i % 3]

        # Find suitable foods for this meal
        suitable_foods = find_suitable_foods(
            db, meal_type, calories_per_meal, dietary_restrictions, popular_foods
        )

        if suitable_foods:
            selected_food = suitable_foods[0]  # Simple selection for now
            meal_plan.append(
                {
                    "meal_type": meal_type,
                    "food": selected_food,
                    "calories": selected_food["calories"],
                    "quantity_g": 100,
                }
            )

    return meal_plan


def get_popular_foods_from_history(user_history: List[Dict]) -> List[str]:
    """Extract popular foods from user's eating history."""
    food_counts = {}
    for entry in user_history:
        food_name = entry.get("name", "").lower()
        if food_name:
            food_counts[food_name] = food_counts.get(food_name, 0) + 1

    # Return top 10 most eaten foods
    return sorted(food_counts.items(), key=lambda x: x[1], reverse=True)[:10]


def find_suitable_foods(
    db,
    meal_type: str,
    target_calories: int,
    dietary_restrictions: List[str],
    popular_foods: List[tuple],
):
    """Find foods suitable for a specific meal type and calorie target."""
    # Build query based on restrictions
    query = """
    SELECT id, name, brand_id, calories, category_id, created_at
    FROM foods
    WHERE calories BETWEEN :min_calories AND :max_calories
    """

    params = {
        "min_calories": target_calories * 0.7,
        "max_calories": target_calories * 1.3,
    }

    # Add dietary restrictions
    if "vegetarian" in dietary_restrictions:
        query += " AND category_id NOT IN (SELECT id FROM categories WHERE name IN ('meat', 'fish', 'poultry'))"
    if "vegan" in dietary_restrictions:
        query += " AND category_id NOT IN (SELECT id FROM categories WHERE name IN ('meat', 'fish', 'poultry', 'dairy', 'eggs'))"

    query += " ORDER BY calories LIMIT 20"

    try:
        rows = db.execute(text(query), params).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        # Log the error and raise it instead of returning dummy data
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.post("/calculate-calories")
def calculate_calories(request: Dict[str, Any]):
    """Calculate calories for a given food and quantity."""
    try:
        food_data = request.get("food")
        quantity_g = request.get("quantity_g", 100)

        if not food_data:
            raise HTTPException(status_code=400, detail="Food data required")

        # Calculate nutrition based on quantity
        calories = (food_data.get("calories", 0) * quantity_g) / 100
        protein_g = (food_data.get("protein_g", 0) * quantity_g) / 100
        carbs_g = (food_data.get("carbs_g", 0) * quantity_g) / 100
        fat_g = (food_data.get("fat_g", 0) * quantity_g) / 100

        return {
            "status": "success",
            "quantity_g": quantity_g,
            "calculated_nutrition": {
                "calories": round(calories, 1),
                "protein_g": round(protein_g, 1),
                "carbs_g": round(carbs_g, 1),
                "fat_g": round(fat_g, 1),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to calculate calories: {e}"
        )


@app.post("/nutrition-recommendations")
def get_nutrition_recommendations(
    request: Dict[str, Any], db=Depends(get_nutrition_db)
):
    """Get personalized nutrition recommendations based on user data."""
    try:
        user_id = request.get("user_id")
        user_profile = request.get("user_profile", {})

        # Get user's recent eating history
        recent_history = get_user_calorie_history(db, user_id)

        # Analyze nutrition patterns
        nutrition_analysis = analyze_nutrition_patterns(recent_history, user_profile)

        # Generate recommendations
        recommendations = generate_nutrition_recommendations(
            nutrition_analysis, user_profile
        )

        return {
            "status": "success",
            "analysis": nutrition_analysis,
            "recommendations": recommendations,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to get recommendations: {e}"
        )


def analyze_nutrition_patterns(history: List[Dict], user_profile: Dict) -> Dict:
    """Analyze user's nutrition patterns from history."""
    if not history:
        return {"error": "No history data available"}

    total_calories = sum(entry.get("calories", 0) for entry in history)
    total_protein = sum(entry.get("protein_g", 0) for entry in history)
    total_carbs = sum(entry.get("carbs_g", 0) for entry in history)
    total_fat = sum(entry.get("fat_g", 0) for entry in history)

    avg_calories = total_calories / len(history)
    avg_protein = total_protein / len(history)
    avg_carbs = total_carbs / len(history)
    avg_fat = total_fat / len(history)

    return {
        "average_daily_calories": round(avg_calories, 1),
        "average_daily_protein_g": round(avg_protein, 1),
        "average_daily_carbs_g": round(avg_carbs, 1),
        "average_daily_fat_g": round(avg_fat, 1),
        "protein_percentage": round((avg_protein * 4 / avg_calories) * 100, 1)
        if avg_calories > 0
        else 0,
        "carbs_percentage": round((avg_carbs * 4 / avg_calories) * 100, 1)
        if avg_calories > 0
        else 0,
        "fat_percentage": round((avg_fat * 9 / avg_calories) * 100, 1)
        if avg_calories > 0
        else 0,
    }


def generate_nutrition_recommendations(analysis: Dict, user_profile: Dict) -> List[str]:
    """Generate personalized nutrition recommendations."""
    recommendations = []

    if "error" in analysis:
        recommendations.append(
            "Start logging your meals to get personalized recommendations."
        )
        return recommendations

    # Calorie recommendations
    target_calories = user_profile.get("daily_calorie_target", 2000)
    current_calories = analysis.get("average_daily_calories", 0)

    if current_calories < target_calories * 0.8:
        recommendations.append(
            "Consider increasing your daily calorie intake to meet your goals."
        )
    elif current_calories > target_calories * 1.2:
        recommendations.append(
            "Consider reducing your daily calorie intake to meet your goals."
        )

    # Protein recommendations
    protein_pct = analysis.get("protein_percentage", 0)
    if protein_pct < 10:
        recommendations.append(
            "Increase protein intake to support muscle health and recovery."
        )
    elif protein_pct > 35:
        recommendations.append(
            "Consider reducing protein intake to maintain balanced nutrition."
        )

    # Carb recommendations
    carbs_pct = analysis.get("carbs_percentage", 0)
    if carbs_pct < 40:
        recommendations.append("Consider increasing carbohydrate intake for energy.")
    elif carbs_pct > 65:
        recommendations.append(
            "Consider reducing carbohydrate intake for better balance."
        )

    # Fat recommendations
    fat_pct = analysis.get("fat_percentage", 0)
    if fat_pct < 20:
        recommendations.append(
            "Include healthy fats in your diet for essential nutrients."
        )
    elif fat_pct > 40:
        recommendations.append("Consider reducing fat intake for better balance.")

    if not recommendations:
        recommendations.append("Great job! Your nutrition is well-balanced.")

    return recommendations


@app.post("/fuzzy-search")
def fuzzy_search_food(request: Dict[str, Any], db=Depends(get_nutrition_db)):
    """Search for foods using fuzzy matching for better results."""
    try:
        query = request.get("query", "")
        limit = request.get("limit", 20)

        if not query:
            raise HTTPException(status_code=400, detail="Search query required")

        # Enhanced search with multiple patterns
        search_patterns = [f"%{query}%", f"%{query.lower()}%", f"%{query.title()}%"]

        results = []
        for pattern in search_patterns:
            rows = db.execute(
                text(
                    """
                SELECT id, name, brand_id, calories, category_id, created_at
                FROM foods
                WHERE LOWER(name) LIKE :pattern OR LOWER(brand_id) LIKE :pattern OR LOWER(category_id) LIKE :pattern
                ORDER BY 
                    CASE 
                        WHEN LOWER(name) = :exact THEN 1
                        WHEN LOWER(name) LIKE :exact_start THEN 2
                        WHEN LOWER(name) LIKE :exact_contains THEN 3
                        ELSE 4
                    END,
                    name
                LIMIT :limit
                """
                ),
                {
                    "pattern": pattern,
                    "exact": query.lower(),
                    "exact_start": f"{query.lower()}%",
                    "exact_contains": f"%{query.lower()}%",
                    "limit": limit,
                },
            ).fetchall()

            results.extend([dict(row) for row in rows])

        # Remove duplicates and limit results
        seen_ids = set()
        unique_results = []
        for result in results:
            if result["id"] not in seen_ids and len(unique_results) < limit:
                seen_ids.add(result["id"])
                unique_results.append(result)

        return {
            "status": "success",
            "results": unique_results,
            "count": len(unique_results),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search failed: {e}")


@app.post("/nutrition-goals")
def track_nutrition_goals(request: Dict[str, Any], db=Depends(get_shared_db)):
    """Track and analyze progress towards nutrition goals."""
    try:
        user_id = request.get("user_id")
        goal_type = request.get("goal_type", "daily_calories")
        target_value = request.get("target_value")

        if not user_id or not target_value:
            raise HTTPException(
                status_code=400, detail="User ID and target value required"
            )

        # Get today's nutrition data
        today_history = get_user_calorie_history(db, user_id)

        # Calculate current progress
        progress = calculate_goal_progress(today_history, goal_type, target_value)

        return {
            "status": "success",
            "goal_type": goal_type,
            "target_value": target_value,
            "current_value": progress["current_value"],
            "progress_percentage": progress["progress_percentage"],
            "remaining": progress["remaining"],
            "goal_status": progress["status"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to track goals: {e}")


def calculate_goal_progress(
    history: List[Dict], goal_type: str, target_value: float
) -> Dict:
    """Calculate progress towards a specific nutrition goal."""
    if goal_type == "daily_calories":
        current_value = sum(entry.get("calories", 0) for entry in history)
    elif goal_type == "daily_protein":
        current_value = sum(entry.get("protein_g", 0) for entry in history)
    elif goal_type == "daily_carbs":
        current_value = sum(entry.get("carbs_g", 0) for entry in history)
    elif goal_type == "daily_fat":
        current_value = sum(entry.get("fat_g", 0) for entry in history)
    else:
        raise ValueError(f"Unknown goal type: {goal_type}")

    progress_percentage = (
        min(100, (current_value / target_value) * 100) if target_value > 0 else 0
    )
    remaining = max(0, target_value - current_value)

    return {
        "current_value": round(current_value, 1),
        "progress_percentage": round(progress_percentage, 1),
        "remaining": round(remaining, 1),
        "status": "on_track" if progress_percentage >= 80 else "needs_attention",
    }


def get_meal_suggestions(
    db, user_id: str, meal_type: str, target_calories: Optional[float] = None
):
    """Get meal suggestions based on user preferences and calorie targets."""
    # Get user's recent food preferences
    recent_foods = db.execute(
        text(
            """
        SELECT food_item_id, COUNT(*) as frequency
        FROM food_log_entries 
        WHERE user_id = :user_id 
        AND consumed_at > NOW() - INTERVAL '30 days'
        GROUP BY food_item_id
        ORDER BY frequency DESC
        LIMIT 10
        """
        ),
        {"user_id": user_id},
    ).fetchall()

    # Get popular foods for this meal type
    popular_foods = db.execute(
        text(
            """
        SELECT food_item_id, COUNT(*) as frequency
        FROM food_log_entries 
        WHERE meal_type = :meal_type
        AND consumed_at > NOW() - INTERVAL '7 days'
        GROUP BY food_item_id
        ORDER BY frequency DESC
        LIMIT 20
        """
        ),
        {"meal_type": meal_type},
    ).fetchall()

    # Combine and get food details
    food_ids = list(set([f["food_item_id"] for f in recent_foods + popular_foods]))

    if not food_ids:
        return {"suggestions": [], "message": "No food preferences found"}

    # Get food details
    placeholders = ",".join([":id" + str(i) for i in range(len(food_ids))])
    params = {f"id{i}": food_id for i, food_id in enumerate(food_ids)}

    foods = db.execute(
        text(
            f"""
        SELECT id, name, brand_id, calories, category_id
        FROM foods 
        WHERE id IN ({placeholders})
        """
        ),
        params,
    ).fetchall()

    suggestions = []
    for food in foods:
        food_dict = dict(food)
        # Calculate suggested portion based on target calories
        if target_calories:
            suggested_quantity = min(200, (target_calories / food["calories"]) * 100)
            food_dict["suggested_quantity_g"] = round(suggested_quantity, 0)
        else:
            food_dict["suggested_quantity_g"] = 100
        suggestions.append(food_dict)

    return {
        "meal_type": meal_type,
        "target_calories": target_calories,
        "suggestions": suggestions[:10],
    }
