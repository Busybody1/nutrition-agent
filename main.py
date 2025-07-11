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

# Environment variable validation
def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        "DATABASE_URL",
        "GROQ_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
            logger.info("Creating nutrition database engine...")
            from shared.database import get_nutrition_db_engine
            _nutrition_engine = get_nutrition_db_engine()
            logger.info("Successfully created nutrition database engine")
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            logger.error(f"Failed to create nutrition database engine: {error_msg}")
            from sqlalchemy import create_engine
            _nutrition_engine = create_engine("sqlite:///:memory:")
    return _nutrition_engine

def get_fitness_engine():
    global _fitness_engine
    if _fitness_engine is None:
        try:
            logger.info("Creating fitness database engine...")
            from shared.database import get_fitness_db_engine
            _fitness_engine = get_fitness_db_engine()
            logger.info("Successfully created fitness database engine")
        except Exception as e:
            import traceback
            error_msg = str(e) if e else "Unknown error"
            logger.error(f"Failed to create fitness database engine: {error_msg}\n{traceback.format_exc()}")
            # Try to create engine directly using DATABASE_URL
            try:
                import os
                from sqlalchemy import create_engine
                database_url = os.getenv('DATABASE_URL')
                if database_url:
                    if database_url.startswith("postgres://"):
                        database_url = database_url.replace("postgres://", "postgresql://", 1)
                    _fitness_engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=3600)
                    logger.info("Created fitness database engine using DATABASE_URL directly")
                else:
                    _fitness_engine = create_engine("sqlite:///:memory:")
                    logger.warning("No DATABASE_URL found, using in-memory SQLite")
            except Exception as fallback_error:
                logger.error(f"Fallback engine creation also failed: {fallback_error}")
                _fitness_engine = create_engine("sqlite:///:memory:")
    return _fitness_engine

def get_nutrition_session_local():
    global _NutritionSessionLocal
    if _NutritionSessionLocal is None:
        try:
            logger.info("Creating nutrition session local...")
            engine = get_nutrition_engine()
            _NutritionSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            logger.info("Successfully created nutrition session local")
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            logger.error(f"Failed to create nutrition session local: {error_msg}")
            # Don't raise here, let the dependency function handle it
            return None
    return _NutritionSessionLocal

def get_fitness_session_local():
    global _FitnessSessionLocal
    if _FitnessSessionLocal is None:
        try:
            logger.info("Creating fitness session local...")
            engine = get_fitness_engine()
            _FitnessSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            logger.info("Successfully created fitness session local")
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            logger.error(f"Failed to create fitness session local: {error_msg}")
            # Don't raise here, let the dependency function handle it
            return None
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
    
    try:
        # Validate environment variables
        validate_environment()
        logger.info("Environment variables validated successfully")
        
        # Initialize database tables
        from shared.database import init_database
        await init_database()
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Nutrition agent shutting down")

app = FastAPI(title="Nutrition Agent", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In a production environment, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


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
        logger.info("Attempting to connect to nutrition database...")
        session_local = get_nutrition_session_local()
        if session_local is None:
            raise Exception("Failed to create nutrition session local")
        db = session_local()
        logger.info("Successfully connected to nutrition database")
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        import traceback
        error_msg = str(e) if e else "Unknown error"
        logger.error(f"Nutrition database connection error: {error_msg}\n{traceback.format_exc()}")
        # Provide more detailed error information
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            detail_msg = f"Nutrition database connection failed: {error_msg}"
        elif "authentication" in error_msg.lower():
            detail_msg = f"Nutrition database authentication failed: {error_msg}"
        else:
            detail_msg = f"Nutrition database service unavailable: {error_msg}"
        raise HTTPException(status_code=503, detail=detail_msg)

# Use shared DB for all user-specific data (logs, goals, history, etc.)
def get_shared_db():
    try:
        logger.info("Attempting to connect to shared database...")
        # Use the same engine as get_fitness_engine() for consistency
        engine = get_fitness_engine()
        logger.info("Got fitness engine, creating session maker...")
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Created session maker, creating session...")
        db = session_local()
        logger.info("Successfully connected to shared database")
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        import traceback
        error_msg = str(e) if e else "Unknown error"
        logger.error(f"Shared database connection error: {error_msg}\n{traceback.format_exc()}")
        # Provide more detailed error information
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            detail_msg = f"Shared database connection failed: {error_msg}"
        elif "authentication" in error_msg.lower():
            detail_msg = f"Shared database authentication failed: {error_msg}"
        else:
            detail_msg = f"Shared database service unavailable: {error_msg}"
        raise HTTPException(status_code=503, detail=detail_msg)


@app.get("/")
def read_root():
    """Serve the test interface at root URL for easy Heroku access."""
    try:
        from fastapi.responses import FileResponse
        return FileResponse("static/test_interface.html")
    except FileNotFoundError:
        return {
            "message": "Nutrition Agent is running!",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "foods_count": "/foods/count",
                "execute_tool": "/execute-tool",
                "test_interface": "/static/test_interface.html",
            },
        }

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


@app.get("/test/database")
def test_database():
    """Test database connections and basic operations."""
    try:
        results = {}
        
        # Test shared database (DATABASE_URL)
        try:
            engine = get_fitness_engine()
            with engine.connect() as conn:
                # Test basic connection
                result = conn.execute(text("SELECT 1")).scalar()
                results["shared_db_connection"] = f"success (result: {result})"
                
                # Test if food_logs table exists
                table_result = conn.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'food_logs'")).fetchone()
                if table_result:
                    results["shared_db_food_logs_table"] = "exists"
                else:
                    results["shared_db_food_logs_table"] = "missing"
                    
        except Exception as e:
            results["shared_db_connection"] = f"failed: {str(e)}"
            results["shared_db_food_logs_table"] = "unknown"
        
        # Test nutrition database (NUTRITION_DB_URI)
        try:
            engine = get_nutrition_engine()
            with engine.connect() as conn:
                # Test basic connection
                result = conn.execute(text("SELECT 1")).scalar()
                results["nutrition_db_connection"] = f"success (result: {result})"
                
                # Test if foods table exists
                table_result = conn.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'foods'")).fetchone()
                if table_result:
                    results["nutrition_db_foods_table"] = "exists"
                else:
                    results["nutrition_db_foods_table"] = "missing"
                    
        except Exception as e:
            results["nutrition_db_connection"] = f"failed: {str(e)}"
            results["nutrition_db_foods_table"] = "unknown"
        
        return {
            "status": "test_completed",
            "results": results,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "test_failed",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

@app.get("/debug/database")
def debug_database():
    """Debug database connections and configurations."""
    try:
        import os
        from shared.config import get_settings
        
        # Get environment variables
        database_url = os.getenv('DATABASE_URL')
        nutrition_db_uri = os.getenv('NUTRITION_DB_URI')
        
        # Get settings
        settings = get_settings()
        
        # Test connections
        fitness_status = "unknown"
        nutrition_status = "unknown"
        
        try:
            fitness_engine = get_fitness_engine()
            with fitness_engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                fitness_status = f"connected (test result: {result})"
        except Exception as e:
            fitness_status = f"failed: {str(e)}"
        
        try:
            nutrition_engine = get_nutrition_engine()
            with nutrition_engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).scalar()
                nutrition_status = f"connected (test result: {result})"
        except Exception as e:
            nutrition_status = f"failed: {str(e)}"
        
        return {
            "environment": {
                "DATABASE_URL": database_url[:50] + "..." if database_url and len(database_url) > 50 else database_url,
                "NUTRITION_DB_URI": nutrition_db_uri[:50] + "..." if nutrition_db_uri and len(nutrition_db_uri) > 50 else nutrition_db_uri,
            },
            "settings": {
                "database_url": settings.database.url[:50] + "..." if len(settings.database.url) > 50 else settings.database.url,
                "multi_db": {
                    "nutrition_db_uri": settings.multi_db.nutrition_db_uri[:50] + "..." if settings.multi_db.nutrition_db_uri and len(settings.multi_db.nutrition_db_uri) > 50 else settings.multi_db.nutrition_db_uri,
                }
            },
            "connections": {
                "fitness_database": fitness_status,
                "nutrition_database": nutrition_status,
            },
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }


@app.get("/debug/schema")
def debug_schema():
    """Debug database schema to check if tables exist."""
    try:
        # Test shared database schema
        shared_tables = []
        try:
            engine = get_fitness_engine()
            with engine.connect() as conn:
                # Get list of tables
                result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
                tables = [row[0] for row in result.fetchall()]
                shared_tables = tables
        except Exception as e:
            shared_tables = f"Error: {str(e)}"
        
        # Test nutrition database schema
        nutrition_tables = []
        try:
            engine = get_nutrition_engine()
            with engine.connect() as conn:
                # Get list of tables
                result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
                tables = [row[0] for row in result.fetchall()]
                nutrition_tables = tables
        except Exception as e:
            nutrition_tables = f"Error: {str(e)}"
        
        return {
            "shared_database_tables": shared_tables,
            "nutrition_database_tables": nutrition_tables,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
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


@app.get("/test-nutrition-db")
def test_nutrition_db():
    """Test nutrition database connection and basic operations."""
    try:
        engine = get_nutrition_engine()
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT 1")).scalar()
            
            # Test if foods table exists
            table_result = conn.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'foods'")).fetchone()
            if table_result:
                # Get a sample food
                food_result = conn.execute(text("SELECT id, name FROM foods LIMIT 1")).fetchone()
                if food_result:
                    food_data = dict(food_result._mapping) if hasattr(food_result, '_mapping') else dict(zip(['id', 'name'], food_result))
                    return {
                        "status": "success",
                        "connection": "working",
                        "foods_table": "exists",
                        "sample_food": food_data
                    }
                else:
                    return {
                        "status": "success",
                        "connection": "working",
                        "foods_table": "exists",
                        "sample_food": "no foods found"
                    }
            else:
                return {
                    "status": "success",
                    "connection": "working",
                    "foods_table": "missing"
                }
    except Exception as e:
        return {
            "status": "error",
            "connection": "failed",
            "error": str(e)
        }


@app.get("/test-shared-db")
def test_shared_db():
    """Test shared database connection and basic operations."""
    try:
        engine = get_fitness_engine()
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT 1")).scalar()
            
            # Test if food_logs table exists
            table_result = conn.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'food_logs'")).fetchone()
            if table_result:
                return {
                    "status": "success",
                    "connection": "working",
                    "food_logs_table": "exists"
                }
            else:
                return {
                    "status": "success",
                    "connection": "working",
                    "food_logs_table": "missing"
                }
    except Exception as e:
        return {
            "status": "error",
            "connection": "failed",
            "error": str(e)
        }


@app.get("/foods/{food_id}")
def get_food_full_view(food_id: int, db=Depends(get_nutrition_db)):
    rows = db.execute(
        text(
            """
        SELECT
          f.id AS food_id,
          f.name AS food_name,
          f.description,
          f.serving_size,
          f.serving_unit,
          f.serving,
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
        """
        ),
        {"food_id": food_id},
    ).fetchall()

    if not rows:
        return {"error": "Food not found"}

    # Convert rows to dictionaries properly
    converted_rows = []
    for row in rows:
        if hasattr(row, '_mapping'):
            converted_rows.append(dict(row._mapping))
        else:
            # Handle tuple case
            columns = ['food_id', 'food_name', 'description', 'serving_size', 'serving_unit', 'serving', 'created_at', 'brand_id', 'brand_name', 'category_id', 'category_name', 'nutrient_id', 'nutrient_name', 'nutrient_unit', 'amount']
            converted_rows.append(dict(zip(columns, row)))

    food_row = converted_rows[0]
    nutrients = [
        {
            "id": r["nutrient_id"],
            "name": r["nutrient_name"],
            "unit": r["nutrient_unit"],
            "amount": r["amount"],
        }
        for r in converted_rows
        if r["nutrient_id"] is not None
    ]

    return {
        "id": food_row["food_id"],
        "name": food_row["food_name"],
        "description": food_row["description"],
        "serving_size": food_row["serving_size"],
        "serving_unit": food_row["serving_unit"],
        "serving": food_row["serving"],
        "created_at": food_row["created_at"],
        "brand": {"id": food_row["brand_id"], "name": food_row["brand_name"]}
        if food_row["brand_id"]
        else None,
        "category": {"id": food_row["category_id"], "name": food_row["category_name"]}
        if food_row["category_id"]
        else None,
        "nutrients": nutrients,
    }


@app.post("/log-food")
def log_food_endpoint(
    request: Dict[str, Any],
    db_nutrition=Depends(get_nutrition_db),
    db_shared=Depends(get_shared_db)
):
    """Direct endpoint for logging food with proper two-database flow."""
    try:
        # Extract parameters
        user_id = request.get("user_id")
        food_id = request.get("food_id")
        quantity_g = request.get("quantity_g", 100)
        meal_type = request.get("meal_type", "snack")
        consumed_at = request.get("consumed_at", datetime.datetime.utcnow().isoformat())
        notes = request.get("notes", "")
        
        if not user_id or not food_id:
            raise HTTPException(status_code=400, detail="Missing required parameters: user_id and food_id")
        
        # Get food details from nutrition database
        food_details = None
        try:
            food_row = db_nutrition.execute(
                text("SELECT id, name, serving_size, serving_unit, serving, calories, protein_g, carbs_g, fat_g FROM foods WHERE id = :food_id"),
                {"food_id": food_id}
            ).fetchone()
            
            if food_row:
                food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'serving_size', 'serving_unit', 'serving', 'calories', 'protein_g', 'carbs_g', 'fat_g'], food_row))
                logger.info(f"Retrieved food details: {food_details.get('name', 'Unknown')}")
            else:
                raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
        except Exception as e:
            logger.error(f"Error getting food details: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get food details: {str(e)}")
        
        # Calculate nutrition based on quantity
        base_calories = food_details.get("calories", 0)
        base_protein = food_details.get("protein_g", 0)
        base_carbs = food_details.get("carbs_g", 0)
        base_fat = food_details.get("fat_g", 0)
        
        # Calculate actual nutrition based on quantity
        actual_calories = (base_calories * quantity_g) / 100
        actual_protein = (base_protein * quantity_g) / 100
        actual_carbs = (base_carbs * quantity_g) / 100
        actual_fat = (base_fat * quantity_g) / 100
        
        # Create log entry
        import uuid
        log_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "food_item_id": food_id,
            "quantity_g": quantity_g,
            "meal_type": meal_type,
            "consumed_at": consumed_at,
            "notes": notes,
            "actual_nutrition": {
                "calories": round(actual_calories, 1),
                "protein_g": round(actual_protein, 1),
                "carbs_g": round(actual_carbs, 1),
                "fat_g": round(actual_fat, 1)
            }
        }
        
        # Create FoodLogEntry object
        FoodItem, FoodLogEntry, NutritionInfo = get_models()
        entry = FoodLogEntry(**log_entry)
        
        # Log to shared database
        result = log_food_to_calorie_log_with_details(db_nutrition, db_shared, entry)
        
        return {
            "status": "success",
            "message": "Food logged successfully",
            "food_name": food_details.get("name"),
            "quantity_g": quantity_g,
            "calories": actual_calories,
            "log_id": log_entry["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in log_food_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log food: {str(e)}")


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
        return log_food_to_calorie_log_with_details(db_nutrition, db_shared, entry)

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
        # Get food data first, then calculate
        food_data = get_food_nutrition(db_nutrition, food_id)
        return calculate_calories({"food": food_data, "quantity_g": quantity_g})

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
        return get_nutrition_recommendations({"user_id": user_id, "user_profile": {}}, db_shared)

    elif tool == "track_nutrition_goals":
        user_id = params.get("user_id")
        goal_type = params.get("goal_type")
        target_value = params.get("target_value")
        if not user_id or not goal_type or not target_value:
            raise HTTPException(status_code=400, detail="Missing required parameters.")
        return track_nutrition_goals({"user_id": user_id, "goal_type": goal_type, "target_value": target_value}, db_shared)

    elif tool == "meal-plan":
        return create_meal_plan(params, db_nutrition, db_shared)

    elif tool == "calculate-calories":
        return calculate_calories(params)

    elif tool == "nutrition-recommendations":
        return get_nutrition_recommendations(params, db_shared)

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
        SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving, created_at
        FROM foods
        WHERE LOWER(name) LIKE :name
        ORDER BY name
        LIMIT 20
        """
        ),
        {"name": f"%{name.lower()}%"},
    ).fetchall()
    
    # Convert rows to dictionaries properly
    result = []
    for row in rows:
        if hasattr(row, '_mapping'):
            result.append(dict(row._mapping))
        else:
            # Handle tuple case
            columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
            result.append(dict(zip(columns, row)))
    return result


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
    
    # Convert row to dictionary properly
    if hasattr(row, '_mapping'):
        return dict(row._mapping)
    else:
        # Handle tuple case - we need to get column names
        columns = [desc[0] for desc in db.execute(text("SELECT * FROM foods LIMIT 0")).cursor.description]
        return dict(zip(columns, row))


def log_food_to_calorie_log(db, entry: FoodLogEntry):
    """Legacy function - kept for backward compatibility"""
    return log_food_to_calorie_log_with_details(None, db, entry)


def log_food_to_calorie_log_with_details(db_nutrition, db_shared, entry: FoodLogEntry):
    """Log food to calorie log with proper two-database flow."""
    # Validate nutrition data
    if not DataValidator.validate_nutrition_data(entry.actual_nutrition.model_dump()):
        raise HTTPException(status_code=400, detail="Invalid nutrition data")
    
    # Check if food_logs table exists in shared database
    try:
        result = db_shared.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'food_logs'")).fetchone()
        if not result:
            logger.error("food_logs table does not exist in the shared database")
            raise HTTPException(status_code=500, detail="Database schema not properly initialized. Please contact administrator.")
    except Exception as e:
        logger.error(f"Error checking if food_logs table exists: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")
    
    # Get food details from nutrition database (read-only)
    food_details = None
    if db_nutrition:
        try:
            # Check if foods table exists in nutrition database
            result = db_nutrition.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'foods'")).fetchone()
            if result:
                food_row = db_nutrition.execute(
                    text("SELECT serving_size, serving_unit, serving, name FROM foods WHERE id = :food_id"),
                    {"food_id": entry.food_item_id}
                ).fetchone()
                if food_row:
                    # Convert Row to dict
                    food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['serving_size', 'serving_unit', 'serving', 'name'], food_row))
                    logger.info(f"Retrieved food details for food_id {entry.food_item_id}: {food_details.get('name', 'Unknown')}")
                else:
                    logger.warning(f"Food with id {entry.food_item_id} not found in nutrition database")
            else:
                logger.warning("foods table does not exist in nutrition database")
        except Exception as e:
            logger.warning(f"Could not fetch food serving details from nutrition database: {e}")
            # Continue without serving details
    
    # Insert log into shared database
    try:
        db_shared.execute(
            text(
                """
            INSERT INTO food_logs (id, user_id, food_item_id, quantity_g, meal_type, consumed_at, calories, protein_g, carbs_g, fat_g, serving_size, serving_unit, serving, notes, created_at)
            VALUES (:id, :user_id, :food_item_id, :quantity_g, :meal_type, :consumed_at, :calories, :protein_g, :carbs_g, :fat_g, :serving_size, :serving_unit, :serving, :notes, :created_at)
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
                "serving_size": food_details.get("serving_size") if food_details else None,
                "serving_unit": food_details.get("serving_unit") if food_details else None,
                "serving": food_details.get("serving") if food_details else None,
                "notes": entry.notes,
                "created_at": datetime.datetime.utcnow(),
            },
        )
        db_shared.commit()
        logger.info(f"Successfully logged food for user {entry.user_id}, food_id {entry.food_item_id}")
        return {"status": "success", "message": "Food logged successfully"}
    except Exception as e:
        logger.error(f"Error inserting food log: {e}")
        db_shared.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to log food: {str(e)}")


def get_user_calorie_history(db, user_id: Any):
    try:
        # Check if food_logs table exists
        result = db.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'food_logs'")).fetchone()
        if not result:
            logger.warning("food_logs table does not exist, returning empty history")
            return []
        
        rows = db.execute(
            text(
                """
            SELECT id, user_id, food_item_id, quantity_g, meal_type, consumed_at, calories, protein_g, carbs_g, fat_g, serving_size, serving_unit, serving, notes, created_at
            FROM food_logs 
            WHERE user_id = :user_id 
            ORDER BY consumed_at DESC 
            LIMIT 100
            """
            ),
            {"user_id": user_id},
        ).fetchall()
        
        # Convert rows to dictionaries properly
        result = []
        for row in rows:
            if hasattr(row, '_mapping'):
                result.append(dict(row._mapping))
            else:
                # Handle tuple case
                columns = ['id', 'user_id', 'food_item_id', 'quantity_g', 'meal_type', 'consumed_at', 'calories', 'protein_g', 'carbs_g', 'fat_g', 'serving_size', 'serving_unit', 'serving', 'notes', 'created_at']
                result.append(dict(zip(columns, row)))
        return result
    except Exception as e:
        logger.error(f"Error getting user calorie history: {e}")
        return []


# --- Advanced Nutrition Features ---


def search_food_fuzzy(db, name: str):
    """Search food with fuzzy matching for better results."""
    name = DataValidator.sanitize_string(name)

    # Get all foods for fuzzy matching
    rows = db.execute(
        text(
            """
        SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving, created_at
        FROM foods
        ORDER BY name
        LIMIT 1000
        """
        )
    ).fetchall()

    # Convert rows to dictionaries properly
    foods = []
    for row in rows:
        if hasattr(row, '_mapping'):
            foods.append(dict(row._mapping))
        else:
            # Handle tuple case
            columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
            foods.append(dict(zip(columns, row)))

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
def create_meal_plan(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Create a personalized meal plan based on user goals and preferences."""
    try:
        user_id = request.get("user_id")
        daily_calories = request.get("daily_calories", 2000)
        meal_count = request.get("meal_count", 3)
        dietary_restrictions = request.get("dietary_restrictions", [])

        # Get user's food preferences and history from shared database
        user_history = get_user_calorie_history(db_shared, user_id)

        # Simple meal planning algorithm
        meal_plan = generate_meal_plan(
            db_nutrition, daily_calories, meal_count, dietary_restrictions, user_history
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
    SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving, created_at
    FROM foods
    """

    params = {}

    # Add dietary restrictions
    if "vegetarian" in dietary_restrictions:
        query += " WHERE category_id NOT IN (SELECT id FROM categories WHERE name IN ('meat', 'fish', 'poultry'))"
    if "vegan" in dietary_restrictions:
        if "WHERE" in query:
            query += " AND category_id NOT IN (SELECT id FROM categories WHERE name IN ('meat', 'fish', 'poultry', 'dairy', 'eggs'))"
        else:
            query += " WHERE category_id NOT IN (SELECT id FROM categories WHERE name IN ('meat', 'fish', 'poultry', 'dairy', 'eggs'))"

    query += " ORDER BY name LIMIT 20"

    try:
        rows = db.execute(text(query), params).fetchall()
        
        # Convert rows to dictionaries properly
        result = []
        for row in rows:
            if hasattr(row, '_mapping'):
                result.append(dict(row._mapping))
            else:
                # Handle tuple case
                columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
                result.append(dict(zip(columns, row)))
        return result
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

        # Validate food data has required fields
        required_fields = ["calories", "protein_g", "carbs_g", "fat_g"]
        missing_fields = [field for field in required_fields if field not in food_data]
        if missing_fields:
            raise HTTPException(
                status_code=400, 
                detail=f"Food data missing required fields: {missing_fields}"
            )

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to calculate calories: {str(e)}"
        )


@app.post("/nutrition-recommendations")
def get_nutrition_recommendations(
    request: Dict[str, Any], db=Depends(get_shared_db)
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
                SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving, created_at
                FROM foods
                WHERE LOWER(name) LIKE :pattern
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

            # Convert rows to dictionaries properly
            for row in rows:
                if hasattr(row, '_mapping'):
                    results.append(dict(row._mapping))
                else:
                    columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
                    results.append(dict(zip(columns, row)))

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
    if goal_type == "daily_calories" or goal_type == "calories":
        current_value = sum(entry.get("calories", 0) for entry in history)
    elif goal_type == "daily_protein" or goal_type == "protein":
        current_value = sum(entry.get("protein_g", 0) for entry in history)
    elif goal_type == "daily_carbs" or goal_type == "carbs":
        current_value = sum(entry.get("carbs_g", 0) for entry in history)
    elif goal_type == "daily_fat" or goal_type == "fat":
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
        FROM food_logs 
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
        FROM food_logs 
        WHERE meal_type = :meal_type
        AND consumed_at > NOW() - INTERVAL '7 days'
        GROUP BY food_item_id
        ORDER BY frequency DESC
        LIMIT 20
        """
        ),
        {"meal_type": meal_type},
    ).fetchall()

    # Convert rows to dictionaries properly
    recent_foods_dict = []
    for row in recent_foods:
        if hasattr(row, '_mapping'):
            recent_foods_dict.append(dict(row._mapping))
        else:
            recent_foods_dict.append(dict(zip(['food_item_id', 'frequency'], row)))

    popular_foods_dict = []
    for row in popular_foods:
        if hasattr(row, '_mapping'):
            popular_foods_dict.append(dict(row._mapping))
        else:
            popular_foods_dict.append(dict(zip(['food_item_id', 'frequency'], row)))

    # Combine and get food details
    food_ids = list(set([f["food_item_id"] for f in recent_foods_dict + popular_foods_dict]))

    if not food_ids:
        return {"suggestions": [], "message": "No food preferences found"}

    # Get food details
    placeholders = ",".join([":id" + str(i) for i in range(len(food_ids))])
    params = {f"id{i}": food_id for i, food_id in enumerate(food_ids)}

    foods = db.execute(
        text(
            f"""
        SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving
        FROM foods 
        WHERE id IN ({placeholders})
        """
        ),
        params,
    ).fetchall()

    # Convert rows to dictionaries properly
    suggestions = []
    for food in foods:
        if hasattr(food, '_mapping'):
            food_dict = dict(food._mapping)
        else:
            columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving']
            food_dict = dict(zip(columns, food))
        
        # Calculate suggested portion based on target calories (default 100g)
        if target_calories:
            # Use a default calorie value since we don't have calories in the nutrition DB
            default_calories = 100  # Default calories per 100g
            suggested_quantity = min(200, (target_calories / default_calories) * 100)
            food_dict["suggested_quantity_g"] = round(suggested_quantity, 0)
        else:
            food_dict["suggested_quantity_g"] = 100
        suggestions.append(food_dict)

    return {
        "meal_type": meal_type,
        "target_calories": target_calories,
        "suggestions": suggestions[:10],
    }
