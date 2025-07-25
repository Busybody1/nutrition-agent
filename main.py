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
import openai

# Import models and utilities
from shared import FoodLogEntry, DataValidator
from shared.config import get_settings

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

from fastapi import FastAPI, Depends, HTTPException, Body, Request
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


@app.get("/sample-food")
def get_sample_food(db=Depends(get_nutrition_db)):
    """Get a sample food for testing purposes."""
    try:
        food_row = db.execute(
            text("SELECT id, name, calories, protein_g, carbs_g, fat_g FROM foods LIMIT 1")
        ).fetchone()
        
        if food_row:
            food_data = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'calories', 'protein_g', 'carbs_g', 'fat_g'], food_row))
            return {
                "status": "success",
                "sample_food": food_data
            }
        else:
            return {
                "status": "error",
                "message": "No foods found in database"
            }
    except Exception as e:
        return {
            "status": "error",
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
          n.category AS nutrient_category,
          fn.amount
        FROM foods f
        LEFT JOIN brands b ON f.brand_id = b.id
        LEFT JOIN categories c ON f.category_id = c.id
        LEFT JOIN food_nutrients fn ON f.id = fn.food_id
        LEFT JOIN nutrients n ON fn.nutrient_id = n.id
        WHERE f.id = :food_id
        ORDER BY n.name
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
            columns = ['food_id', 'food_name', 'description', 'serving_size', 'serving_unit', 'serving', 'created_at', 'brand_id', 'brand_name', 'category_id', 'category_name', 'nutrient_id', 'nutrient_name', 'nutrient_unit', 'nutrient_category', 'amount']
            converted_rows.append(dict(zip(columns, row)))

    food_row = converted_rows[0]
    
    # Create comprehensive nutrients array and nutrition summary
    nutrients = []
    nutrition_summary = {
        "calories": 0,
        "protein_g": 0,
        "carbs_g": 0,
        "fat_g": 0,
        "fiber_g": 0,
        "sugar_g": 0,
        "sodium_mg": 0,
        "cholesterol_mg": 0,
        "vitamin_a_iu": 0,
        "vitamin_c_mg": 0,
        "vitamin_d_iu": 0,
        "calcium_mg": 0,
        "iron_mg": 0
    }
    
    for r in converted_rows:
        if r["nutrient_id"] is not None:
            amount = r["amount"] or 0
            nutrient_name_lower = r["nutrient_name"].lower()
            
            # Add to comprehensive nutrients array
            nutrients.append({
                "id": r["nutrient_id"],
                "name": r["nutrient_name"],
                "unit": r["nutrient_unit"],
                "amount": amount,
                "category": r.get("nutrient_category", "general")
            })
            
            # Also populate summary for backward compatibility
            if 'calorie' in nutrient_name_lower or 'energy' in nutrient_name_lower:
                nutrition_summary['calories'] = amount
            elif 'protein' in nutrient_name_lower:
                nutrition_summary['protein_g'] = amount
            elif 'carbohydrate' in nutrient_name_lower or 'carb' in nutrient_name_lower:
                nutrition_summary['carbs_g'] = amount
            elif 'fat' in nutrient_name_lower and 'total' in nutrient_name_lower:
                nutrition_summary['fat_g'] = amount
            elif 'fat' in nutrient_name_lower:
                nutrition_summary['fat_g'] = amount
            elif 'fiber' in nutrient_name_lower:
                nutrition_summary['fiber_g'] = amount
            elif 'sugar' in nutrient_name_lower:
                nutrition_summary['sugar_g'] = amount
            elif 'sodium' in nutrient_name_lower:
                nutrition_summary['sodium_mg'] = amount
            elif 'cholesterol' in nutrient_name_lower:
                nutrition_summary['cholesterol_mg'] = amount
            elif 'vitamin a' in nutrient_name_lower:
                nutrition_summary['vitamin_a_iu'] = amount
            elif 'vitamin c' in nutrient_name_lower:
                nutrition_summary['vitamin_c_mg'] = amount
            elif 'vitamin d' in nutrient_name_lower:
                nutrition_summary['vitamin_d_iu'] = amount
            elif 'calcium' in nutrient_name_lower:
                nutrition_summary['calcium_mg'] = amount
            elif 'iron' in nutrient_name_lower:
                nutrition_summary['iron_mg'] = amount

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
        "nutrients": nutrients,  # Comprehensive nutrient array
        "nutrition_summary": nutrition_summary,  # Backward compatibility
        "total_nutrients": len(nutrients)
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
        
        # Convert food_id to string if it's an integer
        if isinstance(food_id, int):
            food_id = str(food_id)
        
        # Get food details from nutrition database using the correct schema
        food_details = None
        try:
            # First get basic food info
            food_row = db_nutrition.execute(
                text("SELECT id, name, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                {"food_id": food_id}
            ).fetchone()
            
            if not food_row:
                raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
            
            food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'serving_size', 'serving_unit', 'serving'], food_row))
            
            # Get nutrition data from food_nutrients table
            nutrition_rows = db_nutrition.execute(
                text("""
                    SELECT n.name as nutrient_name, fn.amount, n.unit
                    FROM food_nutrients fn
                    JOIN nutrients n ON fn.nutrient_id = n.id
                    WHERE fn.food_id = :food_id
                """),
                {"food_id": food_id}
            ).fetchall()
            
            # Convert nutrition data to a dictionary
            nutrition_data = {}
            for row in nutrition_rows:
                if hasattr(row, '_mapping'):
                    nutrient = dict(row._mapping)
                else:
                    nutrient = dict(zip(['nutrient_name', 'amount', 'unit'], row))
                
                nutrient_name = nutrient['nutrient_name'].lower()
                amount = nutrient['amount'] or 0
                
                # Map nutrient names to our expected fields
                if 'calorie' in nutrient_name or 'energy' in nutrient_name:
                    nutrition_data['calories'] = amount
                elif 'protein' in nutrient_name:
                    nutrition_data['protein_g'] = amount
                elif 'carbohydrate' in nutrient_name or 'carb' in nutrient_name:
                    nutrition_data['carbs_g'] = amount
                elif 'fat' in nutrient_name and 'total' in nutrient_name:
                    nutrition_data['fat_g'] = amount
                elif 'fat' in nutrient_name:
                    nutrition_data['fat_g'] = amount
            
            # Set default values if not found
            nutrition_data.setdefault('calories', 0)
            nutrition_data.setdefault('protein_g', 0)
            nutrition_data.setdefault('carbs_g', 0)
            nutrition_data.setdefault('fat_g', 0)
            
            # Add nutrition data to food_details
            food_details.update(nutrition_data)
            
            logger.info(f"Retrieved food details: {food_details.get('name', 'Unknown')}")
            
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
        
        # Create log entry with UUID for food_item_id (nutrition DB uses int, shared DB expects UUID)
        import uuid
        # Generate a UUID based on the nutrition database food ID to maintain consistency
        food_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"nutrition_food_{food_id}")
        
        log_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "food_item_id": str(food_uuid),  # Use UUID format for shared database
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


@app.post("/log-food-simple")
def log_food_simple(
    request: Dict[str, Any],
    db_nutrition=Depends(get_nutrition_db),
    db_shared=Depends(get_shared_db)
):
    """Simple endpoint for logging food with minimal parameters."""
    try:
        # Extract minimal parameters
        user_id = request.get("user_id")
        food_id = request.get("food_id")
        quantity_g = request.get("quantity_g", 100)
        meal_type = request.get("meal_type", "snack")
        
        if not user_id or not food_id:
            raise HTTPException(status_code=400, detail="Missing required parameters: user_id and food_id")
        
        # Convert food_id to string if it's an integer
        if isinstance(food_id, int):
            food_id = str(food_id)
        
        # Get food details from nutrition database using the correct schema
        try:
            # First get basic food info
            food_row = db_nutrition.execute(
                text("SELECT id, name, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                {"food_id": food_id}
            ).fetchone()
            
            if not food_row:
                raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
            
            food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'serving_size', 'serving_unit', 'serving'], food_row))
            
            # Get nutrition data from food_nutrients table
            nutrition_rows = db_nutrition.execute(
                text("""
                    SELECT n.name as nutrient_name, fn.amount, n.unit
                    FROM food_nutrients fn
                    JOIN nutrients n ON fn.nutrient_id = n.id
                    WHERE fn.food_id = :food_id
                """),
                {"food_id": food_id}
            ).fetchall()
            
            # Convert nutrition data to a dictionary
            nutrition_data = {}
            for row in nutrition_rows:
                if hasattr(row, '_mapping'):
                    nutrient = dict(row._mapping)
                else:
                    nutrient = dict(zip(['nutrient_name', 'amount', 'unit'], row))
                
                nutrient_name = nutrient['nutrient_name'].lower()
                amount = nutrient['amount'] or 0
                
                # Map nutrient names to our expected fields
                if 'calorie' in nutrient_name or 'energy' in nutrient_name:
                    nutrition_data['calories'] = amount
                elif 'protein' in nutrient_name:
                    nutrition_data['protein_g'] = amount
                elif 'carbohydrate' in nutrient_name or 'carb' in nutrient_name:
                    nutrition_data['carbs_g'] = amount
                elif 'fat' in nutrient_name and 'total' in nutrient_name:
                    nutrition_data['fat_g'] = amount
                elif 'fat' in nutrient_name:
                    nutrition_data['fat_g'] = amount
            
            # Set default values if not found
            nutrition_data.setdefault('calories', 0)
            nutrition_data.setdefault('protein_g', 0)
            nutrition_data.setdefault('carbs_g', 0)
            nutrition_data.setdefault('fat_g', 0)
            
            # Add nutrition data to food_details
            food_details.update(nutrition_data)
            
        except Exception as e:
            logger.error(f"Error getting food details: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get food details: {str(e)}")
        
        # Calculate nutrition based on quantity
        base_calories = food_details.get("calories", 0)
        base_protein = food_details.get("protein_g", 0)
        base_carbs = food_details.get("carbs_g", 0)
        base_fat = food_details.get("fat_g", 0)
        
        actual_calories = (base_calories * quantity_g) / 100
        actual_protein = (base_protein * quantity_g) / 100
        actual_carbs = (base_carbs * quantity_g) / 100
        actual_fat = (base_fat * quantity_g) / 100
        
        # Create log entry with UUID for food_item_id (nutrition DB uses int, shared DB expects UUID)
        import uuid
        # Generate a UUID based on the nutrition database food ID to maintain consistency
        food_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"nutrition_food_{food_id}")
        
        log_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "food_item_id": str(food_uuid),  # Use UUID format for shared database
            "quantity_g": quantity_g,
            "meal_type": meal_type,
            "consumed_at": datetime.datetime.utcnow().isoformat(),
            "notes": "",
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
        logger.error(f"Error in log_food_simple: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log food: {str(e)}")


@app.post("/test-food-logging")
def test_food_logging(
    request: Dict[str, Any],
    db_nutrition=Depends(get_nutrition_db),
    db_shared=Depends(get_shared_db)
):
    """Test endpoint to demonstrate proper food logging with ID format handling."""
    try:
        # Extract parameters
        user_id = request.get("user_id")
        nutrition_food_id = request.get("nutrition_food_id")  # Integer ID from nutrition DB
        quantity_g = request.get("quantity_g", 100)
        meal_type = request.get("meal_type", "snack")
        
        if not user_id or nutrition_food_id is None:
            raise HTTPException(status_code=400, detail="Missing required parameters: user_id and nutrition_food_id")
        
        # Convert nutrition database integer ID to UUID for shared database
        food_uuid = convert_nutrition_id_to_uuid(nutrition_food_id)
        
        # Get food details from nutrition database using the original integer ID and correct schema
        try:
            # First get basic food info
            food_row = db_nutrition.execute(
                text("SELECT id, name, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                {"food_id": nutrition_food_id}
            ).fetchone()
            
            if not food_row:
                raise HTTPException(status_code=404, detail=f"Food with id {nutrition_food_id} not found in nutrition database")
            
            food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'serving_size', 'serving_unit', 'serving'], food_row))
            
            # Get nutrition data from food_nutrients table
            nutrition_rows = db_nutrition.execute(
                text("""
                    SELECT n.name as nutrient_name, fn.amount, n.unit
                    FROM food_nutrients fn
                    JOIN nutrients n ON fn.nutrient_id = n.id
                    WHERE fn.food_id = :food_id
                """),
                {"food_id": nutrition_food_id}
            ).fetchall()
            
            # Convert nutrition data to a dictionary
            nutrition_data = {}
            for row in nutrition_rows:
                if hasattr(row, '_mapping'):
                    nutrient = dict(row._mapping)
                else:
                    nutrient = dict(zip(['nutrient_name', 'amount', 'unit'], row))
                
                nutrient_name = nutrient['nutrient_name'].lower()
                amount = nutrient['amount'] or 0
                
                # Map nutrient names to our expected fields
                if 'calorie' in nutrient_name or 'energy' in nutrient_name:
                    nutrition_data['calories'] = amount
                elif 'protein' in nutrient_name:
                    nutrition_data['protein_g'] = amount
                elif 'carbohydrate' in nutrient_name or 'carb' in nutrient_name:
                    nutrition_data['carbs_g'] = amount
                elif 'fat' in nutrient_name and 'total' in nutrient_name:
                    nutrition_data['fat_g'] = amount
                elif 'fat' in nutrient_name:
                    nutrition_data['fat_g'] = amount
            
            # Set default values if not found
            nutrition_data.setdefault('calories', 0)
            nutrition_data.setdefault('protein_g', 0)
            nutrition_data.setdefault('carbs_g', 0)
            nutrition_data.setdefault('fat_g', 0)
            
            # Add nutrition data to food_details
            food_details.update(nutrition_data)
            
        except Exception as e:
            logger.error(f"Error getting food details: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get food details: {str(e)}")
        
        # Calculate nutrition based on quantity
        base_calories = food_details.get("calories", 0)
        base_protein = food_details.get("protein_g", 0)
        base_carbs = food_details.get("carbs_g", 0)
        base_fat = food_details.get("fat_g", 0)
        
        actual_calories = (base_calories * quantity_g) / 100
        actual_protein = (base_protein * quantity_g) / 100
        actual_carbs = (base_carbs * quantity_g) / 100
        actual_fat = (base_fat * quantity_g) / 100
        
        # Create log entry with proper UUID format
        import uuid
        log_entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "food_item_id": food_uuid,  # Use the converted UUID
            "quantity_g": quantity_g,
            "meal_type": meal_type,
            "consumed_at": datetime.datetime.utcnow().isoformat(),
            "notes": f"Test log - Original nutrition DB ID: {nutrition_food_id}",
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
            "message": "Food logged successfully with proper ID format handling",
            "food_name": food_details.get("name"),
            "nutrition_food_id": nutrition_food_id,
            "food_uuid": food_uuid,
            "quantity_g": quantity_g,
            "calories": actual_calories,
            "log_id": log_entry["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test_food_logging: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log food: {str(e)}")

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
            # Handle the case where the client sends incomplete data
            user_id = params.get("user_id")
            food_id = params.get("food_item_id") or params.get("food_id")
            quantity_g = params.get("quantity_g", 100)
            meal_type = params.get("meal_type", "snack")
            consumed_at = params.get("consumed_at")
            notes = params.get("notes", "")
            
            if not user_id or not food_id:
                raise HTTPException(status_code=400, detail="Missing required parameters: user_id and food_item_id")
            
            # Convert food_id to string if it's an integer
            if isinstance(food_id, int):
                food_id = str(food_id)
            
            # Set consumed_at to current time if not provided
            if not consumed_at:
                consumed_at = datetime.datetime.utcnow().isoformat()
            
            # Get food details from nutrition database using the correct schema
            try:
                # First get basic food info
                food_row = db_nutrition.execute(
                    text("SELECT id, name, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                    {"food_id": food_id}
                ).fetchone()
                
                if not food_row:
                    raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
                
                food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'serving_size', 'serving_unit', 'serving'], food_row))
                
                # Get nutrition data from food_nutrients table
                nutrition_rows = db_nutrition.execute(
                    text("""
                        SELECT n.name as nutrient_name, fn.amount, n.unit
                        FROM food_nutrients fn
                        JOIN nutrients n ON fn.nutrient_id = n.id
                        WHERE fn.food_id = :food_id
                    """),
                    {"food_id": food_id}
                ).fetchall()
                
                # Convert nutrition data to a dictionary
                nutrition_data = {}
                for row in nutrition_rows:
                    if hasattr(row, '_mapping'):
                        nutrient = dict(row._mapping)
                    else:
                        nutrient = dict(zip(['nutrient_name', 'amount', 'unit'], row))
                    
                    nutrient_name = nutrient['nutrient_name'].lower()
                    amount = nutrient['amount'] or 0
                    
                    # Map nutrient names to our expected fields
                    if 'calorie' in nutrient_name or 'energy' in nutrient_name:
                        nutrition_data['calories'] = amount
                    elif 'protein' in nutrient_name:
                        nutrition_data['protein_g'] = amount
                    elif 'carbohydrate' in nutrient_name or 'carb' in nutrient_name:
                        nutrition_data['carbs_g'] = amount
                    elif 'fat' in nutrient_name and 'total' in nutrient_name:
                        nutrition_data['fat_g'] = amount
                    elif 'fat' in nutrient_name:
                        nutrition_data['fat_g'] = amount
                
                # Set default values if not found
                nutrition_data.setdefault('calories', 0)
                nutrition_data.setdefault('protein_g', 0)
                nutrition_data.setdefault('carbs_g', 0)
                nutrition_data.setdefault('fat_g', 0)
                
                # Add nutrition data to food_details
                food_details.update(nutrition_data)
                
                logger.info(f"Retrieved food details: {food_details.get('name', 'Unknown')}")
                
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
            
            # Create complete log entry with UUID for food_item_id (nutrition DB uses int, shared DB expects UUID)
            import uuid
            # Generate a UUID based on the nutrition database food ID to maintain consistency
            food_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"nutrition_food_{food_id}")
            
            log_entry = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "food_item_id": str(food_uuid),  # Use UUID format for shared database
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
            logger.error(f"Error in log_food_to_calorie_log: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to log food: {str(e)}")

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
        return get_meal_suggestions(db_nutrition, db_shared, user_id, meal_type, target_calories)

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
def convert_nutrition_id_to_uuid(nutrition_id: int) -> str:
    """Convert nutrition database integer ID to UUID format for shared database."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nutrition_food_{nutrition_id}"))

def convert_uuid_to_nutrition_id(food_uuid: str) -> int:
    """Extract original nutrition database ID from UUID (if possible)."""
    try:
        # This is a simplified approach - in practice, you might want to maintain a mapping table
        # For now, we'll return None since we can't reliably extract the original ID
        return None
    except Exception:
        return None

def search_food_by_name(db, name: str):
    name = DataValidator.sanitize_string(name)
    logger.info(f"Searching for food with name: '{name}'")
    
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
    
    logger.info(f"Found {len(rows)} foods matching '{name}'")
    
    # Convert rows to structured format with comprehensive nutrients array
    result = []
    for row in rows:
        if hasattr(row, '_mapping'):
            food_data = dict(row._mapping)
        else:
            # Handle tuple case
            columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
            food_data = dict(zip(columns, row))
        
        logger.info(f"Processing food: ID={food_data['id']}, Name='{food_data['name']}'")
        
        # Get comprehensive nutrition data for this food from food_nutrients table
        try:
            nutrition_rows = db.execute(
                text("""
                    SELECT 
                        n.id as nutrient_id, 
                        n.name as nutrient_name, 
                        fn.amount, 
                        n.unit as nutrient_unit,
                        n.category as nutrient_category
                    FROM food_nutrients fn
                    JOIN nutrients n ON fn.nutrient_id = n.id
                    WHERE fn.food_id = :food_id
                    ORDER BY n.name
                """),
                {"food_id": food_data['id']}
            ).fetchall()
            
            # Convert nutrition data to comprehensive nutrients array format
            nutrients = []
            nutrition_summary = {
                "calories": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
                "sugar_g": 0,
                "sodium_mg": 0,
                "cholesterol_mg": 0,
                "vitamin_a_iu": 0,
                "vitamin_c_mg": 0,
                "vitamin_d_iu": 0,
                "calcium_mg": 0,
                "iron_mg": 0
            }
            
            for nutrition_row in nutrition_rows:
                if hasattr(nutrition_row, '_mapping'):
                    nutrient = dict(nutrition_row._mapping)
                else:
                    nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit', 'nutrient_category'], nutrition_row))
                
                amount = nutrient['amount'] or 0
                nutrient_name_lower = nutrient['nutrient_name'].lower()
                
                # Add to comprehensive nutrients array
                nutrients.append({
                    "id": nutrient['nutrient_id'],
                    "name": nutrient['nutrient_name'],
                    "unit": nutrient['nutrient_unit'],
                    "amount": amount,
                    "category": nutrient.get('nutrient_category', 'general')
                })
                
                # Also populate summary for backward compatibility
                if 'calorie' in nutrient_name_lower or 'energy' in nutrient_name_lower:
                    nutrition_summary['calories'] = amount
                elif 'protein' in nutrient_name_lower:
                    nutrition_summary['protein_g'] = amount
                elif 'carbohydrate' in nutrient_name_lower or 'carb' in nutrient_name_lower:
                    nutrition_summary['carbs_g'] = amount
                elif 'fat' in nutrient_name_lower and 'total' in nutrient_name_lower:
                    nutrition_summary['fat_g'] = amount
                elif 'fat' in nutrient_name_lower:
                    nutrition_summary['fat_g'] = amount
                elif 'fiber' in nutrient_name_lower:
                    nutrition_summary['fiber_g'] = amount
                elif 'sugar' in nutrient_name_lower:
                    nutrition_summary['sugar_g'] = amount
                elif 'sodium' in nutrient_name_lower:
                    nutrition_summary['sodium_mg'] = amount
                elif 'cholesterol' in nutrient_name_lower:
                    nutrition_summary['cholesterol_mg'] = amount
                elif 'vitamin a' in nutrient_name_lower:
                    nutrition_summary['vitamin_a_iu'] = amount
                elif 'vitamin c' in nutrient_name_lower:
                    nutrition_summary['vitamin_c_mg'] = amount
                elif 'vitamin d' in nutrient_name_lower:
                    nutrition_summary['vitamin_d_iu'] = amount
                elif 'calcium' in nutrient_name_lower:
                    nutrition_summary['calcium_mg'] = amount
                elif 'iron' in nutrient_name_lower:
                    nutrition_summary['iron_mg'] = amount
            
            logger.info(f"Found {len(nutrients)} nutrients for food {food_data['id']}")
            
        except Exception as e:
            logger.error(f"Error getting nutrition data for food {food_data['id']}: {e}")
            # If nutrition data can't be retrieved, set empty nutrients array
            nutrients = []
            nutrition_summary = {
                "calories": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
                "sugar_g": 0,
                "sodium_mg": 0,
                "cholesterol_mg": 0,
                "vitamin_a_iu": 0,
                "vitamin_c_mg": 0,
                "vitamin_d_iu": 0,
                "calcium_mg": 0,
                "iron_mg": 0
            }
        
        # Create structured food object with comprehensive nutrition data
        food_object = {
            "id": food_data['id'],
            "name": food_data['name'],
            "serving_size": food_data['serving_size'],
            "serving_unit": food_data['serving_unit'],
            "serving": food_data['serving'],
            "created_at": food_data['created_at'],
            "brand": {"id": food_data['brand_id'], "name": None} if food_data['brand_id'] else None,
            "category": {"id": food_data['category_id'], "name": None} if food_data['category_id'] else None,
            "nutrients": nutrients,  # Comprehensive nutrient array
            "nutrition_summary": nutrition_summary,  # Backward compatibility
            "total_nutrients": len(nutrients)
        }
        
        result.append(food_object)
    
    logger.info(f"Returning {len(result)} structured food objects with comprehensive nutrition data")
    return result


def get_food_nutrition(db, food_id: Any):
    """Get comprehensive food nutrition data using the normalized schema."""
    # First get basic food info
    food_row = db.execute(
        text("SELECT id, name, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
        {"food_id": food_id},
    ).fetchone()
    
    if not food_row:
        raise HTTPException(status_code=404, detail="Food not found")
    
    food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'serving_size', 'serving_unit', 'serving'], food_row))
    
    # Get comprehensive nutrition data from food_nutrients table
    nutrition_rows = db.execute(
        text("""
            SELECT 
                n.id as nutrient_id,
                n.name as nutrient_name, 
                fn.amount, 
                n.unit as nutrient_unit,
                n.category as nutrient_category
            FROM food_nutrients fn
            JOIN nutrients n ON fn.nutrient_id = n.id
            WHERE fn.food_id = :food_id
            ORDER BY n.name
        """),
        {"food_id": food_id}
    ).fetchall()
    
    # Convert nutrition data to comprehensive format
    nutrients = []
    nutrition_data = {}
    
    for row in nutrition_rows:
        if hasattr(row, '_mapping'):
            nutrient = dict(row._mapping)
        else:
            nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit', 'nutrient_category'], row))
        
        nutrient_name = nutrient['nutrient_name'].lower()
        amount = nutrient['amount'] or 0
        
        # Add to comprehensive nutrients array
        nutrients.append({
            "id": nutrient['nutrient_id'],
            "name": nutrient['nutrient_name'],
            "unit": nutrient['nutrient_unit'],
            "amount": amount,
            "category": nutrient.get('nutrient_category', 'general')
        })
        
        # Also populate summary for backward compatibility
        if 'calorie' in nutrient_name or 'energy' in nutrient_name:
            nutrition_data['calories'] = amount
        elif 'protein' in nutrient_name:
            nutrition_data['protein_g'] = amount
        elif 'carbohydrate' in nutrient_name or 'carb' in nutrient_name:
            nutrition_data['carbs_g'] = amount
        elif 'fat' in nutrient_name and 'total' in nutrient_name:
            nutrition_data['fat_g'] = amount
        elif 'fat' in nutrient_name:
            nutrition_data['fat_g'] = amount
        elif 'fiber' in nutrient_name:
            nutrition_data['fiber_g'] = amount
        elif 'sugar' in nutrient_name:
            nutrition_data['sugar_g'] = amount
        elif 'sodium' in nutrient_name:
            nutrition_data['sodium_mg'] = amount
        elif 'cholesterol' in nutrient_name:
            nutrition_data['cholesterol_mg'] = amount
        elif 'vitamin a' in nutrient_name:
            nutrition_data['vitamin_a_iu'] = amount
        elif 'vitamin c' in nutrient_name:
            nutrition_data['vitamin_c_mg'] = amount
        elif 'vitamin d' in nutrient_name:
            nutrition_data['vitamin_d_iu'] = amount
        elif 'calcium' in nutrient_name:
            nutrition_data['calcium_mg'] = amount
        elif 'iron' in nutrient_name:
            nutrition_data['iron_mg'] = amount
    
    # Set default values if not found
    nutrition_data.setdefault('calories', 0)
    nutrition_data.setdefault('protein_g', 0)
    nutrition_data.setdefault('carbs_g', 0)
    nutrition_data.setdefault('fat_g', 0)
    nutrition_data.setdefault('fiber_g', 0)
    nutrition_data.setdefault('sugar_g', 0)
    nutrition_data.setdefault('sodium_mg', 0)
    nutrition_data.setdefault('cholesterol_mg', 0)
    nutrition_data.setdefault('vitamin_a_iu', 0)
    nutrition_data.setdefault('vitamin_c_mg', 0)
    nutrition_data.setdefault('vitamin_d_iu', 0)
    nutrition_data.setdefault('calcium_mg', 0)
    nutrition_data.setdefault('iron_mg', 0)
    
    # Combine food details with comprehensive nutrition data
    food_details.update(nutrition_data)
    food_details['nutrients'] = nutrients  # Add comprehensive nutrients array
    food_details['total_nutrients'] = len(nutrients)
    
    return food_details


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
                # Extract the original integer food ID from the UUID if possible
                # The UUID was generated using uuid5 with format "nutrition_food_{original_id}"
                original_food_id = None
                try:
                    # Try to extract the original ID from the UUID
                    if hasattr(entry.food_item_id, 'hex'):
                        # If it's a UUID object, convert to string
                        food_id_str = str(entry.food_item_id)
                    else:
                        food_id_str = str(entry.food_item_id)
                    
                    # For now, we'll skip the nutrition database lookup since we already have the nutrition data
                    # This avoids the ID format mismatch issue
                    logger.info(f"Using nutrition data from entry for food_id {entry.food_item_id}")
                    food_details = {
                        "name": "Food from nutrition database",
                        "serving_size": 100,
                        "serving_unit": "g",
                        "serving": "100g"
                    }
                except Exception as e:
                    logger.warning(f"Could not extract original food ID from UUID: {e}")
                    # Continue without serving details
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
    logger.info(f"Fuzzy searching for food with name: '{name}'")

    # First try exact and partial matches to get immediate results
    exact_query = """
    SELECT DISTINCT
        f.id, f.name, f.brand_id, f.category_id, f.serving_size, f.serving_unit, f.serving, f.created_at
    FROM foods f
    WHERE f.name IS NOT NULL AND f.name != ''
    AND (LOWER(f.name) = LOWER(:name) OR LOWER(f.name) LIKE LOWER(:name_pattern))
    ORDER BY f.name
    LIMIT 50
    """
    
    exact_rows = db.execute(text(exact_query), {
        "name": name,
        "name_pattern": f"%{name}%"
    }).fetchall()
    
    logger.info(f"Found {len(exact_rows)} exact/partial matches for '{name}'")
    
    # If we found exact matches, use those
    if exact_rows:
        rows = exact_rows
    else:
        # Fallback to broader search - get more foods for fuzzy matching
        broader_query = """
        SELECT DISTINCT
            f.id, f.name, f.brand_id, f.category_id, f.serving_size, f.serving_unit, f.serving, f.created_at
        FROM foods f
        WHERE f.name IS NOT NULL AND f.name != ''
        ORDER BY f.name
        LIMIT 500
        """
        
        rows = db.execute(text(broader_query)).fetchall()
        logger.info(f"Retrieved {len(rows)} foods for broader fuzzy matching")
    
    # Convert rows to list of foods with comprehensive nutrition data
    foods = []
    for row in rows:
        if hasattr(row, '_mapping'):
            food_data = dict(row._mapping)
        else:
            columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
            food_data = dict(zip(columns, row))
        
        # Get comprehensive nutrition data for this food from food_nutrients table
        try:
            nutrition_rows = db.execute(
                text("""
                    SELECT 
                        n.id as nutrient_id, 
                        n.name as nutrient_name, 
                        fn.amount, 
                        n.unit as nutrient_unit,
                        n.category as nutrient_category
                    FROM food_nutrients fn
                    JOIN nutrients n ON fn.nutrient_id = n.id
                    WHERE fn.food_id = :food_id
                    ORDER BY n.name
                """),
                {"food_id": food_data['id']}
            ).fetchall()
            
            # Convert nutrition data to comprehensive nutrients array
            nutrients = []
            nutrition_summary = {
                "calories": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
                "sugar_g": 0,
                "sodium_mg": 0,
                "cholesterol_mg": 0,
                "vitamin_a_iu": 0,
                "vitamin_c_mg": 0,
                "vitamin_d_iu": 0,
                "calcium_mg": 0,
                "iron_mg": 0
            }
            
            for nutrition_row in nutrition_rows:
                if hasattr(nutrition_row, '_mapping'):
                    nutrient = dict(nutrition_row._mapping)
                else:
                    nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit', 'nutrient_category'], nutrition_row))
                
                amount = nutrient['amount'] or 0
                nutrient_name_lower = nutrient['nutrient_name'].lower()
                
                # Add to comprehensive nutrients array
                nutrients.append({
                    "id": nutrient['nutrient_id'],
                    "name": nutrient['nutrient_name'],
                    "unit": nutrient['nutrient_unit'],
                    "amount": amount,
                    "category": nutrient.get('nutrient_category', 'general')
                })
                
                # Also populate summary for backward compatibility
                if 'calorie' in nutrient_name_lower or 'energy' in nutrient_name_lower:
                    nutrition_summary['calories'] = amount
                elif 'protein' in nutrient_name_lower:
                    nutrition_summary['protein_g'] = amount
                elif 'carbohydrate' in nutrient_name_lower or 'carb' in nutrient_name_lower:
                    nutrition_summary['carbs_g'] = amount
                elif 'fat' in nutrient_name_lower and 'total' in nutrient_name_lower:
                    nutrition_summary['fat_g'] = amount
                elif 'fat' in nutrient_name_lower:
                    nutrition_summary['fat_g'] = amount
                elif 'fiber' in nutrient_name_lower:
                    nutrition_summary['fiber_g'] = amount
                elif 'sugar' in nutrient_name_lower:
                    nutrition_summary['sugar_g'] = amount
                elif 'sodium' in nutrient_name_lower:
                    nutrition_summary['sodium_mg'] = amount
                elif 'cholesterol' in nutrient_name_lower:
                    nutrition_summary['cholesterol_mg'] = amount
                elif 'vitamin a' in nutrient_name_lower:
                    nutrition_summary['vitamin_a_iu'] = amount
                elif 'vitamin c' in nutrient_name_lower:
                    nutrition_summary['vitamin_c_mg'] = amount
                elif 'vitamin d' in nutrient_name_lower:
                    nutrition_summary['vitamin_d_iu'] = amount
                elif 'calcium' in nutrient_name_lower:
                    nutrition_summary['calcium_mg'] = amount
                elif 'iron' in nutrient_name_lower:
                    nutrition_summary['iron_mg'] = amount
            
        except Exception as e:
            logger.warning(f"Error getting nutrition data for food {food_data['id']}: {e}")
            nutrients = []
            nutrition_summary = {
                "calories": 0,
                "protein_g": 0,
                "carbs_g": 0,
                "fat_g": 0,
                "fiber_g": 0,
                "sugar_g": 0,
                "sodium_mg": 0,
                "cholesterol_mg": 0,
                "vitamin_a_iu": 0,
                "vitamin_c_mg": 0,
                "vitamin_d_iu": 0,
                "calcium_mg": 0,
                "iron_mg": 0
            }
        
        foods.append({
            "id": food_data['id'],
            "name": food_data['name'],
            "brand_id": food_data['brand_id'],
            "category_id": food_data['category_id'],
            "serving_size": food_data['serving_size'],
            "serving_unit": food_data['serving_unit'],
            "serving": food_data['serving'],
            "created_at": food_data['created_at'],
            "nutrients": nutrients,  # Comprehensive nutrient array
            "nutrition_summary": nutrition_summary,  # Backward compatibility
            "total_nutrients": len(nutrients)
        })
    
    logger.info(f"Processing {len(foods)} foods for fuzzy matching")
    
    # Use difflib for fuzzy matching
    matches = []
    for food in foods:
        if not food["name"]:
            continue
        
        # Check for exact match first (highest priority)
        if food["name"].lower() == name.lower():
            similarity = 1.0
        # Check for contains match (high priority)
        elif name.lower() in food["name"].lower():
            similarity = 0.8
        # Check for starts with match (medium priority)
        elif food["name"].lower().startswith(name.lower()):
            similarity = 0.7
        # Check for ends with match (medium priority)
        elif food["name"].lower().endswith(name.lower()):
            similarity = 0.6
        # Use difflib for fuzzy matching (lower priority)
        else:
            similarity = difflib.SequenceMatcher(
                None, name.lower(), food["name"].lower()
            ).ratio()
        
        logger.debug(f"Food '{food['name']}' similarity: {similarity}")
        
        # Lower threshold for more results, but prioritize exact/partial matches
        if similarity > 0.1:  # Lower threshold for more results
            food["similarity"] = similarity
            matches.append(food)

    logger.info(f"Found {len(matches)} foods with similarity > 0.1")

    # Sort by similarity and return top 40
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Convert to structured format like search_food_by_name
    structured_matches = []
    for food in matches[:40]:
        structured_food = {
            "id": food["id"],
            "name": food["name"],
            "serving_size": food["serving_size"],
            "serving_unit": food["serving_unit"],
            "serving": food["serving"],
            "created_at": food["created_at"],
            "brand": {"id": food["brand_id"], "name": None} if food["brand_id"] else None,
            "category": {"id": food["category_id"], "name": None} if food["category_id"] else None,
            "nutrients": food["nutrients"],  # Comprehensive nutrient array
            "nutrition_summary": food["nutrition_summary"],  # Backward compatibility
            "total_nutrients": food["total_nutrients"],
            "similarity": food["similarity"]
        }
        structured_matches.append(structured_food)
    logger.info(f"Returning {len(structured_matches)} structured fuzzy matches with comprehensive nutrition data")
    return structured_matches


@app.post("/meal-plan")
def create_meal_plan(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Create a personalized meal plan using AI, then query nutrition database for accurate macros."""
    try:
        user_id = request.get("user_id")
        daily_calories = request.get("daily_calories", 2000)
        meal_count = request.get("meal_count", 3)
        dietary_restrictions = request.get("dietary_restrictions", [])

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        # Get user's food preferences and history from shared database
        user_history = get_user_calorie_history(db_shared, user_id)

        # AI-powered meal plan creation
        meal_plan = create_ai_meal_plan(
            user_id, daily_calories, meal_count, dietary_restrictions, user_history, db_nutrition
        )

        # Calculate total calories from enriched meal plan
        total_calories = sum(meal["total_calories"] for meal in meal_plan)

        return {
            "status": "success",
            "meal_plan": meal_plan,
            "total_calories": total_calories,
            "meals": len(meal_plan),
            "ai_generated": True,
            "nutrition_verified": True
        }
    except Exception as e:
        logger.error(f"Error creating AI meal plan: {e}")
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
            # Ensure the food has nutrition data, use defaults if missing
            calories = selected_food.get("calories", 0)
            meal_plan.append(
                {
                    "meal_type": meal_type,
                    "food": selected_food,
                    "calories": calories,
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
        
        # Convert rows to dictionaries properly and add nutrition data
        result = []
        for row in rows:
            if hasattr(row, '_mapping'):
                food_data = dict(row._mapping)
            else:
                # Handle tuple case
                columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
                food_data = dict(zip(columns, row))
            
            # Get nutrition data for this food
            try:
                nutrition_rows = db.execute(
                    text("""
                        SELECT n.name as nutrient_name, fn.amount, n.unit
                        FROM food_nutrients fn
                        JOIN nutrients n ON fn.nutrient_id = n.id
                        WHERE fn.food_id = :food_id
                    """),
                    {"food_id": food_data['id']}
                ).fetchall()
                
                # Convert nutrition data to a dictionary
                nutrition_data = {}
                for nutrition_row in nutrition_rows:
                    if hasattr(nutrition_row, '_mapping'):
                        nutrient = dict(nutrition_row._mapping)
                    else:
                        nutrient = dict(zip(['nutrient_name', 'amount', 'unit'], nutrition_row))
                    
                    nutrient_name = nutrient['nutrient_name'].lower()
                    amount = nutrient['amount'] or 0
                    
                    # Map nutrient names to our expected fields
                    if 'calorie' in nutrient_name or 'energy' in nutrient_name:
                        nutrition_data['calories'] = amount
                    elif 'protein' in nutrient_name:
                        nutrition_data['protein_g'] = amount
                    elif 'carbohydrate' in nutrient_name or 'carb' in nutrient_name:
                        nutrition_data['carbs_g'] = amount
                    elif 'fat' in nutrient_name and 'total' in nutrient_name:
                        nutrition_data['fat_g'] = amount
                    elif 'fat' in nutrient_name:
                        nutrition_data['fat_g'] = amount
                
                # Set default values if not found
                nutrition_data.setdefault('calories', 0)
                nutrition_data.setdefault('protein_g', 0)
                nutrition_data.setdefault('carbs_g', 0)
                nutrition_data.setdefault('fat_g', 0)
                
                # Add nutrition data to food_data
                food_data.update(nutrition_data)
                
            except Exception as e:
                # If nutrition data can't be retrieved, set defaults
                food_data.update({
                    'calories': 0,
                    'protein_g': 0,
                    'carbs_g': 0,
                    'fat_g': 0
                })
            
            result.append(food_data)
        
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
async def fuzzy_search_food(request: Request, db=Depends(get_nutrition_db)):
    """Search for foods using fuzzy matching for better results."""
    try:
        # Try to parse JSON body
        try:
            data = await request.json()
        except Exception as e:
            # Try to parse form data as fallback
            try:
                form = await request.form()
                data = dict(form)
            except Exception as e2:
                raw_body = await request.body()
                logger.error(f"/fuzzy-search: Could not parse request body as JSON or form-data. Raw body: {raw_body}")
                raise HTTPException(status_code=400, detail="Request body must be a valid JSON object (e.g., { 'query': 'apple' }) or form-data.")
        
        query = data.get("query", "")
        limit = int(data.get("limit", 20))

        if not query:
            raise HTTPException(status_code=400, detail="Search query required")

        results = search_food_fuzzy(db, query)
        limited_results = results[:limit]

        return {
            "status": "success",
            "results": limited_results,
            "count": len(limited_results),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/fuzzy-search error: {e}")
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
    db_nutrition, db_shared, user_id: str, meal_type: str, target_calories: Optional[float] = None
):
    """Get AI-powered meal suggestions based on user preferences and calorie targets."""
    try:
        settings = get_settings()
        
        # Initialize Groq client
        client = openai.OpenAI(
            api_key=settings.llm.groq_api_key,
            base_url=settings.llm.groq_base_url
        )
        
        # Get user's recent food preferences for context (from shared database)
        recent_foods = db_shared.execute(
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

        # Get popular foods for this meal type (from shared database)
        popular_foods = db_shared.execute(
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

        # Convert rows to dictionaries
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

        # Get food details for context (from nutrition database)
        food_ids = list(set([f["food_item_id"] for f in recent_foods_dict + popular_foods_dict]))
        user_preferences = []
        popular_choices = []
        
        if food_ids:
            placeholders = ",".join([":id" + str(i) for i in range(len(food_ids))])
            params = {f"id{i}": food_id for i, food_id in enumerate(food_ids)}
            
            foods = db_nutrition.execute(
                text(
                    f"""
                SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving
                FROM foods 
                WHERE id IN ({placeholders})
                """
                ),
                params,
            ).fetchall()

            for food in foods:
                if hasattr(food, '_mapping'):
                    food_dict = dict(food._mapping)
                else:
                    columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving']
                    food_dict = dict(zip(columns, food))
                
                # Check if this food is in user's recent preferences
                if any(f["food_item_id"] == food_dict["id"] for f in recent_foods_dict):
                    user_preferences.append(food_dict["name"])
                else:
                    popular_choices.append(food_dict["name"])

        # If no preferences found, get some general foods for this meal type
        if not user_preferences and not popular_choices:
            general_foods = _get_general_foods_for_meal_type(db, meal_type)
            popular_choices = general_foods

        # Prepare context for AI
        context = f"""
        User Profile:
        - Meal type: {meal_type}
        - Target calories: {target_calories if target_calories else 'Not specified'}
        - User's recent food preferences: {', '.join(user_preferences[:5]) if user_preferences else 'None available'}
        - Popular foods for {meal_type}: {', '.join(popular_choices[:5]) if popular_choices else 'None available'}
        
        Requirements:
        - Suggest 5-8 food items suitable for {meal_type}
        - Consider user's preferences and popular choices if available
        - If no preferences available, suggest common nutritious foods for {meal_type}
        - Ensure variety and nutritional balance
        - Focus on whole, nutritious foods
        - If target calories specified, suggest appropriate portion sizes
        """

        # Generate AI suggestions
        ai_suggestions = _generate_ai_meal_suggestions(client, context, target_calories)
        
        # Enrich with nutrition database data
        enriched_suggestions = _enrich_suggestions_with_nutrition(ai_suggestions, db_nutrition)
        
        return {
            "meal_type": meal_type,
            "target_calories": target_calories,
            "suggestions": enriched_suggestions,
            "ai_generated": True,
            "user_preferences_considered": len(user_preferences) > 0,
            "fallback_used": not user_preferences and not popular_choices
        }
        
    except Exception as e:
        logger.error(f"Error in AI meal suggestions: {e}")
        # Fallback to rule-based suggestions
        return _get_fallback_meal_suggestions(db_nutrition, db_shared, user_id, meal_type, target_calories)


def _generate_ai_meal_suggestions(client, context: str, target_calories: Optional[float] = None) -> List[Dict]:
    """Generate meal suggestions using Groq AI."""
    
    settings = get_settings()
    
    prompt = f"""
    {context}
    
    Create 5-8 food suggestions for this meal. For each food, provide:
    1. Food name (specific food names that would be found in a nutrition database)
    2. Suggested quantity in grams
    3. Brief nutritional reasoning
    
    Format your response as a JSON array with this structure:
    [
        {{
            "name": "oatmeal",
            "quantity_g": 50,
            "reasoning": "High fiber, complex carbs for sustained energy"
        }},
        {{
            "name": "banana",
            "quantity_g": 120,
            "reasoning": "Natural sweetness and potassium"
        }}
    ]
    
    Guidelines:
    - Use common food names that would be found in a nutrition database
    - Suggest realistic portion sizes
    - Consider the meal type and calorie target
    - Provide diverse, nutritious options
    - Keep reasoning brief but informative
    - If no user preferences are available, suggest common nutritious foods for the meal type
    - Focus on whole foods and balanced nutrition
    """
    
    try:
        response = client.chat.completions.create(
            model=settings.llm.groq_model,
            messages=[
                {"role": "system", "content": "You are a nutrition expert providing personalized meal suggestions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        return _parse_ai_suggestions(ai_response)
        
    except Exception as e:
        logger.error(f"Error generating AI suggestions: {e}")
        return []


def _parse_ai_suggestions(ai_response: str) -> List[Dict]:
    """Parse AI response into structured suggestions."""
    try:
        # Extract JSON from response
        import json
        import re
        
        # Find JSON array in the response
        json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())
            return suggestions
        else:
            logger.warning("Could not parse AI response as JSON")
            return []
            
    except Exception as e:
        logger.error(f"Error parsing AI suggestions: {e}")
        return []


def _enrich_suggestions_with_nutrition(ai_suggestions: List[Dict], db) -> List[Dict]:
    """Enrich AI suggestions with nutrition database data."""
    enriched_suggestions = []
    
    for suggestion in ai_suggestions:
        food_name = suggestion.get("name", "")
        if not food_name:
            continue
            
        # Search for the food in the database
        food_data = _get_food_nutrition_from_db(db, food_name)
        
        if food_data:
            # Use database data but keep AI reasoning
            enriched_suggestion = {
                "id": food_data.get("id"),
                "name": food_data.get("name", food_name),
                "brand_id": food_data.get("brand_id"),
                "category_id": food_data.get("category_id"),
                "serving_size": food_data.get("serving_size"),
                "serving_unit": food_data.get("serving_unit"),
                "serving": food_data.get("serving"),
                "suggested_quantity_g": suggestion.get("quantity_g", 100),
                "ai_reasoning": suggestion.get("reasoning", ""),
                "nutrition_verified": True
            }
        else:
            # Keep AI suggestion but mark as not verified
            enriched_suggestion = {
                "name": food_name,
                "suggested_quantity_g": suggestion.get("quantity_g", 100),
                "ai_reasoning": suggestion.get("reasoning", ""),
                "nutrition_verified": False
            }
        
        enriched_suggestions.append(enriched_suggestion)
    
    return enriched_suggestions[:10]  # Limit to 10 suggestions


def _get_fallback_meal_suggestions(db_nutrition, db_shared, user_id: str, meal_type: str, target_calories: Optional[float] = None) -> Dict:
    """Fallback rule-based meal suggestions when AI fails."""
    # Get user's recent food preferences (from shared database)
    recent_foods = db_shared.execute(
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

    # Get popular foods for this meal type (from shared database)
    popular_foods = db_shared.execute(
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
        return {"suggestions": [], "message": "No food preferences found", "ai_generated": False}

    # Get food details (from nutrition database)
    placeholders = ",".join([":id" + str(i) for i in range(len(food_ids))])
    params = {f"id{i}": food_id for i, food_id in enumerate(food_ids)}

    foods = db_nutrition.execute(
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
        "ai_generated": False,
        "fallback_used": True
    }


@app.post("/meal-plan-rule-based")
def create_rule_based_meal_plan(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Create a meal plan using the old rule-based algorithm (fallback option)."""
    try:
        user_id = request.get("user_id")
        daily_calories = request.get("daily_calories", 2000)
        meal_count = request.get("meal_count", 3)
        dietary_restrictions = request.get("dietary_restrictions", [])

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        # Get user's food preferences and history from shared database
        user_history = get_user_calorie_history(db_shared, user_id)

        # Rule-based meal planning algorithm
        meal_plan = generate_meal_plan(
            db_nutrition, daily_calories, meal_count, dietary_restrictions, user_history
        )

        return {
            "status": "success",
            "meal_plan": meal_plan,
            "total_calories": sum(meal["calories"] for meal in meal_plan),
            "meals": len(meal_plan),
            "ai_generated": False,
            "method": "rule_based"
        }
    except Exception as e:
        logger.error(f"Error creating rule-based meal plan: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create meal plan: {e}")


@app.post("/test-ai-meal-plan")
def test_ai_meal_plan(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Test endpoint for AI meal plan creation with sample data."""
    try:
        # Use sample data for testing
        user_id = request.get("user_id", "test-user-123")
        daily_calories = request.get("daily_calories", 2000)
        meal_count = request.get("meal_count", 3)
        dietary_restrictions = request.get("dietary_restrictions", ["vegetarian"])
        
        # Sample user history
        sample_history = [
            {"name": "oatmeal", "calories": 150, "protein_g": 5, "carbs_g": 27, "fat_g": 3},
            {"name": "banana", "calories": 105, "protein_g": 1, "carbs_g": 27, "fat_g": 0},
            {"name": "chicken breast", "calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6},
            {"name": "brown rice", "calories": 110, "protein_g": 2.5, "carbs_g": 23, "fat_g": 0.9},
            {"name": "salmon", "calories": 208, "protein_g": 25, "carbs_g": 0, "fat_g": 12}
        ]

        # AI-powered meal plan creation
        meal_plan = create_ai_meal_plan(
            user_id, daily_calories, meal_count, dietary_restrictions, sample_history, db_nutrition
        )

        # Calculate total calories from enriched meal plan
        total_calories = sum(meal["total_calories"] for meal in meal_plan)

        return {
            "status": "success",
            "meal_plan": meal_plan,
            "total_calories": total_calories,
            "meals": len(meal_plan),
            "ai_generated": True,
            "nutrition_verified": True,
            "test_mode": True,
            "sample_data_used": True
        }
    except Exception as e:
        logger.error(f"Error testing AI meal plan: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to test AI meal plan: {e}")

@app.get("/debug/foods")
def debug_foods(db=Depends(get_nutrition_db)):
    """Debug endpoint to check food data in the database."""
    try:
        # Get a sample of foods to check their names
        rows = db.execute(
            text("""
                SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving
                FROM foods
                WHERE name IS NOT NULL AND name != ''
                ORDER BY name
                LIMIT 10
            """)
        ).fetchall()
        
        foods = []
        for row in rows:
            if hasattr(row, '_mapping'):
                food_data = dict(row._mapping)
            else:
                columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving']
                food_data = dict(zip(columns, row))
            
            foods.append(food_data)
        
        return {
            "status": "success",
            "sample_foods": foods,
            "total_foods": len(foods),
            "message": "Sample of foods from database"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/debug/search-test")
def debug_search_test(query: str = "apple", db=Depends(get_nutrition_db)):
    """Debug endpoint to test search functionality with a specific query."""
    try:
        # Test exact match query
        exact_query = """
        SELECT DISTINCT
            f.id, f.name, f.brand_id, f.category_id, f.serving_size, f.serving_unit, f.serving, f.created_at
        FROM foods f
        WHERE f.name IS NOT NULL AND f.name != ''
        AND (LOWER(f.name) = LOWER(:name) OR LOWER(f.name) LIKE LOWER(:name_pattern))
        ORDER BY f.name
        LIMIT 10
        """
        
        exact_rows = db.execute(text(exact_query), {
            "name": query,
            "name_pattern": f"%{query}%"
        }).fetchall()
        
        # Test broader query
        broader_query = """
        SELECT DISTINCT
            f.id, f.name, f.brand_id, f.category_id, f.serving_size, f.serving_unit, f.serving, f.created_at
        FROM foods f
        WHERE f.name IS NOT NULL AND f.name != ''
        ORDER BY f.name
        LIMIT 20
        """
        
        broader_rows = db.execute(text(broader_query)).fetchall()
        
        # Convert to lists
        exact_matches = []
        for row in exact_rows:
            if hasattr(row, '_mapping'):
                exact_matches.append(dict(row._mapping))
            else:
                columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
                exact_matches.append(dict(zip(columns, row)))
        
        broader_matches = []
        for row in broader_rows:
            if hasattr(row, '_mapping'):
                broader_matches.append(dict(row._mapping))
            else:
                columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
                broader_matches.append(dict(zip(columns, row)))
        
        # Test similarity calculations
        similarity_tests = []
        for food in broader_matches[:5]:  # Test first 5 foods
            food_name = food['name'].lower()
            query_lower = query.lower()
            
            # Calculate different similarity metrics
            exact_match = food_name == query_lower
            contains = query_lower in food_name
            starts_with = food_name.startswith(query_lower)
            ends_with = food_name.endswith(query_lower)
            difflib_similarity = difflib.SequenceMatcher(None, query_lower, food_name).ratio()
            
            similarity_tests.append({
                "food_name": food['name'],
                "exact_match": exact_match,
                "contains": contains,
                "starts_with": starts_with,
                "ends_with": ends_with,
                "difflib_similarity": round(difflib_similarity, 3)
            })
        
        return {
            "status": "success",
            "query": query,
            "exact_matches_count": len(exact_matches),
            "exact_matches": exact_matches,
            "broader_matches_count": len(broader_matches),
            "broader_matches": broader_matches[:5],  # Show first 5
            "similarity_tests": similarity_tests
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/foods/{food_id}/nutrients")
def get_food_nutrients_comprehensive(food_id: int, db=Depends(get_nutrition_db)):
    """Get comprehensive nutrient information for a specific food item."""
    try:
        # Get basic food info
        food_row = db.execute(
            text("SELECT id, name, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
            {"food_id": food_id},
        ).fetchone()
        
        if not food_row:
            raise HTTPException(status_code=404, detail="Food not found")
        
        food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'serving_size', 'serving_unit', 'serving'], food_row))
        
        # Get all nutrients for this food
        nutrition_rows = db.execute(
            text("""
                SELECT 
                    n.id as nutrient_id,
                    n.name as nutrient_name, 
                    fn.amount, 
                    n.unit as nutrient_unit,
                    n.category as nutrient_category
                FROM food_nutrients fn
                JOIN nutrients n ON fn.nutrient_id = n.id
                WHERE fn.food_id = :food_id
                ORDER BY n.category, n.name
            """),
            {"food_id": food_id}
        ).fetchall()
        
        # Group nutrients by category
        nutrients_by_category = {}
        all_nutrients = []
        nutrition_summary = {
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "fiber_g": 0,
            "sugar_g": 0,
            "sodium_mg": 0,
            "cholesterol_mg": 0,
            "vitamin_a_iu": 0,
            "vitamin_c_mg": 0,
            "vitamin_d_iu": 0,
            "calcium_mg": 0,
            "iron_mg": 0
        }
        
        for row in nutrition_rows:
            if hasattr(row, '_mapping'):
                nutrient = dict(row._mapping)
            else:
                nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit', 'nutrient_category'], row))
            
            amount = nutrient['amount'] or 0
            nutrient_name_lower = nutrient['nutrient_name'].lower()
            category = nutrient.get('nutrient_category', 'general')
            
            nutrient_obj = {
                "id": nutrient['nutrient_id'],
                "name": nutrient['nutrient_name'],
                "unit": nutrient['nutrient_unit'],
                "amount": amount,
                "category": category
            }
            
            # Add to all nutrients list
            all_nutrients.append(nutrient_obj)
            
            # Group by category
            if category not in nutrients_by_category:
                nutrients_by_category[category] = []
            nutrients_by_category[category].append(nutrient_obj)
            
            # Populate summary for backward compatibility
            if 'calorie' in nutrient_name_lower or 'energy' in nutrient_name_lower:
                nutrition_summary['calories'] = amount
            elif 'protein' in nutrient_name_lower:
                nutrition_summary['protein_g'] = amount
            elif 'carbohydrate' in nutrient_name_lower or 'carb' in nutrient_name_lower:
                nutrition_summary['carbs_g'] = amount
            elif 'fat' in nutrient_name_lower and 'total' in nutrient_name_lower:
                nutrition_summary['fat_g'] = amount
            elif 'fat' in nutrient_name_lower:
                nutrition_summary['fat_g'] = amount
            elif 'fiber' in nutrient_name_lower:
                nutrition_summary['fiber_g'] = amount
            elif 'sugar' in nutrient_name_lower:
                nutrition_summary['sugar_g'] = amount
            elif 'sodium' in nutrient_name_lower:
                nutrition_summary['sodium_mg'] = amount
            elif 'cholesterol' in nutrient_name_lower:
                nutrition_summary['cholesterol_mg'] = amount
            elif 'vitamin a' in nutrient_name_lower:
                nutrition_summary['vitamin_a_iu'] = amount
            elif 'vitamin c' in nutrient_name_lower:
                nutrition_summary['vitamin_c_mg'] = amount
            elif 'vitamin d' in nutrient_name_lower:
                nutrition_summary['vitamin_d_iu'] = amount
            elif 'calcium' in nutrient_name_lower:
                nutrition_summary['calcium_mg'] = amount
            elif 'iron' in nutrient_name_lower:
                nutrition_summary['iron_mg'] = amount
        
        return {
            "food": {
                "id": food_details['id'],
                "name": food_details['name'],
                "serving_size": food_details['serving_size'],
                "serving_unit": food_details['serving_unit'],
                "serving": food_details['serving']
            },
            "nutrients": all_nutrients,
            "nutrients_by_category": nutrients_by_category,
            "nutrition_summary": nutrition_summary,
            "total_nutrients": len(all_nutrients),
            "categories": list(nutrients_by_category.keys())
        }
        
    except Exception as e:
        logger.error(f"Error getting comprehensive nutrients for food {food_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nutrient information: {str(e)}")

@app.post("/search-foods-comprehensive")
async def search_foods_comprehensive(request: Request, db=Depends(get_nutrition_db)):
    """Search for foods with comprehensive nutrient information."""
    try:
        # Parse request body
        try:
            data = await request.json()
        except Exception as e:
            # Try to parse form data as fallback
            try:
                form = await request.form()
                data = dict(form)
            except Exception as e2:
                raw_body = await request.body()
                logger.error(f"/search-foods-comprehensive: Could not parse request body. Raw body: {raw_body}")
                raise HTTPException(status_code=400, detail="Request body must be a valid JSON object (e.g., { 'query': 'apple', 'limit': 10 })")
        
        query = data.get("query", "")
        limit = int(data.get("limit", 10))
        include_nutrients = data.get("include_nutrients", True)
        
        if not query:
            raise HTTPException(status_code=400, detail="Search query required")
        
        # Search for foods
        search_query = """
        SELECT DISTINCT
            f.id, f.name, f.brand_id, f.category_id, f.serving_size, f.serving_unit, f.serving, f.created_at
        FROM foods f
        WHERE f.name IS NOT NULL AND f.name != ''
        AND LOWER(f.name) LIKE LOWER(:query_pattern)
        ORDER BY f.name
        LIMIT :limit
        """
        
        food_rows = db.execute(text(search_query), {
            "query_pattern": f"%{query.lower()}%",
            "limit": limit
        }).fetchall()
        
        results = []
        for food_row in food_rows:
            if hasattr(food_row, '_mapping'):
                food_data = dict(food_row._mapping)
            else:
                columns = ['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at']
                food_data = dict(zip(columns, food_row))
            
            food_result = {
                "id": food_data['id'],
                "name": food_data['name'],
                "serving_size": food_data['serving_size'],
                "serving_unit": food_data['serving_unit'],
                "serving": food_data['serving'],
                "created_at": food_data['created_at'],
                "brand": {"id": food_data['brand_id'], "name": None} if food_data['brand_id'] else None,
                "category": {"id": food_data['category_id'], "name": None} if food_data['category_id'] else None,
            }
            
            # Get comprehensive nutrients if requested
            if include_nutrients:
                try:
                    nutrition_rows = db.execute(
                        text("""
                            SELECT 
                                n.id as nutrient_id,
                                n.name as nutrient_name, 
                                fn.amount, 
                                n.unit as nutrient_unit,
                                n.category as nutrient_category
                            FROM food_nutrients fn
                            JOIN nutrients n ON fn.nutrient_id = n.id
                            WHERE fn.food_id = :food_id
                            ORDER BY n.category, n.name
                        """),
                        {"food_id": food_data['id']}
                    ).fetchall()
                    
                    nutrients = []
                    nutrition_summary = {
                        "calories": 0,
                        "protein_g": 0,
                        "carbs_g": 0,
                        "fat_g": 0,
                        "fiber_g": 0,
                        "sugar_g": 0,
                        "sodium_mg": 0,
                        "cholesterol_mg": 0,
                        "vitamin_a_iu": 0,
                        "vitamin_c_mg": 0,
                        "vitamin_d_iu": 0,
                        "calcium_mg": 0,
                        "iron_mg": 0
                    }
                    
                    for nutrition_row in nutrition_rows:
                        if hasattr(nutrition_row, '_mapping'):
                            nutrient = dict(nutrition_row._mapping)
                        else:
                            nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit', 'nutrient_category'], nutrition_row))
                        
                        amount = nutrient['amount'] or 0
                        nutrient_name_lower = nutrient['nutrient_name'].lower()
                        
                        nutrients.append({
                            "id": nutrient['nutrient_id'],
                            "name": nutrient['nutrient_name'],
                            "unit": nutrient['nutrient_unit'],
                            "amount": amount,
                            "category": nutrient.get('nutrient_category', 'general')
                        })
                        
                        # Populate summary
                        if 'calorie' in nutrient_name_lower or 'energy' in nutrient_name_lower:
                            nutrition_summary['calories'] = amount
                        elif 'protein' in nutrient_name_lower:
                            nutrition_summary['protein_g'] = amount
                        elif 'carbohydrate' in nutrient_name_lower or 'carb' in nutrient_name_lower:
                            nutrition_summary['carbs_g'] = amount
                        elif 'fat' in nutrient_name_lower and 'total' in nutrient_name_lower:
                            nutrition_summary['fat_g'] = amount
                        elif 'fat' in nutrient_name_lower:
                            nutrition_summary['fat_g'] = amount
                        elif 'fiber' in nutrient_name_lower:
                            nutrition_summary['fiber_g'] = amount
                        elif 'sugar' in nutrient_name_lower:
                            nutrition_summary['sugar_g'] = amount
                        elif 'sodium' in nutrient_name_lower:
                            nutrition_summary['sodium_mg'] = amount
                        elif 'cholesterol' in nutrient_name_lower:
                            nutrition_summary['cholesterol_mg'] = amount
                        elif 'vitamin a' in nutrient_name_lower:
                            nutrition_summary['vitamin_a_iu'] = amount
                        elif 'vitamin c' in nutrient_name_lower:
                            nutrition_summary['vitamin_c_mg'] = amount
                        elif 'vitamin d' in nutrient_name_lower:
                            nutrition_summary['vitamin_d_iu'] = amount
                        elif 'calcium' in nutrient_name_lower:
                            nutrition_summary['calcium_mg'] = amount
                        elif 'iron' in nutrient_name_lower:
                            nutrition_summary['iron_mg'] = amount
                    
                    food_result.update({
                        "nutrients": nutrients,
                        "nutrition_summary": nutrition_summary,
                        "total_nutrients": len(nutrients)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error getting nutrients for food {food_data['id']}: {e}")
                    food_result.update({
                        "nutrients": [],
                        "nutrition_summary": {
                            "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0,
                            "fiber_g": 0, "sugar_g": 0, "sodium_mg": 0, "cholesterol_mg": 0,
                            "vitamin_a_iu": 0, "vitamin_c_mg": 0, "vitamin_d_iu": 0,
                            "calcium_mg": 0, "iron_mg": 0
                        },
                        "total_nutrients": 0
                    })
            
            results.append(food_result)
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results),
            "include_nutrients": include_nutrients
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/search-foods-comprehensive error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

# Import models when needed
def get_models():
    from shared.models import FoodItem, FoodLogEntry, NutritionInfo
    return FoodItem, FoodLogEntry, NutritionInfo


def _get_food_nutrition_from_db(db, food_name: str) -> Optional[Dict]:
    """Get nutrition data for a food from the nutrition database."""
    try:
        # Search for food by name (fuzzy search)
        query = """
        SELECT f.id, f.name, f.brand_id, f.category_id, f.serving_size, f.serving_unit, f.serving
        FROM foods f
        WHERE LOWER(f.name) LIKE LOWER(:food_name)
        OR LOWER(f.name) LIKE LOWER(:food_name_pattern)
        LIMIT 1
        """
        
        # Try exact match first, then partial match
        rows = db.execute(
            text(query),
            {"food_name": food_name, "food_name_pattern": f"%{food_name}%"}
        ).fetchall()
        
        if not rows:
            return None
        
        food_row = rows[0]
        food_data = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
        
        return food_data
        
    except Exception as e:
        logger.error(f"Error getting nutrition data for {food_name}: {e}")
        return None


def create_ai_meal_plan(
    user_id: str,
    daily_calories: int,
    meal_count: int,
    dietary_restrictions: List[str],
    user_history: List[Dict],
    db_nutrition
) -> List[Dict]:
    """
    AI-powered meal plan creation using Groq API.
    AI creates the meal plan first, then queries nutrition database for accurate macros.
    """
    try:
        settings = get_settings()
        
        # Initialize Groq client
        client = openai.OpenAI(
            api_key=settings.llm.groq_api_key,
            base_url=settings.llm.groq_base_url
        )
        
        # Prepare user context for AI
        user_context = _prepare_user_context(user_history, dietary_restrictions, daily_calories)
        
        # Generate meal plan with AI
        ai_meal_plan = _generate_meal_plan_with_ai(client, user_context, meal_count)
        
        # Query nutrition database for accurate macros and calories
        enriched_meal_plan = _enrich_meal_plan_with_nutrition(ai_meal_plan, db_nutrition)
        
        return enriched_meal_plan
        
    except Exception as e:
        logger.error(f"Error in AI meal plan creation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create AI meal plan: {str(e)}")


def _prepare_user_context(user_history: List[Dict], dietary_restrictions: List[str], daily_calories: int) -> str:
    """Prepare user context for AI meal plan generation."""
    
    # Extract user's food preferences
    food_preferences = []
    for entry in user_history[:20]:  # Last 20 entries
        food_name = entry.get("name", "")
        if food_name:
            food_preferences.append(food_name.lower())
    
    # Get unique preferences
    unique_preferences = list(set(food_preferences))
    
    context = f"""
    User Profile:
    - Daily calorie target: {daily_calories} calories
    - Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
    - Food preferences (from history): {', '.join(unique_preferences[:10])}
    
    Requirements:
    - Create a balanced meal plan with proper macronutrient distribution
    - Consider user's food preferences and dietary restrictions
    - Ensure variety and nutritional balance
    - Focus on whole, nutritious foods
    """
    
    return context


def _generate_meal_plan_with_ai(client, user_context: str, meal_count: int) -> List[Dict]:
    """Generate meal plan using Groq AI."""
    
    settings = get_settings()
    
    prompt = f"""
    {user_context}
    
    Create a {meal_count}-meal daily plan. For each meal, provide:
    1. Meal type (breakfast/lunch/dinner/snack)
    2. Food items (specific food names that would be found in a nutrition database)
    3. Suggested quantities in grams
    4. Brief nutritional reasoning
    
    Format your response as a JSON array with this structure:
    [
        {{
            "meal_type": "breakfast",
            "foods": [
                {{
                    "name": "oatmeal",
                    "quantity_g": 50,
                    "reasoning": "High fiber, complex carbs for sustained energy"
                }},
                {{
                    "name": "banana",
                    "quantity_g": 120,
                    "reasoning": "Natural sweetness and potassium"
                }}
            ]
        }}
    ]
    
    Focus on common, recognizable food names that would be in a nutrition database.
    """
    
    try:
        response = client.chat.completions.create(
            model=settings.llm.groq_model,
            messages=[
                {"role": "system", "content": "You are a nutrition expert creating personalized meal plans. Provide specific, actionable meal suggestions with common food names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        meal_plan = _parse_ai_meal_plan(ai_response)
        
        return meal_plan
        
    except Exception as e:
        logger.error(f"Error generating meal plan with AI: {e}")
        # Fallback to basic meal plan
        return _create_fallback_meal_plan(meal_count)


def _parse_ai_meal_plan(ai_response: str) -> List[Dict]:
    """Parse AI response into structured meal plan."""
    try:
        # Extract JSON from AI response
        import json
        import re
        
        # Find JSON array in the response
        json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
        if json_match:
            meal_plan = json.loads(json_match.group())
            return meal_plan
        else:
            # If no JSON found, create basic structure
            logger.warning("Could not parse AI response as JSON, using fallback")
            return _create_fallback_meal_plan(3)
            
    except Exception as e:
        logger.error(f"Error parsing AI meal plan: {e}")
        return _create_fallback_meal_plan(3)


def _create_fallback_meal_plan(meal_count: int) -> List[Dict]:
    """Create a basic fallback meal plan."""
    meal_types = ["breakfast", "lunch", "dinner"]
    fallback_plan = []
    
    for i in range(meal_count):
        meal_type = meal_types[i % len(meal_types)]
        fallback_plan.append({
            "meal_type": meal_type,
            "foods": [
                {
                    "name": "chicken breast" if meal_type == "lunch" else "oatmeal" if meal_type == "breakfast" else "salmon",
                    "quantity_g": 150 if meal_type == "lunch" else 50 if meal_type == "breakfast" else 120,
                    "reasoning": "Balanced protein source"
                }
            ]
        })
    
    return fallback_plan


def _enrich_meal_plan_with_nutrition(ai_meal_plan: List[Dict], db_nutrition) -> List[Dict]:
    """Query nutrition database to get accurate macros and calories for AI-generated meals."""
    enriched_plan = []
    
    for meal in ai_meal_plan:
        enriched_meal = {
            "meal_type": meal["meal_type"],
            "foods": [],
            "total_calories": 0,
            "total_protein_g": 0,
            "total_carbs_g": 0,
            "total_fat_g": 0
        }
        
        for food_item in meal["foods"]:
            food_name = food_item["name"]
            quantity_g = food_item["quantity_g"]
            
            # Search for food in nutrition database
            nutrition_data = _get_food_nutrition_from_db(db_nutrition, food_name)
            
            if nutrition_data:
                # Calculate nutrition based on quantity
                actual_calories = (nutrition_data.get("calories", 0) * quantity_g) / 100
                actual_protein = (nutrition_data.get("protein_g", 0) * quantity_g) / 100
                actual_carbs = (nutrition_data.get("carbs_g", 0) * quantity_g) / 100
                actual_fat = (nutrition_data.get("fat_g", 0) * quantity_g) / 100
                
                enriched_food = {
                    "name": food_name,
                    "quantity_g": quantity_g,
                    "reasoning": food_item.get("reasoning", ""),
                    "nutrition": {
                        "calories": round(actual_calories, 1),
                        "protein_g": round(actual_protein, 1),
                        "carbs_g": round(actual_carbs, 1),
                        "fat_g": round(actual_fat, 1)
                    },
                    "found_in_db": True
                }
                
                # Update meal totals
                enriched_meal["total_calories"] += actual_calories
                enriched_meal["total_protein_g"] += actual_protein
                enriched_meal["total_carbs_g"] += actual_carbs
                enriched_meal["total_fat_g"] += actual_fat
                
            else:
                # Food not found in database, use estimated values
                enriched_food = {
                    "name": food_name,
                    "quantity_g": quantity_g,
                    "reasoning": food_item.get("reasoning", ""),
                    "nutrition": {
                        "calories": 0,
                        "protein_g": 0,
                        "carbs_g": 0,
                        "fat_g": 0
                    },
                    "found_in_db": False,
                    "note": "Nutrition data not available in database"
                }
            
            enriched_meal["foods"].append(enriched_food)
        
        # Round totals
        enriched_meal["total_calories"] = round(enriched_meal["total_calories"], 1)
        enriched_meal["total_protein_g"] = round(enriched_meal["total_protein_g"], 1)
        enriched_meal["total_carbs_g"] = round(enriched_meal["total_carbs_g"], 1)
        enriched_meal["total_fat_g"] = round(enriched_meal["total_fat_g"], 1)
        
        enriched_plan.append(enriched_meal)
    
    return enriched_plan


def _get_general_foods_for_meal_type(db, meal_type: str) -> List[str]:
    """Get general food suggestions for a meal type when no user preferences are available."""
    try:
        # Define common foods for each meal type
        meal_type_foods = {
            "breakfast": [
                "oatmeal", "eggs", "yogurt", "banana", "bread", "milk", 
                "cereal", "apple", "orange", "toast", "pancakes", "waffles"
            ],
            "lunch": [
                "chicken breast", "rice", "salad", "sandwich", "soup", 
                "pasta", "fish", "vegetables", "beans", "quinoa", "turkey"
            ],
            "dinner": [
                "salmon", "steak", "pasta", "rice", "vegetables", 
                "chicken", "beef", "pork", "fish", "potatoes", "quinoa"
            ],
            "snack": [
                "apple", "banana", "nuts", "yogurt", "cheese", 
                "crackers", "carrots", "celery", "berries", "granola"
            ]
        }
        
        # Get the common foods for this meal type
        common_foods = meal_type_foods.get(meal_type.lower(), meal_type_foods["lunch"])
        
        # Try to find these foods in the database
        found_foods = []
        for food_name in common_foods:
            # Search for the food in the database
            query = """
            SELECT name FROM foods 
            WHERE LOWER(name) LIKE LOWER(:food_name)
            OR LOWER(name) LIKE LOWER(:food_name_pattern)
            LIMIT 1
            """
            
            rows = db.execute(
                text(query),
                {"food_name": food_name, "food_name_pattern": f"%{food_name}%"}
            ).fetchall()
            
            if rows:
                food_row = rows[0]
                if hasattr(food_row, '_mapping'):
                    found_food = dict(food_row._mapping)["name"]
                else:
                    found_food = food_row[0]  # Assuming name is the first column
                found_foods.append(found_food)
        
        # If we found foods in database, return them
        if found_foods:
            return found_foods[:8]  # Limit to 8 suggestions
        
        # If no foods found in database, return the common foods list
        return common_foods[:8]
        
    except Exception as e:
        logger.error(f"Error getting general foods for {meal_type}: {e}")
        # Return basic fallback foods
        return ["oatmeal", "eggs", "chicken breast", "rice", "vegetables"]
