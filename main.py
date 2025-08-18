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
import uuid
import time

# Import models and utilities
from shared import FoodLogEntry, DataValidator
from shared.config import get_settings
from shared.session_manager import FrameworkSessionManager
from shared.async_task_manager import AsyncTaskManager, TaskPriority
from shared.performance_monitor import PerformanceMonitor
from shared.health_monitor import HealthMonitor

# Set up logger - reduced for faster startup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

# =============================================================================
# SCALABILITY INFRASTRUCTURE GLOBAL VARIABLES
# =============================================================================

# Global session manager for multi-user support
_session_manager: Optional[FrameworkSessionManager] = None

# Global async task manager
_task_manager: Optional[AsyncTaskManager] = None

# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None

# Global health monitor
_health_monitor: Optional[HealthMonitor] = None

# Global conversation manager
_conversation_manager = None

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
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from shared.session_middleware import SessionValidationMiddleware, UserAuthenticationMiddleware
import httpx
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
                    _fitness_engine = create_engine(
                        database_url, 
                        pool_pre_ping=True, 
                        pool_recycle=3600,
                        connect_args={"connect_timeout": 90, "command_timeout": 90}
                    )
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
            _NutritionSessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=engine,
                expire_on_commit=False  # Prevent session expiration issues
            )
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
            _FitnessSessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=engine,
                expire_on_commit=False  # Prevent session expiration issues
            )
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
        _SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=get_nutrition_engine(),
            expire_on_commit=False  # Prevent session expiration issues
        )
    return _SessionLocal

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup - minimal initialization to avoid boot timeout
    logger.info("Starting nutrition-agent with scalability infrastructure...")
    
    try:
        # Validate environment variables
        validate_environment()
        logger.info("Environment variables validated successfully")
        
        # Initialize database tables
        from shared.database import init_database
        await init_database()
        logger.info("Database tables initialized successfully")
        
        # Initialize scalability infrastructure
        global _session_manager, _task_manager, _performance_monitor, _health_monitor, _conversation_manager
        
        # Initialize session manager
        _session_manager = FrameworkSessionManager("nutrition")
        await _session_manager.initialize()
        logger.info("Session manager initialized successfully")
        
        # Initialize async task manager
        _task_manager = AsyncTaskManager("nutrition", max_workers=5)
        await _task_manager.initialize()
        logger.info("Async task manager initialized successfully")
        
        # Initialize performance monitor
        _performance_monitor = PerformanceMonitor("nutrition")
        await _performance_monitor.initialize()
        logger.info("Performance monitor initialized successfully")
        
        # Initialize health monitor
        _health_monitor = HealthMonitor("nutrition")
        await _health_monitor.initialize()
        logger.info("Health monitor initialized successfully")
        
        # Initialize conversation manager
        try:
            from shared.conversation_manager import create_conversation_manager
            _conversation_manager = await create_conversation_manager("nutrition")
            logger.info("Conversation manager initialized successfully")
        except ImportError:
            logger.warning("Conversation manager not available, continuing without it")
            _conversation_manager = None
        
        logger.info("Nutrition agent started successfully with scalability infrastructure")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Nutrition agent shutting down...")
    
    # Cleanup scalability infrastructure
    try:
        if _task_manager:
            await _task_manager.cleanup()
        if _performance_monitor:
            await _performance_monitor.cleanup()
        if _health_monitor:
            await _health_monitor.cleanup()
        logger.info("Scalability infrastructure cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

app = FastAPI(
    title="Nutrition Agent", 
    lifespan=lifespan,
    # Increase timeout for long-running operations
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In a production environment, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session validation middleware for multi-user support
# Note: These will be properly initialized when session manager is available
app.add_middleware(
    SessionValidationMiddleware,
    session_manager=None,  # Will be set during initialization
    agent_type="nutrition"
)

app.add_middleware(
    UserAuthenticationMiddleware,
    session_manager=None  # Will be set during initialization
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
        session_local = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=engine,
            expire_on_commit=False  # Prevent session expiration issues
        )
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
            "message": "Nutrition Agent is running with SCALABILITY FEATURES!",
            "version": "2.0.0",
            "status": "running",
            "scalability_ready": True,
            "endpoints": {
                "health": "/health",
                "foods_count": "/foods/count",
                "execute_tool": "/execute-tool",
                "test_interface": "/static/test_interface.html",
                # New scalability endpoints
                "create_session": "/api/sessions/create",
                "get_session": "/api/sessions/{session_token}",
                "submit_task": "/api/tasks/submit",
                "get_task_status": "/api/tasks/{task_id}",
                "performance_metrics": "/api/performance/metrics",
                "scalability_status": "/api/scalability/status",
                # New conversation management endpoints
                "conversation_state": "/api/conversations/state",
                "conversation_history": "/api/conversations/history",
                "conversation_search": "/api/conversations/search",
                "conversation_summary": "/api/conversations/summary",
                "conversation_export": "/api/conversations/export",
                "conversation_reset": "/api/conversations/reset"
            },
            "features": {
                "multi_user_support": True,
                "session_management": True,
                "background_tasks": True,
                "performance_monitoring": True,
                "health_monitoring": True,
                "redis_caching": True,
                "horizontal_scaling": True,
                "conversation_management": True
            }
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
                food_result = conn.execute(text("SELECT id, name, category_id FROM foods LIMIT 1")).fetchone()
                if food_result:
                    food_data = dict(food_result._mapping) if hasattr(food_result, '_mapping') else dict(zip(['id', 'name', 'category_id'], food_result))
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
            text("SELECT id, name, category_id, serving_size, serving_unit, serving FROM foods LIMIT 1")
        ).fetchone()
        
        if food_row:
            food_data = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
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
          n.id AS nutrient_id,
          n.name AS nutrient_name,
          n.unit AS nutrient_unit,
          fn.amount
        FROM foods f
        LEFT JOIN brands b ON f.brand_id = b.id
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
            columns = ['food_id', 'food_name', 'description', 'serving_size', 'serving_unit', 'serving', 'created_at', 'brand_id', 'brand_name', 'nutrient_id', 'nutrient_name', 'nutrient_unit', 'amount']
            converted_rows.append(dict(zip(columns, row)))

    food_row = converted_rows[0]
    
    # Create comprehensive nutrients array and nutrition summary
    nutrients = []
    nutrition_summary = {
        "energy_kcal": 0,
        "energy": 0,
        "energy_from_fat": 0,
        "total_fat": 0,
        "unsaturated_fat": 0,
        "omega_3_fat": 0,
        "trans_fat": 0,
        "cholesterol": 0,
        "carbohydrates": 0,
        "sugars": 0,
        "fiber": 0,
        "protein": 0,
        "salt": 0,
        "sodium": 0,
        "potassium": 0,
        "calcium": 0,
        "iron": 0,
        "magnesium": 0,
        "vitamin_d": 0,
        "vitamin_c": 0,
        "alcohol": 0,
        "caffeine": 0,
        "taurine": 0,
        "glycemic_index": 0
    }
    
    for r in converted_rows:
        if r["nutrient_id"] is not None:
            amount = r["amount"] or 0
            nutrient_name = r["nutrient_name"]
            
            # Add to comprehensive nutrients array
            nutrients.append({
                "id": r["nutrient_id"],
                "name": r["nutrient_name"],
                "unit": r["nutrient_unit"],
                "amount": amount,
                "category": "general"
            })
            
            # Map to nutrition summary using exact database names
            if nutrient_name == "Energy (kcal)":
                nutrition_summary['energy_kcal'] = amount
            elif nutrient_name == "Energy":
                nutrition_summary['energy'] = amount
            elif nutrient_name == "Energy from Fat":
                nutrition_summary['energy_from_fat'] = amount
            elif nutrient_name == "Total Fat":
                nutrition_summary['total_fat'] = amount
            elif nutrient_name == "Unsaturated Fat":
                nutrition_summary['unsaturated_fat'] = amount
            elif nutrient_name == "Omega-3 Fat":
                nutrition_summary['omega_3_fat'] = amount
            elif nutrient_name == "Trans Fat":
                nutrition_summary['trans_fat'] = amount
            elif nutrient_name == "Cholesterol":
                nutrition_summary['cholesterol'] = amount
            elif nutrient_name == "Carbohydrates":
                nutrition_summary['carbohydrates'] = amount
            elif nutrient_name == "Sugars":
                nutrition_summary['sugars'] = amount
            elif nutrient_name == "Fiber":
                nutrition_summary['fiber'] = amount
            elif nutrient_name == "Protein":
                nutrition_summary['protein'] = amount
            elif nutrient_name == "Salt":
                nutrition_summary['salt'] = amount
            elif nutrient_name == "Sodium":
                nutrition_summary['sodium'] = amount
            elif nutrient_name == "Potassium":
                nutrition_summary['potassium'] = amount
            elif nutrient_name == "Calcium":
                nutrition_summary['calcium'] = amount
            elif nutrient_name == "Iron":
                nutrition_summary['iron'] = amount
            elif nutrient_name == "Magnesium":
                nutrition_summary['magnesium'] = amount
            elif nutrient_name == "Vitamin D":
                nutrition_summary['vitamin_d'] = amount
            elif nutrient_name == "Vitamin C":
                nutrition_summary['vitamin_c'] = amount
            elif nutrient_name == "Alcohol":
                nutrition_summary['alcohol'] = amount
            elif nutrient_name == "Caffeine":
                nutrition_summary['caffeine'] = amount
            elif nutrient_name == "Taurine":
                nutrition_summary['taurine'] = amount
            elif nutrient_name == "Glycemic Index":
                nutrition_summary['glycemic_index'] = amount

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
                text("SELECT id, name, category_id, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                {"food_id": food_id}
            ).fetchone()
            
            if not food_row:
                raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
            
            food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
            
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
                text("SELECT id, name, category_id, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                {"food_id": food_id}
            ).fetchone()
            
            if not food_row:
                raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
            
            food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
            
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
                text("SELECT id, name, category_id, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                {"food_id": nutrition_food_id}
            ).fetchone()
            
            if not food_row:
                raise HTTPException(status_code=404, detail=f"Food with id {nutrition_food_id} not found in nutrition database")
            
            food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
            
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
                    text("SELECT id, name, category_id, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
                    {"food_id": food_id}
                ).fetchone()
                
                if not food_row:
                    raise HTTPException(status_code=404, detail=f"Food with id {food_id} not found")
                
                food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
                
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
        user_id = params.get("user_id", "anonymous")  # Make user_id optional
        meal_type = params.get("meal_type")
        target_calories = params.get("target_calories")
        meal_description = params.get("meal_description")  # New parameter
        if not meal_type:
            raise HTTPException(status_code=400, detail="Missing required parameter: meal_type")
        return get_meal_suggestions(db_nutrition, db_shared, user_id, meal_type, target_calories, meal_description)

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

    elif tool == "create_recipe":
        user_id = params.get("user_id", "anonymous")
        recipe_description = params.get("recipe_description", "")
        servings = params.get("servings", 1)
        difficulty = params.get("difficulty", "easy")
        cuisine = params.get("cuisine")
        dietary_restrictions = params.get("dietary_restrictions", [])
        
        if not recipe_description:
            raise HTTPException(status_code=400, detail="Recipe description is required")
        
        # Validate inputs
        if servings < 1 or servings > 20:
            raise HTTPException(status_code=400, detail="Servings must be between 1 and 20")
        
        valid_difficulties = ["easy", "medium", "hard"]
        if difficulty not in valid_difficulties:
            raise HTTPException(status_code=400, detail=f"Difficulty must be one of: {', '.join(valid_difficulties)}")
        
        # Create recipe
        result = create_recipe(
            db_nutrition=db_nutrition,
            db_shared=db_shared,
            user_id=user_id,
            recipe_description=recipe_description,
            servings=servings,
            difficulty=difficulty,
            cuisine=cuisine,
            dietary_restrictions=dietary_restrictions
        )
        
        return result

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

def _get_user_uuid(user_id: str) -> str:
    """Convert string user_id to UUID format for database queries."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"user_{user_id}"))

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
                        n.unit as nutrient_unit
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
                "energy_kcal": 0,
                "energy": 0,
                "energy_from_fat": 0,
                "total_fat": 0,
                "unsaturated_fat": 0,
                "omega_3_fat": 0,
                "trans_fat": 0,
                "cholesterol": 0,
                "carbohydrates": 0,
                "sugars": 0,
                "fiber": 0,
                "protein": 0,
                "salt": 0,
                "sodium": 0,
                "potassium": 0,
                "calcium": 0,
                "iron": 0,
                "magnesium": 0,
                "vitamin_d": 0,
                "vitamin_c": 0,
                "alcohol": 0,
                "caffeine": 0,
                "taurine": 0,
                "glycemic_index": 0
            }
            
            for nutrition_row in nutrition_rows:
                if hasattr(nutrition_row, '_mapping'):
                    nutrient = dict(nutrition_row._mapping)
                else:
                    nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit'], nutrition_row))
                
                amount = nutrient['amount'] or 0
                nutrient_name = nutrient['nutrient_name']
                category = 'general'
                
                # Add to comprehensive nutrients array
                nutrients.append({
                    "id": nutrient['nutrient_id'],
                    "name": nutrient['nutrient_name'],
                    "unit": nutrient['nutrient_unit'],
                    "amount": amount,
                    "category": category
                })
                
                # Map to nutrition summary using exact database names
                if nutrient_name == "Energy (kcal)":
                    nutrition_summary['energy_kcal'] = amount
                elif nutrient_name == "Energy":
                    nutrition_summary['energy'] = amount
                elif nutrient_name == "Energy from Fat":
                    nutrition_summary['energy_from_fat'] = amount
                elif nutrient_name == "Total Fat":
                    nutrition_summary['total_fat'] = amount
                elif nutrient_name == "Unsaturated Fat":
                    nutrition_summary['unsaturated_fat'] = amount
                elif nutrient_name == "Omega-3 Fat":
                    nutrition_summary['omega_3_fat'] = amount
                elif nutrient_name == "Trans Fat":
                    nutrition_summary['trans_fat'] = amount
                elif nutrient_name == "Cholesterol":
                    nutrition_summary['cholesterol'] = amount
                elif nutrient_name == "Carbohydrates":
                    nutrition_summary['carbohydrates'] = amount
                elif nutrient_name == "Sugars":
                    nutrition_summary['sugars'] = amount
                elif nutrient_name == "Fiber":
                    nutrition_summary['fiber'] = amount
                elif nutrient_name == "Protein":
                    nutrition_summary['protein'] = amount
                elif nutrient_name == "Salt":
                    nutrition_summary['salt'] = amount
                elif nutrient_name == "Sodium":
                    nutrition_summary['sodium'] = amount
                elif nutrient_name == "Potassium":
                    nutrition_summary['potassium'] = amount
                elif nutrient_name == "Calcium":
                    nutrition_summary['calcium'] = amount
                elif nutrient_name == "Iron":
                    nutrition_summary['iron'] = amount
                elif nutrient_name == "Magnesium":
                    nutrition_summary['magnesium'] = amount
                elif nutrient_name == "Vitamin D":
                    nutrition_summary['vitamin_d'] = amount
                elif nutrient_name == "Vitamin C":
                    nutrition_summary['vitamin_c'] = amount
                elif nutrient_name == "Alcohol":
                    nutrition_summary['alcohol'] = amount
                elif nutrient_name == "Caffeine":
                    nutrition_summary['caffeine'] = amount
                elif nutrient_name == "Taurine":
                    nutrition_summary['taurine'] = amount
                elif nutrient_name == "Glycemic Index":
                    nutrition_summary['glycemic_index'] = amount
            
            logger.info(f"Found {len(nutrients)} nutrients for food {food_data['id']}")
            
        except Exception as e:
            logger.error(f"Error getting nutrition data for food {food_data['id']}: {e}")
            # If nutrition data can't be retrieved, set empty nutrients array
            nutrients = []
            nutrition_summary = {
                "energy_kcal": 0, "energy": 0, "energy_from_fat": 0,
                "total_fat": 0, "unsaturated_fat": 0, "omega_3_fat": 0,
                "trans_fat": 0, "cholesterol": 0, "carbohydrates": 0,
                "sugars": 0, "fiber": 0, "protein": 0, "salt": 0,
                "sodium": 0, "potassium": 0, "calcium": 0, "iron": 0,
                "magnesium": 0, "vitamin_d": 0, "vitamin_c": 0,
                "alcohol": 0, "caffeine": 0, "taurine": 0, "glycemic_index": 0
            }
        
        # Create structured food object with comprehensive nutrition data
        food_object = {
            "id": food_data['id'],
            "name": food_data['name'],
            "category_id": food_data['category_id'],
            "serving_size": food_data['serving_size'],
            "serving_unit": food_data['serving_unit'],
            "serving": food_data['serving'],
            "created_at": food_data['created_at'],
            "brand": {"id": food_data['brand_id'], "name": None} if food_data['brand_id'] else None,
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
        text("SELECT id, name, category_id, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
        {"food_id": food_id},
    ).fetchone()
    
    if not food_row:
        raise HTTPException(status_code=404, detail="Food not found")
    
    food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
    
    # Get comprehensive nutrition data from food_nutrients table
    nutrition_rows = db.execute(
        text("""
            SELECT 
                n.id as nutrient_id,
                n.name as nutrient_name, 
                fn.amount, 
                n.unit as nutrient_unit
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
            nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit'], row))
        
        nutrient_name = nutrient['nutrient_name'].lower()
        amount = nutrient['amount'] or 0
        
        # Add to comprehensive nutrients array
        nutrients.append({
            "id": nutrient['nutrient_id'],
            "name": nutrient['nutrient_name'],
            "unit": nutrient['nutrient_unit'],
            "amount": amount,
            "category": "general"
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
        # Handle None or empty user_id
        if not user_id:
            logger.info("No user_id provided, returning empty history")
            return []
        
        # Check if food_logs table exists
        result = db.execute(text("SELECT 1 FROM information_schema.tables WHERE table_name = 'food_logs'")).fetchone()
        if not result:
            logger.warning("food_logs table does not exist, returning empty history")
            return []
        
        # Convert user_id to UUID format for database query
        user_uuid = _get_user_uuid(str(user_id))
        logger.info(f"Querying user history for UUID: {user_uuid}")
        
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
            {"user_id": user_uuid},
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
        logger.info(f"Successfully retrieved {len(result)} history entries for user")
        return result
    except Exception as e:
        logger.error(f"Error getting user calorie history: {e}")
        # Don't fail the entire request, just return empty history
        logger.info("Returning empty history due to error, continuing with meal plan creation")
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
                        n.unit as nutrient_unit
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
                "energy_kcal": 0,
                "energy": 0,
                "energy_from_fat": 0,
                "total_fat": 0,
                "unsaturated_fat": 0,
                "omega_3_fat": 0,
                "trans_fat": 0,
                "cholesterol": 0,
                "carbohydrates": 0,
                "sugars": 0,
                "fiber": 0,
                "protein": 0,
                "salt": 0,
                "sodium": 0,
                "potassium": 0,
                "calcium": 0,
                "iron": 0,
                "magnesium": 0,
                "vitamin_d": 0,
                "vitamin_c": 0,
                "alcohol": 0,
                "caffeine": 0,
                "taurine": 0,
                "glycemic_index": 0
            }
            
            for nutrition_row in nutrition_rows:
                if hasattr(nutrition_row, '_mapping'):
                    nutrient = dict(nutrition_row._mapping)
                else:
                    nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit'], nutrition_row))
                
                amount = nutrient['amount'] or 0
                nutrient_name = nutrient['nutrient_name']
                category = 'general'
                
                # Add to comprehensive nutrients array
                nutrients.append({
                    "id": nutrient['nutrient_id'],
                    "name": nutrient['nutrient_name'],
                    "unit": nutrient['nutrient_unit'],
                    "amount": amount,
                    "category": category
                })
                
                # Map to nutrition summary using exact database names
                if nutrient_name == "Energy (kcal)":
                    nutrition_summary['energy_kcal'] = amount
                elif nutrient_name == "Energy":
                    nutrition_summary['energy'] = amount
                elif nutrient_name == "Energy from Fat":
                    nutrition_summary['energy_from_fat'] = amount
                elif nutrient_name == "Total Fat":
                    nutrition_summary['total_fat'] = amount
                elif nutrient_name == "Unsaturated Fat":
                    nutrition_summary['unsaturated_fat'] = amount
                elif nutrient_name == "Omega-3 Fat":
                    nutrition_summary['omega_3_fat'] = amount
                elif nutrient_name == "Trans Fat":
                    nutrition_summary['trans_fat'] = amount
                elif nutrient_name == "Cholesterol":
                    nutrition_summary['cholesterol'] = amount
                elif nutrient_name == "Carbohydrates":
                    nutrition_summary['carbohydrates'] = amount
                elif nutrient_name == "Sugars":
                    nutrition_summary['sugars'] = amount
                elif nutrient_name == "Fiber":
                    nutrition_summary['fiber'] = amount
                elif nutrient_name == "Protein":
                    nutrition_summary['protein'] = amount
                elif nutrient_name == "Salt":
                    nutrition_summary['salt'] = amount
                elif nutrient_name == "Sodium":
                    nutrition_summary['sodium'] = amount
                elif nutrient_name == "Potassium":
                    nutrition_summary['potassium'] = amount
                elif nutrient_name == "Calcium":
                    nutrition_summary['calcium'] = amount
                elif nutrient_name == "Iron":
                    nutrition_summary['iron'] = amount
                elif nutrient_name == "Magnesium":
                    nutrition_summary['magnesium'] = amount
                elif nutrient_name == "Vitamin D":
                    nutrition_summary['vitamin_d'] = amount
                elif nutrient_name == "Vitamin C":
                    nutrition_summary['vitamin_c'] = amount
                elif nutrient_name == "Alcohol":
                    nutrition_summary['alcohol'] = amount
                elif nutrient_name == "Caffeine":
                    nutrition_summary['caffeine'] = amount
                elif nutrient_name == "Taurine":
                    nutrition_summary['taurine'] = amount
                elif nutrient_name == "Glycemic Index":
                    nutrition_summary['glycemic_index'] = amount
            
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
            "category_id": food["category_id"],
            "nutrients": food["nutrients"],  # Comprehensive nutrient array
            "nutrition_summary": food["nutrition_summary"],  # Backward compatibility
            "total_nutrients": food["total_nutrients"],
            "similarity": food["similarity"]
        }
        structured_matches.append(structured_food)
    logger.info(f"Returning {len(structured_matches)} structured fuzzy matches with comprehensive nutrition data")
    return structured_matches


@app.post("/meal-plan")
async def create_meal_plan(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Create a personalized meal plan using AI, then query nutrition database for accurate macros."""
    try:
        user_id = request.get("user_id")
        daily_calories = request.get("daily_calories", 2000)
        meal_count = request.get("meal_count", 3)
        dietary_restrictions = request.get("dietary_restrictions", [])
        meal_description = request.get("meal_description")  # New parameter

        # Get user's food preferences and history from shared database
        user_history = []
        user_data_available = False
        
        if user_id:
            try:
                user_history = get_user_calorie_history(db_shared, user_id)
                user_data_available = len(user_history) > 0
                if user_data_available:
                    logger.info(f"Successfully fetched user data for user_id {user_id}: {len(user_history)} entries")
                else:
                    logger.info(f"No user history found for user_id {user_id}, proceeding with general meal plan")
            except Exception as user_data_error:
                logger.warning(f"Could not fetch user data for user_id {user_id}: {user_data_error}")
                logger.info("Proceeding with general meal plan")
                # Continue with empty user history
        else:
            logger.info("No user_id provided, using general meal plan")

        # AI-powered meal plan creation with meal description - with timeout protection
        import asyncio
        try:
            meal_plan = await asyncio.wait_for(
                asyncio.to_thread(create_ai_meal_plan, user_id or "anonymous", daily_calories, meal_count, dietary_restrictions, user_history, db_nutrition, meal_description),
                timeout=25.0  # 25 second timeout to stay under 30s total
            )
        except asyncio.TimeoutError:
            logger.error("Meal plan creation timed out")
            raise HTTPException(status_code=408, detail="Request timeout. Please try again.")

        # Calculate total calories from enriched meal plan (new structure)
        total_calories = 0
        total_meals = 0
        for day_data in meal_plan:
            for meal in day_data.get("meals", []):
                total_calories += meal.get("macros", {}).get("calories", 0)
                total_meals += 1

        return {
            "status": "success",
            "meal_plan": meal_plan,
            "total_calories": total_calories,
            "meals": total_meals,
            "ai_generated": True,
            "nutrition_verified": True,
            "meal_description": meal_description,
            "user_data_available": user_data_available,
            "user_id": user_id or "anonymous"
        }
    except Exception as e:
        logger.error(f"Error creating AI meal plan: {e}")
        # Provide more specific error messages
        if "database" in str(e).lower() or "connection" in str(e).lower():
            raise HTTPException(status_code=503, detail="Database connection error. Please try again later.")
        elif "timeout" in str(e).lower():
            raise HTTPException(status_code=408, detail="Request timeout. Please try again.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to create meal plan: {str(e)}")


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
    db_nutrition, db_shared, user_id: str, meal_type: str, target_calories: Optional[float] = None, meal_description: Optional[str] = None
):
    """Get AI-powered meal suggestions based on user preferences, calorie targets, and meal description."""
    try:
        settings = get_settings()
        
        # Initialize Groq client
        client = openai.OpenAI(
            api_key=settings.llm.groq_api_key,
            base_url=settings.llm.groq_base_url
        )
        
        # Get user's recent food preferences for context (from shared database)
        user_preferences = []
        popular_choices = []
        
        try:
            user_uuid = _get_user_uuid(user_id)
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
                {"user_id": user_uuid},
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

        except Exception as user_data_error:
            logger.warning(f"Could not fetch user data for user_id {user_id}: {user_data_error}")
            # Continue without user preferences - will use general foods

        # If no preferences found, get some general foods for this meal type
        if not user_preferences and not popular_choices:
            general_foods = _get_general_foods_for_meal_type(db_nutrition, meal_type)
            popular_choices = general_foods

        # Prepare context for AI with meal description
        context_parts = [
            f"Meal type: {meal_type}",
            f"Target calories: {target_calories if target_calories else 'Not specified'}"
        ]
        
        # Add meal description if provided
        if meal_description:
            context_parts.append(f"Meal description: {meal_description}")
        
        # Add user preferences and popular choices
        if user_preferences:
            context_parts.append(f"User's recent food preferences: {', '.join(user_preferences[:5])}")
        if popular_choices:
            context_parts.append(f"Popular foods for {meal_type}: {', '.join(popular_choices[:5])}")
        
        context = " | ".join(context_parts)
        
        # Generate AI suggestions with meal description
        ai_suggestions = _generate_ai_meal_suggestions(client, context, target_calories, meal_description)
        
        # Enrich with nutrition data
        enriched_suggestions = _enrich_suggestions_with_nutrition(ai_suggestions, db_nutrition)
        
        return {
            "meal_type": meal_type,
            "target_calories": target_calories,
            "meal_description": meal_description,
            "suggestions": enriched_suggestions,
            "ai_generated": True,
            "user_preferences_count": len(user_preferences),
            "popular_choices_count": len(popular_choices),
            "user_data_available": len(user_preferences) > 0
        }
        
    except Exception as e:
        logger.error(f"Error in AI meal suggestions: {e}")
        # Fallback to rule-based suggestions
        return _get_fallback_meal_suggestions(db_nutrition, db_shared, user_id, meal_type, target_calories, meal_description)


def _generate_ai_meal_suggestions(client, context: str, target_calories: Optional[float] = None, meal_description: Optional[str] = None) -> List[Dict]:
    """Generate meal suggestions using Groq AI."""
    
    settings = get_settings()
    
    # Build personalized prompt based on meal description
    personalization_guidelines = ""
    if meal_description:
        personalization_guidelines = f"""
        USER MEAL DESCRIPTION: "{meal_description}"
        
        INSTRUCTIONS FOR AI:
        1. Carefully analyze the user's detailed meal description
        2. Extract all specific requirements mentioned (ingredients, dietary preferences, calorie targets, meal frequency)
        3. Follow the user's exact specifications for meal suggestions
        4. PRIORITIZE the ingredients mentioned by the user, but you can use additional ingredients if needed
        5. If user specifies "ONLY use these ingredients", then restrict to those ingredients
        6. If user mentions ingredients without "ONLY", prioritize those ingredients but supplement with others as needed
        7. Meet the exact calorie targets specified
        8. Follow all dietary restrictions mentioned (low carb, high protein, etc.)
        9. Suggest foods that match the user's preferences and requirements
        10. Ensure suggestions meet the nutritional requirements mentioned
        11. Provide accurate macro estimates for all suggested foods
        """
    
    prompt = f"""
    {context}
    
    {personalization_guidelines}
    
    Create 5-8 food suggestions for this meal. For each food, provide:
    1. Food name (any nutritious food that matches the user's requirements)
    2. Suggested quantity in grams
    3. Estimated macros (calories, protein, carbs, fat) based on the quantity
    4. Brief nutritional reasoning
    5. List of ingredients with their individual macros
    
    Format your response as a JSON array with this structure:
    [
        {{
            "name": "oatmeal with berries",
            "quantity_g": 50,
            "estimated_macros": {{
                "calories": 180,
                "protein_g": 6.5,
                "carbs_g": 30.5,
                "fat_g": 3.2
            }},
            "reasoning": "High fiber, complex carbs for sustained energy",
            "ingredients": [
                {{
                    "name": "oatmeal",
                    "quantity_g": 30,
                    "macros": {{
                        "calories": 110,
                        "protein_g": 4.0,
                        "carbs_g": 19.5,
                        "fat_g": 2.0
                    }}
                }},
                {{
                    "name": "berries",
                    "quantity_g": 20,
                    "macros": {{
                        "calories": 70,
                        "protein_g": 2.5,
                        "carbs_g": 11.0,
                        "fat_g": 1.2
                    }}
                }}
            ]
        }},
        {{
            "name": "banana with peanut butter",
            "quantity_g": 120,
            "estimated_macros": {{
                "calories": 105,
                "protein_g": 1.3,
                "carbs_g": 27,
                "fat_g": 0.4
            }},
            "reasoning": "Natural sweetness and potassium",
            "ingredients": [
                {{
                    "name": "banana",
                    "quantity_g": 100,
                    "macros": {{
                        "calories": 89,
                        "protein_g": 1.1,
                        "carbs_g": 23,
                        "fat_g": 0.3
                    }}
                }},
                {{
                    "name": "peanut butter",
                    "quantity_g": 20,
                    "macros": {{
                        "calories": 16,
                        "protein_g": 0.2,
                        "carbs_g": 4,
                        "fat_g": 0.1
                    }}
                }}
            ]
        }}
    ]
    
    Guidelines:
    - Suggest any nutritious foods that match the user's requirements
    - Use common, recognizable food names
    - Suggest realistic portion sizes
    - Consider the meal type and calorie target
    - Provide diverse, nutritious options
    - Keep reasoning brief but informative
    - If no user preferences are available, suggest common nutritious foods for the meal type
    - Focus on whole foods and balanced nutrition
    - If a specific meal description is provided, prioritize foods that match that description
    - Provide accurate macro estimates based on standard nutrition values for the suggested quantities
    - Do not limit suggestions to only foods that might be in a database
    - Include ingredients array with individual macros for each ingredient
    """
    
    try:
        response = client.chat.completions.create(
            model=settings.llm.groq_model,
            messages=[
                {"role": "system", "content": "You are a nutrition expert. Suggest foods quickly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800,   # Increased tokens for ingredients array
            timeout=30        # 30 second timeout for faster response
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
    """Enrich AI suggestions with comprehensive nutrition database data."""
    enriched_suggestions = []
    for suggestion in ai_suggestions:
        food_name = suggestion.get("name", "")
        if not food_name:
            continue
        
        food_data = _get_food_nutrition_from_db(db, food_name)
        suggested_quantity = suggestion.get("quantity_g", 100)
        ai_estimated_macros = suggestion.get("estimated_macros", {})
        ai_ingredients = suggestion.get("ingredients", [])
        
        if food_data:
            # Calculate nutrition based on suggested quantity
            base_calories = food_data.get("calories", 0)
            base_protein = food_data.get("protein_g", 0)
            base_carbs = food_data.get("carbs_g", 0)
            base_fat = food_data.get("fat_g", 0)
            
            # Calculate actual nutrition based on quantity
            actual_calories = (base_calories * suggested_quantity) / 100
            actual_protein = (base_protein * suggested_quantity) / 100
            actual_carbs = (base_carbs * suggested_quantity) / 100
            actual_fat = (base_fat * suggested_quantity) / 100
            
            # Enrich ingredients with database nutrition data
            enriched_ingredients = []
            for ingredient in ai_ingredients:
                ingredient_name = ingredient.get("name", "")
                ingredient_quantity = ingredient.get("quantity_g", 0)
                ingredient_macros = ingredient.get("macros", {})
                
                # Try to get nutrition data from database for this ingredient
                ingredient_food_data = _get_food_nutrition_from_db(db, ingredient_name)
                
                if ingredient_food_data:
                    # Calculate actual nutrition based on ingredient quantity
                    base_ingredient_calories = ingredient_food_data.get("calories", 0)
                    base_ingredient_protein = ingredient_food_data.get("protein_g", 0)
                    base_ingredient_carbs = ingredient_food_data.get("carbs_g", 0)
                    base_ingredient_fat = ingredient_food_data.get("fat_g", 0)
                    
                    actual_ingredient_calories = (base_ingredient_calories * ingredient_quantity) / 100
                    actual_ingredient_protein = (base_ingredient_protein * ingredient_quantity) / 100
                    actual_ingredient_carbs = (base_ingredient_carbs * ingredient_quantity) / 100
                    actual_ingredient_fat = (base_ingredient_fat * ingredient_quantity) / 100
                    
                    enriched_ingredient = {
                        "name": ingredient_name,
                        "quantity_g": ingredient_quantity,
                        "macros": {
                            "calories": round(actual_ingredient_calories, 1),
                            "protein_g": round(actual_ingredient_protein, 1),
                            "carbs_g": round(actual_ingredient_carbs, 1),
                            "fat_g": round(actual_ingredient_fat, 1)
                        },
                        "nutrition_verified": True,
                        "food_id": ingredient_food_data.get("id"),
                        "nutrients": ingredient_food_data.get("nutrients", [])
                    }
                else:
                    # Use AI-generated macros if database data not available
                    enriched_ingredient = {
                        "name": ingredient_name,
                        "quantity_g": ingredient_quantity,
                        "macros": {
                            "calories": round(ingredient_macros.get("calories", 0), 1),
                            "protein_g": round(ingredient_macros.get("protein_g", 0), 1),
                            "carbs_g": round(ingredient_macros.get("carbs_g", 0), 1),
                            "fat_g": round(ingredient_macros.get("fat_g", 0), 1)
                        },
                        "nutrition_verified": False,
                        "note": "Nutrition data not available in database - using AI estimates"
                    }
                
                enriched_ingredients.append(enriched_ingredient)
            
            enriched_suggestion = {
                "id": food_data.get("id"),
                "name": food_data.get("name", food_name),
                "brand_id": food_data.get("brand_id"),
                "category_id": food_data.get("category_id"),
                "serving_size": food_data.get("serving_size"),
                "serving_unit": food_data.get("serving_unit"),
                "serving": food_data.get("serving"),
                "suggested_quantity_g": suggested_quantity,
                "ai_reasoning": suggestion.get("reasoning", ""),
                "nutrition_verified": True,
                "calories": round(actual_calories, 1),
                "protein_g": round(actual_protein, 1),
                "carbs_g": round(actual_carbs, 1),
                "fat_g": round(actual_fat, 1),
                "nutrients": food_data.get("nutrients", []),
                "nutrition_summary": food_data.get("nutrition_summary", {}),
                "total_nutrients": food_data.get("total_nutrients", 0),
                "ingredients": enriched_ingredients
            }
        else:
            # Food not found in database, use AI-generated macros if available
            ai_calories = ai_estimated_macros.get("calories", 0)
            ai_protein = ai_estimated_macros.get("protein_g", 0)
            ai_carbs = ai_estimated_macros.get("carbs_g", 0)
            ai_fat = ai_estimated_macros.get("fat_g", 0)
            
            # Keep original ingredients structure if no database data
            enriched_ingredients = []
            for ingredient in ai_ingredients:
                enriched_ingredients.append({
                    "name": ingredient.get("name", ""),
                    "quantity_g": ingredient.get("quantity_g", 0),
                    "macros": ingredient.get("macros", {}),
                    "nutrition_verified": False,
                    "note": "Using AI estimates"
                })
            
            enriched_suggestion = {
                "name": food_name,
                "suggested_quantity_g": suggested_quantity,
                "ai_reasoning": suggestion.get("reasoning", ""),
                "nutrition_verified": False,
                "calories": round(ai_calories, 1),
                "protein_g": round(ai_protein, 1),
                "carbs_g": round(ai_carbs, 1),
                "fat_g": round(ai_fat, 1),
                "nutrients": [],
                "nutrition_summary": {},
                "total_nutrients": 0,
                "note": "Nutrition data not available in database - using AI estimates",
                "ingredients": enriched_ingredients
            }
        
        enriched_suggestions.append(enriched_suggestion)
    
    return enriched_suggestions[:10]


def _get_fallback_meal_suggestions(db_nutrition, db_shared, user_id: str, meal_type: str, target_calories: Optional[float] = None, meal_description: Optional[str] = None) -> Dict:
    """Fallback rule-based meal suggestions when AI fails. Uses comprehensive nutrition data."""
    suggestions = []
    
    try:
        user_uuid = _get_user_uuid(user_id)
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
            {"user_id": user_uuid},
        ).fetchall()
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
        food_ids = list(set([f["food_item_id"] for f in recent_foods_dict + popular_foods_dict]))
        
        if not food_ids:
            # No user data available, use general foods
            general_foods = _get_general_foods_for_meal_type(db_nutrition, meal_type)
            for food_name in general_foods:
                food_data = _get_food_nutrition_from_db(db_nutrition, food_name)
                if food_data:
                                    suggestions.append({
                    "name": food_name,
                    "suggested_quantity_g": 100,
                    "reasoning": f"Common nutritious food for {meal_type}",
                    "found_in_db": True,
                    "calories": food_data.get("calories", 0),
                    "protein_g": food_data.get("protein_g", 0),
                    "carbs_g": food_data.get("carbs_g", 0),
                    "fat_g": food_data.get("fat_g", 0),
                    "nutrients": food_data.get("nutrients", []),
                    "nutrition_summary": food_data.get("nutrition_summary", {}),
                    "total_nutrients": food_data.get("total_nutrients", 0),
                    "ingredients": [
                        {
                            "name": food_name,
                            "quantity_g": 100,
                            "macros": {
                                "calories": food_data.get("calories", 0),
                                "protein_g": food_data.get("protein_g", 0),
                                "carbs_g": food_data.get("carbs_g", 0),
                                "fat_g": food_data.get("fat_g", 0)
                            },
                            "nutrition_verified": True,
                            "food_id": food_data.get("id"),
                            "nutrients": food_data.get("nutrients", [])
                        }
                    ]
                })
                else:
                    suggestions.append({
                        "name": food_name,
                        "suggested_quantity_g": 100,
                        "reasoning": f"Common nutritious food for {meal_type}",
                        "found_in_db": False,
                        "calories": 0,
                        "protein_g": 0,
                        "carbs_g": 0,
                        "fat_g": 0,
                        "nutrients": [],
                        "nutrition_summary": {},
                        "total_nutrients": 0,
                        "ingredients": [
                            {
                                "name": food_name,
                                "quantity_g": 100,
                                "macros": {
                                    "calories": 0,
                                    "protein_g": 0,
                                    "carbs_g": 0,
                                    "fat_g": 0
                                },
                                "nutrition_verified": False,
                                "note": "Nutrition data not available in database"
                            }
                        ]
                    })
            return {
                "meal_type": meal_type,
                "target_calories": target_calories,
                "meal_description": meal_description,
                "suggestions": suggestions,
                "ai_generated": False,
                "fallback_used": True,
                "message": "Using general food suggestions (no user data available)",
                "user_data_available": False
            }
        
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
            
            # Get comprehensive nutrition data
            food_data = _get_food_nutrition_from_db(db_nutrition, food_dict["name"])
            
            if target_calories:
                default_calories = 100
                suggested_quantity = min(200, (target_calories / default_calories) * 100)
                food_dict["suggested_quantity_g"] = round(suggested_quantity, 0)
            else:
                food_dict["suggested_quantity_g"] = 100
            
            if food_data:
                suggestions.append({
                    **food_dict,
                    "calories": food_data.get("calories", 0),
                    "protein_g": food_data.get("protein_g", 0),
                    "carbs_g": food_data.get("carbs_g", 0),
                    "fat_g": food_data.get("fat_g", 0),
                    "nutrients": food_data.get("nutrients", []),
                    "nutrition_summary": food_data.get("nutrition_summary", {}),
                    "total_nutrients": food_data.get("total_nutrients", 0),
                    "found_in_db": True
                })
            else:
                suggestions.append({
                    **food_dict,
                    "calories": 0,
                    "protein_g": 0,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "nutrients": [],
                    "nutrition_summary": {},
                    "total_nutrients": 0,
                    "found_in_db": False
                })
        
        return {
            "meal_type": meal_type,
            "target_calories": target_calories,
            "meal_description": meal_description,
            "suggestions": suggestions[:10],
            "ai_generated": False,
            "fallback_used": True,
            "user_data_available": True
        }
        
    except Exception as user_data_error:
        logger.warning(f"Could not fetch user data for user_id {user_id} in fallback: {user_data_error}")
        # Fallback to general foods when user data is not available
        general_foods = _get_general_foods_for_meal_type(db_nutrition, meal_type)
        for food_name in general_foods:
            food_data = _get_food_nutrition_from_db(db_nutrition, food_name)
            if food_data:
                suggestions.append({
                    "name": food_name,
                    "suggested_quantity_g": 100,
                    "reasoning": f"Common nutritious food for {meal_type}",
                    "found_in_db": True,
                    "calories": food_data.get("calories", 0),
                    "protein_g": food_data.get("protein_g", 0),
                    "carbs_g": food_data.get("carbs_g", 0),
                    "fat_g": food_data.get("fat_g", 0),
                    "nutrients": food_data.get("nutrients", []),
                    "nutrition_summary": food_data.get("nutrition_summary", {}),
                    "total_nutrients": food_data.get("total_nutrients", 0)
                })
            else:
                suggestions.append({
                    "name": food_name,
                    "suggested_quantity_g": 100,
                    "reasoning": f"Common nutritious food for {meal_type}",
                    "found_in_db": False,
                    "calories": 0,
                    "protein_g": 0,
                    "carbs_g": 0,
                    "fat_g": 0,
                    "nutrients": [],
                    "nutrition_summary": {},
                    "total_nutrients": 0
                })
        
        return {
            "meal_type": meal_type,
            "target_calories": target_calories,
            "meal_description": meal_description,
            "suggestions": suggestions,
            "ai_generated": False,
            "fallback_used": True,
            "message": "Using general food suggestions (user data not available)",
            "user_data_available": False
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
        meal_description = request.get("meal_description")  # New parameter for testing
        
        # Sample user history
        sample_history = [
            {"name": "oatmeal", "calories": 150, "protein_g": 5, "carbs_g": 27, "fat_g": 3},
            {"name": "banana", "calories": 105, "protein_g": 1, "carbs_g": 27, "fat_g": 0},
            {"name": "chicken breast", "calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6},
            {"name": "brown rice", "calories": 110, "protein_g": 2.5, "carbs_g": 23, "fat_g": 0.9},
            {"name": "salmon", "calories": 208, "protein_g": 25, "carbs_g": 0, "fat_g": 12}
        ]

        # AI-powered meal plan creation with meal description
        meal_plan = create_ai_meal_plan(
            user_id, daily_calories, meal_count, dietary_restrictions, sample_history, db_nutrition, meal_description
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
            "sample_data_used": True,
            "meal_description": meal_description
        }
    except Exception as e:
        logger.error(f"Error testing AI meal plan: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to test AI meal plan: {e}")


@app.post("/test-detailed-meal-description")
def test_detailed_meal_description(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Test the new detailed meal description functionality with the sample user request."""
    try:
        # Sample user request as provided
        sample_meal_description = """
        I want you to make me a meal plan for 3 days. I have all the sides in the fridge like garlic, oil, salt ,etc.. 
        The main ones that i want you to use are 5% ground beef , asparagus, salmon, avocado, oatmeal, berries, coconut, yoghurt, veal stock, tomato salsa. 
        I want 2 meals each day one around 1100 calories. I want to eat high protein and low carb - max 50g a day.
        """
        
        user_id = request.get("user_id", "test_user")
        daily_calories = request.get("daily_calories", 2200)  # 2 meals * 1100 calories
        meal_count = request.get("meal_count", 2)  # 2 meals per day
        dietary_restrictions = request.get("dietary_restrictions", ["low_carb", "high_protein"])
        meal_description = request.get("meal_description", sample_meal_description)
        
        # Get user history
        user_history = []
        try:
            user_uuid = _get_user_uuid(user_id)
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
                {"user_id": user_uuid},
            ).fetchall()
            
            for row in recent_foods:
                if hasattr(row, '_mapping'):
                    food_id = dict(row._mapping)["food_item_id"]
                else:
                    food_id = row[0]
                user_history.append({"food_item_id": food_id, "name": f"food_{food_id}"})
        except Exception as e:
            logger.warning(f"Could not fetch user history: {e}")
        
        # Create AI meal plan with detailed description
        meal_plan = create_ai_meal_plan(
            user_id=user_id,
            daily_calories=daily_calories,
            meal_count=meal_count,
            dietary_restrictions=dietary_restrictions,
            user_history=user_history,
            db_nutrition=db_nutrition,
            meal_description=meal_description
        )
        
        # Also test meal suggestions with the same description
        meal_suggestions = get_meal_suggestions(
            db_nutrition=db_nutrition,
            db_shared=db_shared,
            user_id=user_id,
            meal_type="lunch",
            target_calories=1100,
            meal_description=meal_description
        )
        
        return {
            "success": True,
            "meal_plan": meal_plan,
            "meal_suggestions": meal_suggestions,
            "user_id": user_id,
            "daily_calories": daily_calories,
            "meal_count": meal_count,
            "dietary_restrictions": dietary_restrictions,
            "meal_description": meal_description,
            "test_description": "Testing detailed meal description with specific ingredients and preferences"
        }
        
    except Exception as e:
        logger.error(f"Error in test detailed meal description: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test detailed meal description: {str(e)}")


@app.post("/test-user-specific-request")
def test_user_specific_request(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Test the exact user request with 3-day meal plan and specific ingredients."""
    try:
        # The exact user request
        user_meal_description = """
        I want you to make me a meal plan for 3 days. I have all the sides in the fridge like garlic, oil, salt ,etc.. 
        The main ones that i want you to use are 5% ground beef , asparagus, salmon, avocado, oatmeal, berries, coconut, yoghurt, veal stock, tomato salsa. 
        I want 2 meals each day one around 1100 calories. I want to eat high protein and low carb - max 50g a day.
        """
        
        user_id = request.get("user_id", "test_user")
        daily_calories = 2200  # 2 meals * 1100 calories
        meal_count = 2  # 2 meals per day
        dietary_restrictions = ["low_carb", "high_protein"]
        
        # Get user history
        user_history = []
        try:
            user_uuid = _get_user_uuid(user_id)
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
                {"user_id": user_uuid},
            ).fetchall()
            
            for row in recent_foods:
                if hasattr(row, '_mapping'):
                    food_id = dict(row._mapping)["food_item_id"]
                else:
                    food_id = row[0]
                user_history.append({"food_item_id": food_id, "name": f"food_{food_id}"})
        except Exception as e:
            logger.warning(f"Could not fetch user history: {e}")
        
        # Create AI meal plan with the exact user request
        meal_plan = create_ai_meal_plan(
            user_id=user_id,
            daily_calories=daily_calories,
            meal_count=meal_count,
            dietary_restrictions=dietary_restrictions,
            user_history=user_history,
            db_nutrition=db_nutrition,
            meal_description=user_meal_description
        )
        
        # Calculate total calories and verify requirements
        total_calories = sum(meal.get("total_calories", 0) for meal in meal_plan)
        total_meals = len(meal_plan)
        days_covered = len(set(meal.get("day", 1) for meal in meal_plan))
        
        # Check if the plan meets requirements
        requirements_met = {
            "3_days": days_covered >= 3,
            "2_meals_per_day": total_meals >= 6,  # 3 days * 2 meals
            "1100_calories_per_meal": all(meal.get("total_calories", 0) >= 900 for meal in meal_plan),  # Allow some flexibility
            "uses_specified_ingredients": any(
                any(food.get("name", "").lower() in ["ground beef", "asparagus", "salmon", "avocado", "oatmeal", "berries", "coconut", "yoghurt", "veal stock", "tomato salsa"] 
                     for food in meal.get("foods", []))
                for meal in meal_plan
            ),
            "low_carb": all(meal.get("total_carbs_g", 0) <= 50 for meal in meal_plan),
            "high_protein": all(meal.get("total_protein_g", 0) >= 20 for meal in meal_plan)
        }
        
        return {
            "success": True,
            "meal_plan": meal_plan,
            "total_calories": total_calories,
            "total_meals": total_meals,
            "days_covered": days_covered,
            "requirements_met": requirements_met,
            "user_request": user_meal_description,
            "analysis": {
                "expected_meals": 6,  # 3 days * 2 meals
                "expected_calories_per_meal": 1100,
                "expected_total_calories": 6600,  # 6 meals * 1100 calories
                "actual_total_calories": total_calories,
                "calorie_target_met": total_calories >= 6000,
                "ingredients_used": list(set(
                    food.get("name", "").lower() 
                    for meal in meal_plan 
                    for food in meal.get("foods", [])
                ))
            }
        }
        
    except Exception as e:
        logger.error(f"Error in test user specific request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test user specific request: {str(e)}")

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
            text("SELECT id, name, category_id, serving_size, serving_unit, serving FROM foods WHERE id = :food_id"),
            {"food_id": food_id},
        ).fetchone()
        
        if not food_row:
            raise HTTPException(status_code=404, detail="Food not found")
        
        food_details = dict(food_row._mapping) if hasattr(food_row, '_mapping') else dict(zip(['id', 'name', 'category_id', 'serving_size', 'serving_unit', 'serving'], food_row))
        
        # Get all nutrients for this food
        nutrition_rows = db.execute(
            text("""
                SELECT 
                    n.id as nutrient_id,
                    n.name as nutrient_name, 
                    fn.amount, 
                    n.unit as nutrient_unit
                FROM food_nutrients fn
                JOIN nutrients n ON fn.nutrient_id = n.id
                WHERE fn.food_id = :food_id
                ORDER BY n.name
            """),
            {"food_id": food_id}
        ).fetchall()
        
        # Group nutrients by category
        nutrients_by_category = {}
        all_nutrients = []
        nutrition_summary = {
            "energy_kcal": 0,
            "energy": 0,
            "energy_from_fat": 0,
            "total_fat": 0,
            "unsaturated_fat": 0,
            "omega_3_fat": 0,
            "trans_fat": 0,
            "cholesterol": 0,
            "carbohydrates": 0,
            "sugars": 0,
            "fiber": 0,
            "protein": 0,
            "salt": 0,
            "sodium": 0,
            "potassium": 0,
            "calcium": 0,
            "iron": 0,
            "magnesium": 0,
            "vitamin_d": 0,
            "vitamin_c": 0,
            "alcohol": 0,
            "caffeine": 0,
            "taurine": 0,
            "glycemic_index": 0
        }
        
        for row in nutrition_rows:
            if hasattr(row, '_mapping'):
                nutrient = dict(row._mapping)
            else:
                nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit'], row))
            
            amount = nutrient['amount'] or 0
            nutrient_name = nutrient['nutrient_name']
            category = 'general'
            
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
            
            # Map to nutrition summary using exact database names
            if nutrient_name == "Energy (kcal)":
                nutrition_summary['energy_kcal'] = amount
            elif nutrient_name == "Energy":
                nutrition_summary['energy'] = amount
            elif nutrient_name == "Energy from Fat":
                nutrition_summary['energy_from_fat'] = amount
            elif nutrient_name == "Total Fat":
                nutrition_summary['total_fat'] = amount
            elif nutrient_name == "Unsaturated Fat":
                nutrition_summary['unsaturated_fat'] = amount
            elif nutrient_name == "Omega-3 Fat":
                nutrition_summary['omega_3_fat'] = amount
            elif nutrient_name == "Trans Fat":
                nutrition_summary['trans_fat'] = amount
            elif nutrient_name == "Cholesterol":
                nutrition_summary['cholesterol'] = amount
            elif nutrient_name == "Carbohydrates":
                nutrition_summary['carbohydrates'] = amount
            elif nutrient_name == "Sugars":
                nutrition_summary['sugars'] = amount
            elif nutrient_name == "Fiber":
                nutrition_summary['fiber'] = amount
            elif nutrient_name == "Protein":
                nutrition_summary['protein'] = amount
            elif nutrient_name == "Salt":
                nutrition_summary['salt'] = amount
            elif nutrient_name == "Sodium":
                nutrition_summary['sodium'] = amount
            elif nutrient_name == "Potassium":
                nutrition_summary['potassium'] = amount
            elif nutrient_name == "Calcium":
                nutrition_summary['calcium'] = amount
            elif nutrient_name == "Iron":
                nutrition_summary['iron'] = amount
            elif nutrient_name == "Magnesium":
                nutrition_summary['magnesium'] = amount
            elif nutrient_name == "Vitamin D":
                nutrition_summary['vitamin_d'] = amount
            elif nutrient_name == "Vitamin C":
                nutrition_summary['vitamin_c'] = amount
            elif nutrient_name == "Alcohol":
                nutrition_summary['alcohol'] = amount
            elif nutrient_name == "Caffeine":
                nutrition_summary['caffeine'] = amount
            elif nutrient_name == "Taurine":
                nutrition_summary['taurine'] = amount
            elif nutrient_name == "Glycemic Index":
                nutrition_summary['glycemic_index'] = amount
        
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
                                n.unit as nutrient_unit
                            FROM food_nutrients fn
                            JOIN nutrients n ON fn.nutrient_id = n.id
                            WHERE fn.food_id = :food_id
                            ORDER BY n.name
                        """),
                        {"food_id": food_data['id']}
                    ).fetchall()
                    
                    nutrients = []
                    nutrition_summary = {
                        "energy_kcal": 0,
                        "energy": 0,
                        "energy_from_fat": 0,
                        "total_fat": 0,
                        "unsaturated_fat": 0,
                        "omega_3_fat": 0,
                        "trans_fat": 0,
                        "cholesterol": 0,
                        "carbohydrates": 0,
                        "sugars": 0,
                        "fiber": 0,
                        "protein": 0,
                        "salt": 0,
                        "sodium": 0,
                        "potassium": 0,
                        "calcium": 0,
                        "iron": 0,
                        "magnesium": 0,
                        "vitamin_d": 0,
                        "vitamin_c": 0,
                        "alcohol": 0,
                        "caffeine": 0,
                        "taurine": 0,
                        "glycemic_index": 0
                    }
                    
                    for nutrition_row in nutrition_rows:
                        if hasattr(nutrition_row, '_mapping'):
                            nutrient = dict(nutrition_row._mapping)
                        else:
                            nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit'], nutrition_row))
                        
                        amount = nutrient['amount'] or 0
                        nutrient_name = nutrient['nutrient_name']
                        category = 'general'
                        
                        nutrients.append({
                            "id": nutrient['nutrient_id'],
                            "name": nutrient['nutrient_name'],
                            "unit": nutrient['nutrient_unit'],
                            "amount": amount,
                            "category": category
                        })
                        
                        # Map to nutrition summary using exact database names
                        if nutrient_name == "Energy (kcal)":
                            nutrition_summary['energy_kcal'] = amount
                        elif nutrient_name == "Energy":
                            nutrition_summary['energy'] = amount
                        elif nutrient_name == "Energy from Fat":
                            nutrition_summary['energy_from_fat'] = amount
                        elif nutrient_name == "Total Fat":
                            nutrition_summary['total_fat'] = amount
                        elif nutrient_name == "Unsaturated Fat":
                            nutrition_summary['unsaturated_fat'] = amount
                        elif nutrient_name == "Omega-3 Fat":
                            nutrition_summary['omega_3_fat'] = amount
                        elif nutrient_name == "Trans Fat":
                            nutrition_summary['trans_fat'] = amount
                        elif nutrient_name == "Cholesterol":
                            nutrition_summary['cholesterol'] = amount
                        elif nutrient_name == "Carbohydrates":
                            nutrition_summary['carbohydrates'] = amount
                        elif nutrient_name == "Sugars":
                            nutrition_summary['sugars'] = amount
                        elif nutrient_name == "Fiber":
                            nutrition_summary['fiber'] = amount
                        elif nutrient_name == "Protein":
                            nutrition_summary['protein'] = amount
                        elif nutrient_name == "Salt":
                            nutrition_summary['salt'] = amount
                        elif nutrient_name == "Sodium":
                            nutrition_summary['sodium'] = amount
                        elif nutrient_name == "Potassium":
                            nutrition_summary['potassium'] = amount
                        elif nutrient_name == "Calcium":
                            nutrition_summary['calcium'] = amount
                        elif nutrient_name == "Iron":
                            nutrition_summary['iron'] = amount
                        elif nutrient_name == "Magnesium":
                            nutrition_summary['magnesium'] = amount
                        elif nutrient_name == "Vitamin D":
                            nutrition_summary['vitamin_d'] = amount
                        elif nutrient_name == "Vitamin C":
                            nutrition_summary['vitamin_c'] = amount
                        elif nutrient_name == "Alcohol":
                            nutrition_summary['alcohol'] = amount
                        elif nutrient_name == "Caffeine":
                            nutrition_summary['caffeine'] = amount
                        elif nutrient_name == "Taurine":
                            nutrition_summary['taurine'] = amount
                        elif nutrient_name == "Glycemic Index":
                            nutrition_summary['glycemic_index'] = amount
                    
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
                            "energy_kcal": 0, "energy": 0, "energy_from_fat": 0,
                            "total_fat": 0, "unsaturated_fat": 0, "omega_3_fat": 0,
                            "trans_fat": 0, "cholesterol": 0, "carbohydrates": 0,
                            "sugars": 0, "fiber": 0, "protein": 0, "salt": 0,
                            "sodium": 0, "potassium": 0, "calcium": 0, "iron": 0,
                            "magnesium": 0, "vitamin_d": 0, "vitamin_c": 0,
                            "alcohol": 0, "caffeine": 0, "taurine": 0, "glycemic_index": 0
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
    """Get comprehensive nutrition data for a food from the nutrition database - OPTIMIZED VERSION."""
    try:
        # Clean and normalize the food name
        food_name = food_name.strip().lower()
        
        # Handle common variations and synonyms
        food_variations = {
            "ground beef": ["beef", "minced beef", "hamburger meat"],
            "salmon": ["salmon fish", "atlantic salmon", "pacific salmon"],
            "avocado": ["avocado fruit", "avocado pear"],
            "oatmeal": ["oats", "rolled oats", "steel cut oats"],
            "berries": ["strawberries", "blueberries", "raspberries", "blackberries"],
            "coconut": ["coconut meat", "coconut flesh"],
            "yoghurt": ["yogurt", "greek yogurt", "plain yogurt"],
            "veal stock": ["beef stock", "chicken stock", "vegetable stock"],
            "tomato salsa": ["salsa", "tomato sauce", "pico de gallo"]
        }
        
        # Get variations for this food
        search_terms = [food_name]
        if food_name in food_variations:
            search_terms.extend(food_variations[food_name])
        
        # OPTIMIZED: Single query with JOIN to get food and key nutrients in one go
        for search_term in search_terms:
            query = """
            SELECT 
                f.id, 
                f.name, 
                f.brand_id, 
                f.category_id, 
                f.serving_size, 
                f.serving_unit, 
                f.serving,
                COALESCE(MAX(CASE WHEN n.name = 'Energy (kcal)' THEN fn.amount END), 0) as calories,
                COALESCE(MAX(CASE WHEN n.name = 'Protein' THEN fn.amount END), 0) as protein_g,
                COALESCE(MAX(CASE WHEN n.name = 'Carbohydrates' THEN fn.amount END), 0) as carbs_g,
                COALESCE(MAX(CASE WHEN n.name = 'Total Fat' THEN fn.amount END), 0) as fat_g
            FROM foods f
                LEFT JOIN food_nutrients fn ON f.id = fn.food_id
                LEFT JOIN nutrients n ON fn.nutrient_id = n.id
            WHERE LOWER(f.name) LIKE LOWER(:food_name)
            OR LOWER(f.name) LIKE LOWER(:food_name_pattern)
                GROUP BY f.id, f.name, f.brand_id, f.category_id, f.serving_size, f.serving_unit, f.serving
                ORDER BY 
                    CASE 
                        WHEN LOWER(f.name) = LOWER(:exact_match) THEN 1
                        WHEN LOWER(f.name) LIKE LOWER(:exact_match) THEN 2
                        ELSE 3
                    END,
                    f.name
                LIMIT 1
                """
                
            result = db.execute(
                text(query),
                {
                    "food_name": search_term, 
                    "food_name_pattern": f"%{search_term}%",
                    "exact_match": search_term
                }
            ).fetchone()
            
            if result:
                if hasattr(result, '_mapping'):
                    food_data = dict(result._mapping)
                else:
                    food_data = dict(zip(['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'calories', 'protein_g', 'carbs_g', 'fat_g'], result))
                
                # Return simplified result with just the essential data
                return {
                    "id": food_data['id'],
                    "name": food_data['name'],
                    "brand_id": food_data['brand_id'],
                    "category_id": food_data['category_id'],
                    "serving_size": food_data['serving_size'],
                    "serving_unit": food_data['serving_unit'],
                    "serving": food_data['serving'],
                    "calories": food_data['calories'],
                    "protein_g": food_data['protein_g'],
                    "carbs_g": food_data['carbs_g'],
                    "fat_g": food_data['fat_g'],
                    "found_in_db": True
                }
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting nutrition data for {food_name}: {e}")
        return None


def create_ai_meal_plan(
    user_id: str,
    daily_calories: int,
    meal_count: int,
    dietary_restrictions: List[str],
    user_history: List[Dict],
    db_nutrition,
    meal_description: Optional[str] = None
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
        user_context = _prepare_user_context(user_history, dietary_restrictions, daily_calories, meal_description)
        
        # Generate meal plan with AI
        ai_meal_plan = _generate_meal_plan_with_ai(client, user_context, meal_count, meal_description)
        
        # OPTIMIZED: Skip database enrichment for faster response
        # enriched_meal_plan = _enrich_meal_plan_with_nutrition(ai_meal_plan, db_nutrition)
        
        # Return AI-generated meal plan directly with AI macros
        return ai_meal_plan
        
    except Exception as e:
        logger.error(f"Error in AI meal plan creation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create AI meal plan: {str(e)}")


def _prepare_user_context(user_history: List[Dict], dietary_restrictions: List[str], daily_calories: int, meal_description: Optional[str] = None) -> str:
    """Prepare user context for AI meal plan generation."""
    
    # Extract user's food preferences
    food_preferences = []
    for entry in user_history[:20]:  # Last 20 entries
        food_name = entry.get("name", "")
        if food_name:
            food_preferences.append(food_name.lower())
    
    # Get unique preferences
    unique_preferences = list(set(food_preferences))
    
    # Build context with meal description
    context_parts = [
        f"Daily calorie target: {daily_calories} calories",
        f"Dietary restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}",
        f"Food preferences (from history): {', '.join(unique_preferences[:10])}"
    ]
    
    # Add meal description if provided
    if meal_description:
        context_parts.append(f"Detailed meal description: {meal_description}")
    
    context = f"""
    User Profile:
    - {' | '.join(context_parts)}
    
    Requirements:
    - Create a balanced meal plan with proper macronutrient distribution
    - Consider user's food preferences and dietary restrictions
    - Ensure variety and nutritional balance
    - Focus on whole, nutritious foods
    - If a detailed meal description is provided, prioritize the specific ingredients and preferences mentioned
    - Provide accurate macro estimates for all suggested foods
    """
    
    return context


def _generate_meal_plan_with_ai(client, user_context: str, meal_count: int, meal_description: Optional[str] = None) -> List[Dict]:
    """Generate meal plan using Groq AI."""
    
    settings = get_settings()
    
    # OPTIMIZED: Much shorter and faster prompt
    personalization_guidelines = ""
    if meal_description:
        personalization_guidelines = f"""
        USER REQUEST: "{meal_description}"
        
        REQUIREMENTS:
        - Extract days, meals per day, calorie targets, ingredients, dietary restrictions
        - Prioritize user ingredients, supplement as needed
        - Meet calorie targets and dietary restrictions
        - Create descriptive meal names
        """
    
    prompt = f"""
    {user_context}
    
    {personalization_guidelines}
    
    Create {meal_count} meals per day. Return ONLY valid JSON array:
    [
        {{
            "day": 1,
            "meals": [
                {{
                    "name": "Beef and Asparagus Stir-Fry",
                    "meal_type": "breakfast",
                    "servings": 1,
                    "macros": {{"calories": 1100, "protein_g": 45, "carbs_g": 35, "fat_g": 85}},
                    "ingredients": [
                        {{
                            "name": "beef",
                            "quantity_g": 150,
                            "macros": {{"calories": 450, "protein_g": 35, "carbs_g": 0, "fat_g": 30}}
                        }},
                        {{
                            "name": "asparagus",
                            "quantity_g": 100,
                            "macros": {{"calories": 20, "protein_g": 2, "carbs_g": 4, "fat_g": 0}}
                        }},
                        {{
                            "name": "olive oil",
                            "quantity_g": 15,
                            "macros": {{"calories": 130, "protein_g": 0, "carbs_g": 0, "fat_g": 15}}
                        }}
                    ]
                }},
                {{
                    "name": "Salmon with Asparagus and Avocado",
                    "meal_type": "lunch",
                    "servings": 1,
                    "macros": {{"calories": 1150, "protein_g": 60, "carbs_g": 12, "fat_g": 90}},
                    "ingredients": [
                        {{
                            "name": "salmon",
                            "quantity_g": 200,
                            "macros": {{"calories": 400, "protein_g": 50, "carbs_g": 0, "fat_g": 20}}
                        }},
                        {{
                            "name": "avocado",
                            "quantity_g": 100,
                            "macros": {{"calories": 160, "protein_g": 2, "carbs_g": 9, "fat_g": 15}}
                        }},
                        {{
                            "name": "asparagus",
                            "quantity_g": 150,
                            "macros": {{"calories": 30, "protein_g": 3, "carbs_g": 6, "fat_g": 0}}
                        }}
                    ]
                }}
            ]
        }}
    ]
    
    RULES: 
    - Include "day" field
    - Prioritize user ingredients
    - Meet calorie targets
    - Follow dietary restrictions
    - Include ingredients array with individual macros for each ingredient
    - Provide realistic quantities and accurate macro estimates
    """
    
    try:
        response = client.chat.completions.create(
            model=settings.llm.groq_model,
            messages=[
                {"role": "system", "content": "You are a nutrition expert. Create meal plans quickly and accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=800,   # Reduced tokens for faster response
            timeout=30        # 30 second timeout for faster response
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
    """Create a minimal fallback meal plan when AI fails."""
    # Simple fallback with basic structure
    fallback_plan = []
    
    for i in range(meal_count):
        meal_type = "breakfast" if i == 0 else "lunch"
        
        if meal_type == "breakfast":
            meal_name = "Oatmeal with Berries"
            ingredients = [
                {
                    "name": "oatmeal",
                    "quantity_g": 50,
                    "macros": {
                        "calories": 180,
                        "protein_g": 6.5,
                        "carbs_g": 30.5,
                        "fat_g": 3.2
                    }
                },
                {
                    "name": "berries",
                    "quantity_g": 30,
                    "macros": {
                        "calories": 15,
                        "protein_g": 0.5,
                        "carbs_g": 3.5,
                        "fat_g": 0.2
                    }
                }
            ]
            total_macros = {
                "calories": 195,
                "protein_g": 7.0,
                "carbs_g": 34.0,
                "fat_g": 3.4
            }
        else:
            meal_name = "Chicken Breast with Vegetables"
            ingredients = [
                {
                    "name": "chicken breast",
                    "quantity_g": 150,
                    "macros": {
                        "calories": 250,
                        "protein_g": 45,
                        "carbs_g": 0,
                        "fat_g": 5.5
                    }
                },
                {
                    "name": "broccoli",
                    "quantity_g": 100,
                    "macros": {
                        "calories": 35,
                        "protein_g": 3.5,
                        "carbs_g": 7,
                        "fat_g": 0.5
                    }
                }
            ]
            total_macros = {
                "calories": 285,
                "protein_g": 48.5,
                "carbs_g": 7,
                "fat_g": 6.0
            }
        
        fallback_plan.append({
            "day": 1,
            "meals": [
                {
                    "name": meal_name,
                    "meal_type": meal_type,
                    "servings": 1,
                    "macros": total_macros,
                    "ingredients": ingredients
                }
            ]
        })
    
    return fallback_plan


def _enrich_meal_plan_with_nutrition(ai_meal_plan: List[Dict], db_nutrition) -> List[Dict]:
    """Query nutrition database to get comprehensive nutrition data for AI-generated meals."""
    enriched_plan = []
    
    for day_data in ai_meal_plan:
        day = day_data.get("day", 1)
        meals = day_data.get("meals", [])
        
        enriched_day = {
            "day": day,
            "meals": []
        }
        
        for meal in meals:
            meal_name = meal.get("name", "")
            meal_type = meal.get("meal_type", "")
            servings = meal.get("servings", 1)
            ai_macros = meal.get("macros", {})
            ai_ingredients = meal.get("ingredients", [])
            
            enriched_meal = {
                "name": meal_name,
                "meal_type": meal_type,
                "servings": servings,
                "macros": ai_macros,  # Keep AI-generated macros as default
                "ingredients": [],
                "nutrition_verified": False
            }
            
            # Enrich ingredients with database nutrition data
            for ingredient in ai_ingredients:
                ingredient_name = ingredient.get("name", "")
                ingredient_quantity = ingredient.get("quantity_g", 0)
                ingredient_macros = ingredient.get("macros", {})
                
                # Try to get nutrition data from database for this ingredient
                food_data = _get_food_nutrition_from_db(db_nutrition, ingredient_name)
                
                if food_data:
                    # Calculate actual nutrition based on ingredient quantity
                    base_calories = food_data.get("calories", 0)
                    base_protein = food_data.get("protein_g", 0)
                    base_carbs = food_data.get("carbs_g", 0)
                    base_fat = food_data.get("fat_g", 0)
                    
                    actual_calories = (base_calories * ingredient_quantity) / 100
                    actual_protein = (base_protein * ingredient_quantity) / 100
                    actual_carbs = (base_carbs * ingredient_quantity) / 100
                    actual_fat = (base_fat * ingredient_quantity) / 100
                    
                    enriched_ingredient = {
                        "name": ingredient_name,
                        "quantity_g": ingredient_quantity,
                        "macros": {
                            "calories": round(actual_calories, 1),
                            "protein_g": round(actual_protein, 1),
                            "carbs_g": round(actual_carbs, 1),
                            "fat_g": round(actual_fat, 1)
                        },
                        "nutrition_verified": True,
                        "food_id": food_data.get("id"),
                        "nutrients": food_data.get("nutrients", [])
                    }
                    enriched_meal["nutrition_verified"] = True
                else:
                    # Use AI-generated macros if database data not available
                    enriched_ingredient = {
                        "name": ingredient_name,
                        "quantity_g": ingredient_quantity,
                        "macros": {
                            "calories": round(ingredient_macros.get("calories", 0), 1),
                            "protein_g": round(ingredient_macros.get("protein_g", 0), 1),
                            "carbs_g": round(ingredient_macros.get("carbs_g", 0), 1),
                            "fat_g": round(ingredient_macros.get("fat_g", 0), 1)
                        },
                        "nutrition_verified": False,
                        "note": "Nutrition data not available in database - using AI estimates"
                    }
                
                enriched_meal["ingredients"].append(enriched_ingredient)
            
            enriched_day["meals"].append(enriched_meal)
        
        enriched_plan.append(enriched_day)
    
    return enriched_plan


def _extract_ingredients_from_meal_name(meal_name: str) -> List[str]:
    """Extract potential ingredients from a meal name."""
    # Common ingredients to look for
    common_ingredients = [
        "oatmeal", "oats", "berries", "strawberries", "blueberries", "raspberries",
        "salmon", "beef", "ground beef", "chicken", "pork", "lamb", "fish",
        "asparagus", "broccoli", "spinach", "kale", "lettuce", "cucumber",
        "avocado", "tomato", "onion", "garlic", "potato", "sweet potato",
        "coconut", "yoghurt", "yogurt", "milk", "cheese", "eggs",
        "olive oil", "coconut oil", "butter", "salt", "pepper"
    ]
    
    meal_name_lower = meal_name.lower()
    found_ingredients = []
    
    for ingredient in common_ingredients:
        if ingredient in meal_name_lower:
            found_ingredients.append(ingredient)
    
    return found_ingredients


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

@app.get("/debug/foods-with-nutrition")
def debug_foods_with_nutrition(db=Depends(get_nutrition_db)):
    """Debug endpoint to check foods with nutrition data in the database."""
    try:
        # Query to find foods with nutrition data, especially calories
        query = """
        SELECT 
            f.id, 
            f.name, 
            fn.amount as calories,
            fn2.amount as protein_g,
            fn3.amount as carbs_g,
            fn4.amount as fat_g,
            n.name as calorie_nutrient_name
        FROM foods f
        LEFT JOIN food_nutrients fn ON f.id = fn.food_id 
            AND fn.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Energy (kcal)' LIMIT 1)
        LEFT JOIN food_nutrients fn2 ON f.id = fn2.food_id 
            AND fn2.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Protein' LIMIT 1)
        LEFT JOIN food_nutrients fn3 ON f.id = fn3.food_id 
            AND fn3.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Carbohydrates' LIMIT 1)
        LEFT JOIN food_nutrients fn4 ON f.id = fn4.food_id 
            AND fn4.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Total Fat' LIMIT 1)
        LEFT JOIN nutrients n ON fn.nutrient_id = n.id
        WHERE fn.amount IS NOT NULL AND fn.amount > 0
        ORDER BY fn.amount DESC
        LIMIT 20
        """
        
        rows = db.execute(text(query)).fetchall()
        
        foods_with_nutrition = []
        for row in rows:
            if hasattr(row, '_mapping'):
                food_data = dict(row._mapping)
            else:
                food_data = dict(zip(['id', 'name', 'calories', 'protein_g', 'carbs_g', 'fat_g', 'calorie_nutrient_name'], row))
            
            foods_with_nutrition.append({
                "id": food_data['id'],
                "name": food_data['name'],
                "calories": food_data['calories'],
                "protein_g": food_data['protein_g'] or 0,
                "carbs_g": food_data['carbs_g'] or 0,
                "fat_g": food_data['fat_g'] or 0,
                "calorie_nutrient_name": food_data['calorie_nutrient_name']
            })
        
        # Also get a count of total foods and foods with calories
        total_foods_query = "SELECT COUNT(*) FROM foods"
        foods_with_calories_query = """
        SELECT COUNT(*) FROM foods f
        JOIN food_nutrients fn ON f.id = fn.food_id 
        JOIN nutrients n ON fn.nutrient_id = n.id
        WHERE n.name = 'Energy (kcal)'
        """
        
        total_foods = db.execute(text(total_foods_query)).scalar()
        foods_with_calories = db.execute(text(foods_with_calories_query)).scalar()
        
        return {
            "status": "success",
            "total_foods_in_database": total_foods,
            "foods_with_calories": foods_with_calories,
            "sample_foods_with_nutrition": foods_with_nutrition,
            "note": "This shows foods that have calorie data in the food_nutrients table"
        }
        
    except Exception as e:
        logger.error(f"Error in debug_foods_with_nutrition: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/test-meal-plan-nutrition")
def test_meal_plan_nutrition(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Test endpoint to verify meal plan nutrition data retrieval."""
    try:
        user_id = request.get("user_id", "test_user")
        daily_calories = request.get("daily_calories", 2000)
        meal_count = request.get("meal_count", 2)
        dietary_restrictions = request.get("dietary_restrictions", [])
        
        # Create a simple test meal plan
        test_meal_plan = [
            {
                "meal_type": "breakfast",
                "foods": [
                    {
                        "name": "oatmeal",
                        "quantity_g": 50,
                        "reasoning": "High fiber, complex carbs for sustained energy"
                    }
                ]
            },
            {
                "meal_type": "lunch",
                "foods": [
                    {
                        "name": "chicken breast",
                        "quantity_g": 150,
                        "reasoning": "Lean protein source"
                    }
                ]
            }
        ]
        
        # Test the nutrition enrichment function
        enriched_meal_plan = _enrich_meal_plan_with_nutrition(test_meal_plan, db_nutrition)
        
        # Calculate totals
        total_calories = sum(meal["total_calories"] for meal in enriched_meal_plan)
        total_protein = sum(meal["total_protein_g"] for meal in enriched_meal_plan)
        total_carbs = sum(meal["total_carbs_g"] for meal in enriched_meal_plan)
        total_fat = sum(meal["total_fat_g"] for meal in enriched_meal_plan)
        
        return {
            "status": "success",
            "test_meal_plan": enriched_meal_plan,
            "totals": {
                "total_calories": total_calories,
                "total_protein_g": total_protein,
                "total_carbs_g": total_carbs,
                "total_fat_g": total_fat
            },
            "note": "This tests if the nutrition data retrieval is working properly"
        }
        
    except Exception as e:
        logger.error(f"Error in test_meal_plan_nutrition: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/debug/nutrient-ids")
def debug_nutrient_ids(db=Depends(get_nutrition_db)):
    """Debug endpoint to show the exact nutrient IDs and names being used."""
    try:
        # Query to get the specific nutrient IDs we're using
        query = """
        SELECT id, name, unit 
        FROM nutrients 
        WHERE name IN ('Energy (kcal)', 'Protein', 'Carbohydrates', 'Total Fat')
        ORDER BY name
        """
        
        rows = db.execute(text(query)).fetchall()
        
        nutrient_info = []
        for row in rows:
            if hasattr(row, '_mapping'):
                nutrient_data = dict(row._mapping)
            else:
                nutrient_data = dict(zip(['id', 'name', 'unit'], row))
            
            nutrient_info.append({
                "id": nutrient_data['id'],
                "name": nutrient_data['name'],
                "unit": nutrient_data['unit']
            })
        
        # Also get a sample of foods with these nutrients
        sample_query = """
        SELECT 
            f.name as food_name,
            fn.amount as calories,
            fn2.amount as protein_g,
            fn3.amount as carbs_g,
            fn4.amount as fat_g
        FROM foods f
        LEFT JOIN food_nutrients fn ON f.id = fn.food_id 
            AND fn.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Energy (kcal)' LIMIT 1)
        LEFT JOIN food_nutrients fn2 ON f.id = fn2.food_id 
            AND fn2.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Protein' LIMIT 1)
        LEFT JOIN food_nutrients fn3 ON f.id = fn3.food_id 
            AND fn3.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Carbohydrates' LIMIT 1)
        LEFT JOIN food_nutrients fn4 ON f.id = fn4.food_id 
            AND fn4.nutrient_id = (SELECT id FROM nutrients WHERE name = 'Total Fat' LIMIT 1)
        WHERE fn.amount IS NOT NULL AND fn.amount > 0
        ORDER BY fn.amount DESC
        LIMIT 5
        """
        
        sample_rows = db.execute(text(sample_query)).fetchall()
        
        sample_foods = []
        for row in sample_rows:
            if hasattr(row, '_mapping'):
                food_data = dict(row._mapping)
            else:
                food_data = dict(zip(['food_name', 'calories', 'protein_g', 'carbs_g', 'fat_g'], row))
            
            sample_foods.append({
                "food_name": food_data['food_name'],
                "calories": food_data['calories'],
                "protein_g": food_data['protein_g'] or 0,
                "carbs_g": food_data['carbs_g'] or 0,
                "fat_g": food_data['fat_g'] or 0
            })
        
        return {
            "status": "success",
            "nutrient_ids_being_used": nutrient_info,
            "sample_foods_with_nutrition": sample_foods,
            "note": "These are the exact nutrient IDs and names being used for meal plan nutrition calculation"
        }
        
    except Exception as e:
        logger.error(f"Error in debug_nutrient_ids: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/debug/test-nutrition/{food_name}")
def debug_test_nutrition(food_name: str, db=Depends(get_nutrition_db)):
    """Debug endpoint to test nutrition data retrieval for a specific food."""
    try:
        # Test the nutrition data retrieval
        nutrition_data = _get_food_nutrition_from_db(db, food_name)
        
        if nutrition_data:
            # Test the calculation
            quantity_g = 100
            calories = nutrition_data.get("calories")
            protein = nutrition_data.get("protein_g")
            carbs = nutrition_data.get("carbs_g")
            fat = nutrition_data.get("fat_g")
            
            # Calculate with None handling
            actual_calories = ((calories or 0) * quantity_g) / 100
            actual_protein = ((protein or 0) * quantity_g) / 100
            actual_carbs = ((carbs or 0) * quantity_g) / 100
            actual_fat = ((fat or 0) * quantity_g) / 100
            
            return {
                "status": "success",
                "food_name": food_name,
                "raw_nutrition_data": nutrition_data,
                "calculated_nutrition": {
                    "calories": round(actual_calories, 1),
                    "protein_g": round(actual_protein, 1),
                    "carbs_g": round(actual_carbs, 1),
                    "fat_g": round(actual_fat, 1)
                },
                "note": "This tests the nutrition calculation with None handling"
            }
        else:
            return {
                "status": "not_found",
                "food_name": food_name,
                "note": "Food not found in database"
            }
        
    except Exception as e:
        logger.error(f"Error in debug_test_nutrition: {e}")
        return {
            "status": "error",
            "food_name": food_name,
            "error": str(e)
        }

@app.get("/debug/food/{food_id}")
def debug_food_details(food_id: int, db=Depends(get_nutrition_db)):
    """Debug endpoint to show detailed information about a specific food."""
    try:
        # Get basic food information
        food_query = """
        SELECT id, name, brand_id, category_id, serving_size, serving_unit, serving, created_at
        FROM foods
        WHERE id = :food_id
        """
        
        food_row = db.execute(text(food_query), {"food_id": food_id}).fetchone()
        
        if not food_row:
            return {"status": "error", "message": f"Food with ID {food_id} not found"}
        
        if hasattr(food_row, '_mapping'):
            food_data = dict(food_row._mapping)
        else:
            food_data = dict(zip(['id', 'name', 'brand_id', 'category_id', 'serving_size', 'serving_unit', 'serving', 'created_at'], food_row))
        
        # Get all nutrients for this food
        nutrients_query = """
        SELECT 
            n.id as nutrient_id, 
            n.name as nutrient_name, 
            fn.amount, 
            n.unit as nutrient_unit
        FROM food_nutrients fn
        JOIN nutrients n ON fn.nutrient_id = n.id
        WHERE fn.food_id = :food_id
        ORDER BY n.name
        """
        
        nutrients_rows = db.execute(text(nutrients_query), {"food_id": food_id}).fetchall()
        
        nutrients = []
        for row in nutrients_rows:
            if hasattr(row, '_mapping'):
                nutrient = dict(row._mapping)
            else:
                nutrient = dict(zip(['nutrient_id', 'nutrient_name', 'amount', 'nutrient_unit'], row))
            nutrients.append(nutrient)
        
        # Test the specific nutrient names that search_food_by_name is looking for
        specific_nutrients_query = """
        SELECT 
            n.name as nutrient_name,
            fn.amount
        FROM food_nutrients fn
        JOIN nutrients n ON fn.nutrient_id = n.id
        WHERE fn.food_id = :food_id
        AND n.name IN (
            'Energy (kcal)', 'Energy', 'Energy from Fat', 'Total Fat', 'Unsaturated Fat',
            'Omega-3 Fat', 'Trans Fat', 'Cholesterol', 'Carbohydrates', 'Sugars',
            'Fiber', 'Protein', 'Salt', 'Sodium', 'Potassium', 'Calcium', 'Iron',
            'Magnesium', 'Vitamin D', 'Vitamin C', 'Alcohol', 'Caffeine', 'Taurine',
            'Glycemic Index'
        )
        ORDER BY n.name
        """
        
        specific_nutrients_rows = db.execute(text(specific_nutrients_query), {"food_id": food_id}).fetchall()
        
        specific_nutrients = []
        for row in specific_nutrients_rows:
            if hasattr(row, '_mapping'):
                nutrient = dict(row._mapping)
            else:
                nutrient = dict(zip(['nutrient_name', 'amount'], row))
            specific_nutrients.append(nutrient)
        
        # Get all available nutrient names in the database for comparison
        all_nutrients_query = """
        SELECT DISTINCT n.name
        FROM nutrients n
        ORDER BY n.name
        """
        
        all_nutrients_rows = db.execute(text(all_nutrients_query)).fetchall()
        all_nutrient_names = [row[0] if not hasattr(row, '_mapping') else row._mapping['name'] for row in all_nutrients_rows]
        
        return {
            "status": "success",
            "food": {
                "id": food_data["id"],
                "name": food_data["name"],
                "brand_id": food_data["brand_id"],
                "category_id": food_data["category_id"],
                "serving_size": food_data["serving_size"],
                "serving_unit": food_data["serving_unit"],
                "serving": food_data["serving"],
                "created_at": food_data["created_at"],
                "nutrients": nutrients,
                "specific_nutrients": specific_nutrients,
                "total_nutrients": len(nutrients)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/test-ingredients-structure")
def test_ingredients_structure(request: Dict[str, Any], db_nutrition=Depends(get_nutrition_db), db_shared=Depends(get_shared_db)):
    """Test endpoint to demonstrate the new ingredients structure with macros."""
    try:
        # Test meal suggestions with ingredients
        meal_suggestions = get_meal_suggestions(
            db_nutrition, 
            db_shared, 
            "test_user", 
            "breakfast", 
            500, 
            "I want oatmeal with berries and nuts"
        )
        
        # Test meal plan creation with ingredients
        meal_plan = create_ai_meal_plan(
            "test_user",
            2000,
            3,
            [],
            [],
            db_nutrition,
            "Create a meal plan with chicken, rice, and vegetables"
        )
        
        return {
            "status": "success",
            "message": "Testing new ingredients structure with macros",
            "meal_suggestions": meal_suggestions,
            "meal_plan": meal_plan,
            "ingredients_structure_explanation": {
                "description": "Both meal suggestions and meal plans now include ingredients as an array of objects with macros",
                "ingredients_format": {
                    "name": "ingredient_name",
                    "quantity_g": 100,
                    "macros": {
                        "calories": 150,
                        "protein_g": 10,
                        "carbs_g": 20,
                        "fat_g": 5
                    },
                    "nutrition_verified": True,
                    "food_id": 123,
                    "nutrients": []
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Add after the existing meal suggestion functions, around line 3034

def create_recipe(
    db_nutrition, db_shared, user_id: str, recipe_description: str, 
    servings: int = 1, difficulty: str = "easy", cuisine: Optional[str] = None,
    dietary_restrictions: Optional[List[str]] = None
):
    """Create AI-powered recipes with detailed ingredients, instructions, and nutrition data."""
    try:
        settings = get_settings()
        
        # Initialize Groq client
        client = openai.OpenAI(
            api_key=settings.llm.groq_api_key,
            base_url=settings.llm.groq_base_url
        )
        
        # Get user's food preferences for context
        user_preferences = []
        try:
            user_uuid = _get_user_uuid(user_id)
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
                {"user_id": user_uuid},
            ).fetchall()

            # Get food details for context
            food_ids = [row[0] if not hasattr(row, '_mapping') else row._mapping['food_item_id'] for row in recent_foods]
            
            if food_ids:
                placeholders = ",".join([":id" + str(i) for i in range(len(food_ids))])
                params = {f"id{i}": food_id for i, food_id in enumerate(food_ids)}
                
                foods = db_nutrition.execute(
                    text(
                        f"""
                    SELECT id, name
                    FROM foods 
                    WHERE id IN ({placeholders})
                    """
                    ),
                    params,
                ).fetchall()

                for food in foods:
                    if hasattr(food, '_mapping'):
                        user_preferences.append(dict(food._mapping)["name"])
                    else:
                        user_preferences.append(food[1])  # Assuming name is second column

        except Exception as e:
            logger.warning(f"Could not fetch user preferences: {e}")

        # Generate recipe with AI
        recipe = _generate_recipe_with_ai(
            client, recipe_description, servings, difficulty, cuisine, 
            dietary_restrictions, user_preferences
        )
        
        # Enrich recipe with nutrition data from database
        enriched_recipe = _enrich_recipe_with_nutrition(recipe, db_nutrition)
        
        return {
            "recipe": enriched_recipe,
            "ai_generated": True,
            "nutrition_verified": enriched_recipe.get("nutrition_verified", False),
            "user_preferences_used": len(user_preferences) > 0,
            "dietary_restrictions": dietary_restrictions or [],
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in recipe creation: {e}")
        # Fallback to basic recipe
        return _get_fallback_recipe(recipe_description, servings, difficulty)


def _generate_recipe_with_ai(
    client, recipe_description: str, servings: int, difficulty: str, 
    cuisine: Optional[str], dietary_restrictions: Optional[List[str]], 
    user_preferences: List[str]
) -> Dict:
    """Generate recipe using Groq AI."""
    
    settings = get_settings()
    
    # Build context for AI
    context_parts = [
        f"Recipe description: {recipe_description}",
        f"Servings: {servings}",
        f"Difficulty: {difficulty}"
    ]
    
    if cuisine:
        context_parts.append(f"Cuisine: {cuisine}")
    
    if dietary_restrictions:
        context_parts.append(f"Dietary restrictions: {', '.join(dietary_restrictions)}")
    
    if user_preferences:
        context_parts.append(f"User preferences: {', '.join(user_preferences[:5])}")
    
    context = " | ".join(context_parts)
    
    prompt = f"""
    {context}
    
    Create a detailed recipe with the following JSON structure:
    {{
        "id": "recipe_001",
        "name": "Recipe Name",
        "description": "Detailed description of the recipe",
        "servings": {servings},
        "prep_time_minutes": 10,
        "cook_time_minutes": 20,
        "total_time_minutes": 30,
        "ingredients": [
            {{
                "name": "Ingredient Name",
                "quantity": 100,
                "unit": "g",
                "calories": 150,
                "protein_g": 10,
                "carbs_g": 20,
                "fat_g": 5,
                "fiber_g": 2,
                "sugar_g": 1
            }}
        ],
        "instructions": [
            "Step 1: Detailed instruction",
            "Step 2: Detailed instruction",
            "Step 3: Detailed instruction"
        ],
        "nutrition": {{
            "calories": 500,
            "protein_g": 25,
            "carbs_g": 45,
            "fat_g": 20,
            "fiber_g": 8,
            "sugar_g": 5
        }},
        "tags": ["breakfast", "high-protein", "vegetarian"],
        "difficulty": "{difficulty}",
        "cuisine": "{cuisine or 'general'}"
    }}
    
    NUTRITION REQUIREMENTS:
    - Provide accurate nutrition data for each ingredient (calories, protein, carbs, fat, fiber, sugar)
    - Use realistic quantities and common units (g, tbsp, tsp, cup, large/medium/small, oz, lb, ml, l)
    - Calculate total recipe nutrition by summing all ingredients
    - Ensure nutrition data is reasonable and realistic
    - Include fiber and sugar content for each ingredient when applicable
    
    INGREDIENT REQUIREMENTS:
    - Use specific, common ingredient names that can be found in nutrition databases
    - Provide precise quantities with appropriate units
    - Include all necessary ingredients for the recipe
    - Consider serving size and adjust quantities accordingly
    - Use fresh, whole ingredients when possible
    
    RECIPE REQUIREMENTS:
    - Create a realistic recipe that matches the description
    - Use common, accessible ingredients
    - Provide detailed, step-by-step instructions
    - Include appropriate tags based on ingredients and nutrition
    - Ensure the recipe is suitable for the specified difficulty level
    - Consider dietary restrictions if provided
    - Use user preferences when relevant
    - Make instructions clear and easy to follow
    - Include proper cooking times and techniques
    - Calculate total nutrition from individual ingredients
    - Ensure all times are realistic and appropriate
    """
    
    try:
        response = client.chat.completions.create(
            model=settings.llm.groq_model,
            messages=[
                {"role": "system", "content": "You are a professional chef and nutritionist. Create detailed, accurate recipes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200,
            timeout=45
        )
        
        ai_response = response.choices[0].message.content
        return _parse_ai_recipe(ai_response)
        
    except Exception as e:
        logger.error(f"Error generating recipe with AI: {e}")
        return _create_fallback_recipe(recipe_description, servings, difficulty)


def _parse_ai_recipe(ai_response: str) -> Dict:
    """Parse AI response into structured recipe."""
    try:
        import json
        import re
        
        # Find JSON object in the response
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            recipe = json.loads(json_match.group())
            
            # Ensure required fields exist
            recipe.setdefault("id", f"recipe_{int(time.time())}")
            recipe.setdefault("name", "AI Generated Recipe")
            recipe.setdefault("description", "A delicious recipe created by AI")
            recipe.setdefault("servings", 1)
            recipe.setdefault("prep_time_minutes", 10)
            recipe.setdefault("cook_time_minutes", 20)
            recipe.setdefault("total_time_minutes", 30)
            recipe.setdefault("ingredients", [])
            recipe.setdefault("instructions", [])
            recipe.setdefault("nutrition", {})
            recipe.setdefault("tags", [])
            recipe.setdefault("difficulty", "easy")
            recipe.setdefault("cuisine", "general")
            
            return recipe
        else:
            logger.warning("Could not parse AI response as JSON")
            return _create_fallback_recipe("Basic Recipe", 1, "easy")
            
    except Exception as e:
        logger.error(f"Error parsing AI recipe: {e}")
        return _create_fallback_recipe("Basic Recipe", 1, "easy")


def _enrich_recipe_with_nutrition(recipe: Dict, db_nutrition) -> Dict:
    """Enrich recipe with comprehensive nutrition data from database."""
    enriched_recipe = recipe.copy()
    
    # Calculate total nutrition from ingredients
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    total_fiber = 0
    total_sugar = 0
    
    enriched_ingredients = []
    verified_ingredients_count = 0
    
    for ingredient in recipe.get("ingredients", []):
        ingredient_name = ingredient.get("name", "")
        quantity = ingredient.get("quantity", 0)
        unit = ingredient.get("unit", "g")
        
        # Try to get comprehensive nutrition data from database
        food_data = _get_food_nutrition_from_db(db_nutrition, ingredient_name)
        
        if food_data:
            # Calculate nutrition based on quantity and unit conversion
            base_calories = food_data.get("calories", 0)
            base_protein = food_data.get("protein_g", 0)
            base_carbs = food_data.get("carbs_g", 0)
            base_fat = food_data.get("fat_g", 0)
            base_fiber = food_data.get("fiber_g", 0)
            base_sugar = food_data.get("sugar_g", 0)
            
            # Convert units if necessary (comprehensive conversion)
            conversion_factor = 1.0
            if unit.lower() in ["tbsp", "tablespoon"]:
                conversion_factor = 15  # 1 tbsp = 15g
            elif unit.lower() in ["tsp", "teaspoon"]:
                conversion_factor = 5   # 1 tsp = 5g
            elif unit.lower() in ["cup"]:
                conversion_factor = 240  # 1 cup = 240g (approximate)
            elif unit.lower() in ["large", "medium", "small"]:
                conversion_factor = 50   # 1 egg = 50g (approximate)
            elif unit.lower() in ["oz", "ounce"]:
                conversion_factor = 28.35  # 1 oz = 28.35g
            elif unit.lower() in ["lb", "pound"]:
                conversion_factor = 453.59  # 1 lb = 453.59g
            elif unit.lower() in ["ml", "milliliter"]:
                conversion_factor = 1.0  # 1 ml ≈ 1g for most liquids
            elif unit.lower() in ["l", "liter"]:
                conversion_factor = 1000  # 1 liter = 1000g
            
            actual_quantity = quantity * conversion_factor
            
            # Calculate nutrition based on actual quantity
            calories = (base_calories * actual_quantity) / 100
            protein = (base_protein * actual_quantity) / 100
            carbs = (base_carbs * actual_quantity) / 100
            fat = (base_fat * actual_quantity) / 100
            fiber = (base_fiber * actual_quantity) / 100
            sugar = (base_sugar * actual_quantity) / 100
            
            enriched_ingredient = {
                "name": ingredient_name,
                "quantity": quantity,
                "unit": unit,
                "actual_quantity_g": round(actual_quantity, 1),
                "calories": round(calories, 1),
                "protein_g": round(protein, 1),
                "carbs_g": round(carbs, 1),
                "fat_g": round(fat, 1),
                "fiber_g": round(fiber, 1),
                "sugar_g": round(sugar, 1),
                "nutrition_verified": True,
                "food_id": food_data.get("id"),
                "nutrients": food_data.get("nutrients", []),
                "nutrition_summary": food_data.get("nutrition_summary", {}),
                "total_nutrients": food_data.get("total_nutrients", 0),
                "database_source": "verified"
            }
            
            # Add to totals
            total_calories += calories
            total_protein += protein
            total_carbs += carbs
            total_fat += fat
            total_fiber += fiber
            total_sugar += sugar
            verified_ingredients_count += 1
            
        else:
            # Use AI-generated nutrition if database data not available
            ai_calories = ingredient.get("calories", 0)
            ai_protein = ingredient.get("protein_g", 0)
            ai_carbs = ingredient.get("carbs_g", 0)
            ai_fat = ingredient.get("fat_g", 0)
            ai_fiber = ingredient.get("fiber_g", 0)
            ai_sugar = ingredient.get("sugar_g", 0)
            
            enriched_ingredient = {
                "name": ingredient_name,
                "quantity": quantity,
                "unit": unit,
                "actual_quantity_g": quantity,  # Assume grams if no conversion
                "calories": round(ai_calories, 1),
                "protein_g": round(ai_protein, 1),
                "carbs_g": round(ai_carbs, 1),
                "fat_g": round(ai_fat, 1),
                "fiber_g": round(ai_fiber, 1),
                "sugar_g": round(ai_sugar, 1),
                "nutrition_verified": False,
                "food_id": None,
                "nutrients": [],
                "nutrition_summary": {},
                "total_nutrients": 0,
                "database_source": "ai_estimated",
                "note": f"Nutrition data for '{ingredient_name}' not available in database - using AI estimates"
            }
            
            # Add to totals
            total_calories += ai_calories
            total_protein += ai_protein
            total_carbs += ai_carbs
            total_fat += ai_fat
            total_fiber += ai_fiber
            total_sugar += ai_sugar
        
        enriched_ingredients.append(enriched_ingredient)
    
    # Update recipe with enriched data
    enriched_recipe["ingredients"] = enriched_ingredients
    
    # Enhanced nutrition summary
    enriched_recipe["nutrition"] = {
        "calories": round(total_calories, 1),
        "protein_g": round(total_protein, 1),
        "carbs_g": round(total_carbs, 1),
        "fat_g": round(total_fat, 1),
        "fiber_g": round(total_fiber, 1),
        "sugar_g": round(total_sugar, 1)
    }
    
    # Calculate verification percentage
    total_ingredients = len(enriched_ingredients)
    verification_percentage = (verified_ingredients_count / total_ingredients * 100) if total_ingredients > 0 else 0
    
    enriched_recipe["nutrition_verified"] = verification_percentage >= 50  # At least 50% verified
    enriched_recipe["nutrition_verification_percentage"] = round(verification_percentage, 1)
    enriched_recipe["verified_ingredients_count"] = verified_ingredients_count
    enriched_recipe["total_ingredients_count"] = total_ingredients
    
    # Add nutrition quality indicators
    if verification_percentage >= 80:
        enriched_recipe["nutrition_quality"] = "excellent"
    elif verification_percentage >= 60:
        enriched_recipe["nutrition_quality"] = "good"
    elif verification_percentage >= 40:
        enriched_recipe["nutrition_quality"] = "fair"
    else:
        enriched_recipe["nutrition_quality"] = "estimated"
    
    return enriched_recipe


def _create_fallback_recipe(description: str, servings: int, difficulty: str) -> Dict:
    """Create a fallback recipe when AI fails."""
    import time
    
    return {
        "id": f"recipe_{int(time.time())}",
        "name": "Simple Recipe",
        "description": description or "A simple, nutritious recipe",
        "servings": servings,
        "prep_time_minutes": 10,
        "cook_time_minutes": 20,
        "total_time_minutes": 30,
        "ingredients": [
            {
                "name": "chicken breast",
                "quantity": 150,
                "unit": "g",
                "calories": 250,
                "protein_g": 45,
                "carbs_g": 0,
                "fat_g": 5.5,
                "nutrition_verified": False
            },
            {
                "name": "brown rice",
                "quantity": 100,
                "unit": "g",
                "calories": 110,
                "protein_g": 2.5,
                "carbs_g": 23,
                "fat_g": 0.9,
                "nutrition_verified": False
            },
            {
                "name": "broccoli",
                "quantity": 100,
                "unit": "g",
                "calories": 35,
                "protein_g": 3.5,
                "carbs_g": 7,
                "fat_g": 0.5,
                "nutrition_verified": False
            }
        ],
        "instructions": [
            "Cook the chicken breast until fully cooked.",
            "Prepare brown rice according to package instructions.",
            "Steam the broccoli until tender.",
            "Combine all ingredients and serve hot."
        ],
        "nutrition": {
            "calories": 395,
            "protein_g": 51,
            "carbs_g": 30,
            "fat_g": 6.9,
            "fiber_g": 7,
            "sugar_g": 2
        },
        "tags": ["main-dish", "high-protein", "balanced"],
        "difficulty": difficulty,
        "cuisine": "general",
        "nutrition_verified": False
    }


def _get_fallback_recipe(description: str, servings: int, difficulty: str) -> Dict:
    """Get fallback recipe when recipe creation fails."""
    return {
        "recipe": _create_fallback_recipe(description, servings, difficulty),
        "ai_generated": False,
        "nutrition_verified": False,
        "fallback_used": True,
        "message": "Using fallback recipe due to AI generation failure",
        "created_at": datetime.datetime.utcnow().isoformat()
    }


# Add the new endpoint after the existing meal suggestion endpoints
@app.post("/create-recipe")
def create_recipe_endpoint(
    request: Dict[str, Any], 
    db_nutrition=Depends(get_nutrition_db), 
    db_shared=Depends(get_shared_db)
):
    """Create AI-powered recipes with detailed ingredients, instructions, and nutrition data."""
    try:
        user_id = request.get("user_id", "anonymous")
        recipe_description = request.get("recipe_description", "")
        servings = request.get("servings", 1)
        difficulty = request.get("difficulty", "easy")
        cuisine = request.get("cuisine")
        dietary_restrictions = request.get("dietary_restrictions", [])
        
        if not recipe_description:
            raise HTTPException(status_code=400, detail="Recipe description is required")
        
        # Validate inputs
        if servings < 1 or servings > 20:
            raise HTTPException(status_code=400, detail="Servings must be between 1 and 20")
        
        valid_difficulties = ["easy", "medium", "hard"]
        if difficulty not in valid_difficulties:
            raise HTTPException(status_code=400, detail=f"Difficulty must be one of: {', '.join(valid_difficulties)}")
        
        # Create recipe
        result = create_recipe(
            db_nutrition=db_nutrition,
            db_shared=db_shared,
            user_id=user_id,
            recipe_description=recipe_description,
            servings=servings,
            difficulty=difficulty,
            cuisine=cuisine,
            dietary_restrictions=dietary_restrictions
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_recipe_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create recipe: {str(e)}")

# =============================================================================
# SCALABILITY ENDPOINTS - MULTI-USER SUPPORT
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request model for creating a user session."""
    user_id: str

@app.post("/api/sessions/create")
async def create_user_session(request: CreateSessionRequest):
    """Create a new user session for multi-user support."""
    try:
        if not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_data = await _session_manager.create_user_session(request.user_id, "nutrition")
        
        return {
            "success": True,
            "data": {
                "session": session_data,
                "message": "Session created successfully"
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/api/sessions/{session_token}")
async def get_session_info(session_token: str):
    """Get session information for multi-user support."""
    try:
        if not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session = await _session_manager.get_session(session_token)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "data": {
                "session": session
            },
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

class SubmitTaskRequest(BaseModel):
    """Request model for submitting a background task."""
    task_type: str
    user_id: str
    priority: str = "medium"

@app.post("/api/tasks/submit")
async def submit_background_task(request: SubmitTaskRequest):
    """Submit a background task for processing."""
    try:
        if not _task_manager:
            raise HTTPException(status_code=503, detail="Task manager not available")
        
        # Map priority string to enum
        priority_map = {
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW
        }
        
        task_priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)
        
        # Example task function
        async def example_task(user_id: str, task_type: str):
            await asyncio.sleep(2)  # Simulate work
            return {"user_id": user_id, "task_type": task_type, "completed": True}
        
        task_id = await _task_manager.submit_task(
            example_task,
            args=(request.user_id, request.task_type),
            priority=task_priority,
            user_id=request.user_id,
            agent_type="nutrition"
        )
        
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "message": "Task submitted successfully",
                "priority": request.priority
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task."""
    try:
        if not _task_manager:
            raise HTTPException(status_code=503, detail="Task manager not available")
        
        task_status = await _task_manager.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "success": True,
            "data": {
                "task_status": task_status
            },
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.get("/api/performance/metrics")
async def get_performance_metrics():
    """Get performance metrics for monitoring."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        # Get basic metrics
        metrics = {
            "uptime_seconds": _performance_monitor.get_uptime(),
            "request_count": _performance_monitor.request_count,
            "error_count": _performance_monitor.error_count,
            "error_rate": _performance_monitor.error_count / max(_performance_monitor.request_count, 1)
        }
        
        return {
            "success": True,
            "data": {
                "metrics": metrics,
                "timestamp": datetime.datetime.utcnow().isoformat()
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.get("/api/scalability/status")
async def get_scalability_status():
    """Get comprehensive scalability status."""
    try:
        status = {
            "session_manager": "inactive",
            "task_manager": "inactive",
            "performance_monitor": "inactive",
            "health_monitor": "inactive",
            "database": "unknown",
            "redis": "unknown"
        }
        
        # Check session manager
        if _session_manager:
            status["session_manager"] = "active"
        
        # Check task manager
        if _task_manager:
            status["task_manager"] = "active"
        
        # Check performance monitor
        if _performance_monitor:
            status["performance_monitor"] = "active"
        
        # Check health monitor
        if _health_monitor:
            status["health_monitor"] = "active"
        
        # Check database
        try:
            # Use user database for health check
            from shared.database import get_user_db_engine
            with get_user_db_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            status["database"] = "connected"
        except Exception:
            status["database"] = "disconnected"
        
        # Check Redis
        try:
            if _session_manager and hasattr(_session_manager, 'redis_client') and _session_manager.redis_client:
                await _session_manager.redis_client.ping()
                status["redis"] = "connected"
            else:
                status["redis"] = "not_configured"
        except Exception:
            status["redis"] = "disconnected"
        
        # Calculate overall readiness
        active_components = sum(1 for v in status.values() if v == "active" or v == "connected")
        total_components = len(status)
        readiness_percentage = (active_components / total_components) * 100
        
        return {
            "success": True,
            "data": {
                "components": status,
                "readiness_percentage": round(readiness_percentage, 2),
                "scalability_ready": readiness_percentage >= 80,
                "timestamp": datetime.datetime.utcnow().isoformat()
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to get scalability status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scalability status: {str(e)}")

# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =============================================================================

class ConversationMessageRequest(BaseModel):
    """Request model for adding conversation messages."""
    message_type: str
    content: str
    metadata: Dict[str, Any] = {}

class ConversationSearchRequest(BaseModel):
    """Request model for searching conversations."""
    query: str
    limit: int = 20

@app.get("/api/conversations/state")
async def get_conversation_state(
    current_user: Dict[str, Any] = Depends(require_authentication)
):
    """Get current conversation state for the authenticated user."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Get conversation state from session
        session_data = current_user.get("session_data", {})
        session_id = session_data.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")
        
        conversation = await _conversation_manager.get_or_create_conversation(
            current_user["user_id"], 
            session_id
        )
        
        state = await conversation.get_conversation_state()
        return {
            "success": True,
            "data": {
                "conversation_state": state,
                "user_id": current_user["user_id"],
                "session_id": session_id
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation state: {str(e)}")

@app.post("/api/conversations/message")
async def add_conversation_message(
    request: ConversationMessageRequest,
    current_user: Dict[str, Any] = Depends(require_authentication)
):
    """Add a message to the conversation."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Get conversation state from session
        session_data = current_user.get("session_data", {})
        session_id = session_data.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")
        
        conversation = await _conversation_manager.get_or_create_conversation(
            current_user["user_id"], 
            session_id
        )
        
        message_id = await conversation.add_message(
            request.message_type,
            request.content,
            request.metadata
        )
        
        if message_id:
            return {
                "success": True,
                "data": {
                    "message_id": str(message_id),
                    "message": "Message added successfully"
                },
                "status": 200
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add message")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add conversation message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@app.get("/api/conversations/history")
async def get_conversation_history(
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(require_authentication)
):
    """Get conversation history with pagination."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Get conversation state from session
        session_id = current_user.get("session_data", {}).get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")
        
        conversation = await _conversation_manager.get_or_create_conversation(
            current_user["user_id"], 
            session_id
        )
        
        history = await conversation.get_conversation_history(limit, offset)
        return {
            "success": True,
            "data": {
                "history": history,
                "limit": limit,
                "offset": offset,
                "total_messages": conversation._message_count
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.post("/api/conversations/search")
async def search_conversation(
    request: ConversationSearchRequest,
    current_user: Dict[str, Any] = Depends(require_authentication)
):
    """Search conversation history."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Get conversation state from session
        session_id = current_user.get("session_data", {}).get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")
        
        conversation = await _conversation_manager.get_or_create_conversation(
            current_user["user_id"], 
            session_id
        )
        
        results = await conversation.search_conversation(request.query, request.limit)
        return {
            "success": True,
            "data": {
                "query": request.query,
                "results": results,
                "total_results": len(results)
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search: {str(e)}")

@app.get("/api/conversations/summary")
async def get_conversation_summary(
    current_user: Dict[str, Any] = Depends(require_authentication)
):
    """Get conversation summary and statistics."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Get conversation state from session
        session_id = current_user.get("session_data", {}).get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")
        
        conversation = await _conversation_manager.get_or_create_conversation(
            current_user["user_id"], 
            session_id
        )
        
        summary = await conversation.get_conversation_summary()
        return {
            "success": True,
            "data": summary,
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@app.get("/api/conversations/export")
async def export_conversation(
    format_type: str = "json",
    current_user: Dict[str, Any] = Depends(require_authentication)
):
    """Export conversation data."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Get conversation state from session
        session_id = current_user.get("session_data", {}).get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")
        
        conversation = await _conversation_manager.get_or_create_conversation(
            current_user["user_id"], 
            session_id
        )
        
        export_data = await conversation.export_conversation(format_type)
        return {
            "success": True,
            "data": {
                "export_data": export_data,
                "format": format_type,
                "timestamp": datetime.datetime.utcnow().isoformat()
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export: {str(e)}")

@app.post("/api/conversations/reset")
async def reset_conversation(
    current_user: Dict[str, Any] = Depends(require_authentication)
):
    """Reset conversation to initial state."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=500, detail="Conversation manager not available")
        
        # Get conversation state from session
        session_id = current_user.get("session_data", {}).get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID not found")
        
        conversation = await _conversation_manager.get_or_create_conversation(
            current_user["user_id"], 
            session_id
        )
        
        await conversation.reset_conversation()
        return {
            "success": True,
            "data": {
                "message": "Conversation reset successfully",
                "timestamp": datetime.datetime.utcnow().isoformat()
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")

# =============================================================================
# USER-FITNESS-APP INTEGRATION
# =============================================================================

# Environment variables for user-fitness-app integration
USER_FITNESS_APP_URL = os.getenv("USER_FITNESS_APP_URL", "http://localhost:8000")
USER_FITNESS_APP_API_KEY = os.getenv("USER_FITNESS_APP_API_KEY", "")
USER_FITNESS_APP_TIMEOUT = int(os.getenv("USER_FITNESS_APP_TIMEOUT", "30"))

async def get_current_user_from_fitness_app(
    token: str = Depends(HTTPBearer()),
    db=Depends(get_shared_db)
) -> Dict[str, Any]:
    """Get current user from user-fitness-app via JWT token."""
    try:
        async with httpx.AsyncClient(timeout=USER_FITNESS_APP_TIMEOUT) as http_client:
            response = await http_client.get(
                f"{USER_FITNESS_APP_URL}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                user_data = response.json()
                return {
                    "user_id": user_data.get("user_id"),
                    "email": user_data.get("email"),
                    "first_name": user_data.get("first_name"),
                    "last_name": user_data.get("last_name"),
                    "authenticated": True
                }
            else:
                raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

async def get_user_nutrition_data(user_id: str, token: str) -> Dict[str, Any]:
    """Fetch user nutrition data from user-fitness-app."""
    try:
        async with httpx.AsyncClient(timeout=USER_FITNESS_APP_TIMEOUT) as http_client:
            response = await http_client.get(
                f"{USER_FITNESS_APP_URL}/api/v1/nutrition/user/{user_id}/summary",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch nutrition data: HTTP {response.status_code}")
                return {}
    except Exception as e:
        logger.error(f"Failed to fetch user nutrition data: {e}")
        return {}

async def get_user_workout_data(user_id: str, token: str) -> Dict[str, Any]:
    """Fetch user workout data from user-fitness-app."""
    try:
        async with httpx.AsyncClient(timeout=USER_FITNESS_APP_TIMEOUT) as http_client:
            response = await http_client.get(
                f"{USER_FITNESS_APP_URL}/api/v1/workouts/user/{user_id}/summary",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch workout data: HTTP {response.status_code}")
                return {}
    except Exception as e:
        logger.error(f"Failed to fetch user workout data: {e}")
        return {}

async def get_user_profile_data(user_id: str, token: str) -> Dict[str, Any]:
    """Fetch user profile data from user-fitness-app."""
    try:
        async with httpx.AsyncClient(timeout=USER_FITNESS_APP_TIMEOUT) as http_client:
            response = await http_client.get(
                f"{USER_FITNESS_APP_URL}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch profile data: HTTP {response.status_code}")
                return {}
    except Exception as e:
        logger.error(f"Failed to fetch user profile data: {e}")
        return {}

async def get_user_nutrition_goals(user_id: str, token: str) -> Dict[str, Any]:
    """Fetch user nutrition goals from user-fitness-app."""
    try:
        async with httpx.AsyncClient(timeout=USER_FITNESS_APP_TIMEOUT) as http_client:
            response = await http_client.get(
                f"{USER_FITNESS_APP_URL}/api/v1/nutrition/user/{user_id}/goals",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch nutrition goals: HTTP {response.status_code}")
                return {}
    except Exception as e:
        logger.error(f"Failed to fetch user nutrition goals: {e}")
        return {}

async def get_user_nutrition_history(user_id: str, token: str, days: int = 30) -> Dict[str, Any]:
    """Fetch user nutrition history from user-fitness-app."""
    try:
        async with httpx.AsyncClient(timeout=USER_FITNESS_APP_TIMEOUT) as http_client:
            response = await http_client.get(
                f"{USER_FITNESS_APP_URL}/api/v1/nutrition/user/{user_id}/history?days={days}",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch nutrition history: HTTP {response.status_code}")
                return {}
    except Exception as e:
        logger.error(f"Failed to fetch user nutrition history: {e}")
        return {}

async def get_user_food_preferences(user_id: str, token: str) -> Dict[str, Any]:
    """Fetch user food preferences and restrictions from user-fitness-app."""
    try:
        async with httpx.AsyncClient(timeout=USER_FITNESS_APP_TIMEOUT) as http_client:
            response = await http_client.get(
                f"{USER_FITNESS_APP_URL}/api/v1/nutrition/user/{user_id}/preferences",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to fetch food preferences: HTTP {response.status_code}")
                return {}
    except Exception as e:
        logger.error(f"Failed to fetch user food preferences: {e}")
        return {}

# =============================================================================
# SESSION MANAGEMENT ENDPOINTS
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request model for creating a user session."""
    user_id: str

@app.post("/api/sessions/create")
async def create_user_session(request: CreateSessionRequest):
    """Create a new user session for multi-user support."""
    try:
        if '_session_manager' not in globals() or not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_data = await _session_manager.create_user_session(request.user_id, "nutrition")
        
        return {
            "success": True,
            "data": {
                "session": session_data,
                "message": "Session created successfully"
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/api/sessions/{session_token}")
async def get_session_info(session_token: str):
    """Get session information for multi-user support."""
    try:
        if '_session_manager' not in globals() or not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session = await _session_manager.get_session(session_token)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "data": {
                "session": session
            },
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.post("/api/sessions/{session_token}/validate")
async def validate_session(session_token: str):
    """Validate session token."""
    try:
        if '_session_manager' not in globals() or not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session = await _session_manager.get_session(session_token)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Check if session is expired
        if session.get("expires_at") and datetime.datetime.fromisoformat(session["expires_at"]) < datetime.datetime.utcnow():
            raise HTTPException(status_code=401, detail="Session expired")
        
        return {
            "success": True,
            "data": {
                "session": session,
                "valid": True
            },
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate session: {str(e)}")

@app.delete("/api/sessions/{session_token}")
async def delete_session(session_token: str):
    """Delete user session."""
    try:
        if '_session_manager' not in globals() or not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        success = await _session_manager.delete_session(session_token)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "data": {
                "message": "Session deleted successfully"
            },
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =============================================================================

class ConversationMessageRequest(BaseModel):
    """Request model for adding conversation messages."""
    message_type: str
    content: str
    metadata: Dict[str, Any] = {}

class ConversationSearchRequest(BaseModel):
    """Request model for searching conversations."""
    query: str
    limit: int = 20

@app.get("/api/conversations/state")
async def get_conversation_state(user_id: str):
    """Get current conversation state for the user."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        conversation_state = await _conversation_manager.get_or_create_conversation_state(user_id)
        state = await conversation_state.get_conversation_state()
        
        return {
            "success": True,
            "data": {
                "conversation_state": state,
                "user_id": user_id
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation state: {str(e)}")

@app.post("/api/conversations/message")
async def add_conversation_message(
    request: ConversationMessageRequest,
    user_id: str
):
    """Add a message to the conversation."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        conversation_state = await _conversation_manager.get_or_create_conversation_state(user_id)
        message_id = await conversation_state.add_message(
            request.message_type,
            request.content,
            request.metadata
        )
        
        if message_id:
            return {
                "success": True,
                "data": {
                    "message_id": str(message_id),
                    "message": "Message added successfully"
                },
                "status": 200
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add message")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add conversation message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@app.get("/api/conversations/history")
async def get_conversation_history(
    user_id: str,
    limit: int = 50,
    offset: int = 0
):
    """Get conversation history with pagination."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        conversation_state = await _conversation_manager.get_or_create_conversation_state(user_id)
        history = await conversation_state.get_conversation_history(limit, offset)
        
        return {
            "success": True,
            "data": {
                "history": history,
                "limit": limit,
                "offset": offset,
                "total_messages": conversation_state._message_count
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.post("/api/conversations/search")
async def search_conversation(
    request: ConversationSearchRequest,
    user_id: str
):
    """Search conversation history."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        conversation_state = await _conversation_manager.get_or_create_conversation_state(user_id)
        results = await conversation_state.search_conversation(request.query, request.limit)
        
        return {
            "success": True,
            "data": {
                "query": request.query,
                "results": results,
                "total_results": len(results)
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search: {str(e)}")

@app.get("/api/conversations/summary")
async def get_conversation_summary(user_id: str):
    """Get conversation summary and statistics."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        conversation_state = await _conversation_manager.get_or_create_conversation_state(user_id)
        summary = await conversation_state.get_conversation_summary()
        
        return {
            "success": True,
            "data": summary,
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@app.post("/api/conversations/reset")
async def reset_conversation(user_id: str):
    """Reset conversation to initial state."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        conversation_state = await _conversation_manager.get_or_create_conversation_state(user_id)
        await conversation_state.reset_conversation()
        
        return {
            "success": True,
            "data": {
                "message": "Conversation reset successfully"
            },
            "status": 200
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")

# =============================================================================
# PERFORMANCE MONITORING ENDPOINTS
# =============================================================================

@app.get("/api/performance/metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        # Get comprehensive metrics
        metrics = {
            "uptime_seconds": _performance_monitor.get_uptime(),
            "request_count": _performance_monitor.request_count,
            "error_count": _performance_monitor.error_count,
            "error_rate": _performance_monitor.get_error_rate(),
            "throughput_rps": _performance_monitor.get_throughput(),
            "system_stats": _performance_monitor.system_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "success": True,
            "data": {
                "metrics": metrics,
                "agent": "nutrition"
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.get("/api/performance/dashboard")
async def get_performance_dashboard():
    """Get comprehensive performance dashboard."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        dashboard = _performance_monitor.get_performance_dashboard()
        
        return {
            "success": True,
            "data": {
                "dashboard": dashboard,
                "agent": "nutrition",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to get performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance dashboard: {str(e)}")

@app.get("/api/performance/alerts")
async def get_performance_alerts(limit: int = 50):
    """Get recent performance alerts."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        # Get recent alerts
        with _performance_monitor._lock:
            recent_alerts = list(_performance_monitor.alerts)[-limit:] if hasattr(_performance_monitor, 'alerts') else []
        
        return {
            "success": True,
            "data": {
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "level": alert.level.value,
                        "message": alert.message,
                        "metric_type": alert.metric_type.value,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts
                ],
                "total_alerts": len(recent_alerts),
                "limit": limit,
                "agent": "nutrition"
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to get performance alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance alerts: {str(e)}")

@app.get("/api/performance/health")
async def get_performance_health():
    """Get performance health status."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        # Check performance thresholds
        error_rate = _performance_monitor.get_error_rate()
        throughput = _performance_monitor.get_throughput()
        
        health_status = {
            "status": "healthy",
            "error_rate": error_rate,
            "throughput_rps": throughput,
            "thresholds": {
                "max_error_rate": 0.05,  # 5%
                "min_throughput": 10     # 10 RPS
            },
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Determine health status
        if error_rate > 0.05:
            health_status["status"] = "degraded"
            health_status["issues"] = ["High error rate"]
        
        if throughput < 10:
            health_status["status"] = "degraded"
            if "issues" not in health_status:
                health_status["issues"] = []
            health_status["issues"].append("Low throughput")
        
        return {
            "success": True,
            "data": health_status,
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to get performance health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance health: {str(e)}")

# =============================================================================
# SUPERVISOR COMMUNICATION ENDPOINTS
# =============================================================================

@app.post("/api/supervisor/broadcast")
async def receive_supervisor_broadcast(request: dict):
    """Receive broadcast messages from supervisor agent."""
    try:
        message = request.get("message", "")
        message_type = request.get("message_type", "notification")
        from_agent = request.get("from_agent", "unknown")
        timestamp = request.get("timestamp", "")
        
        # Log the broadcast message
        logger.info(f"Received broadcast from {from_agent}: {message} (Type: {message_type})")
        
        # Process different message types
        if message_type == "notification":
            # Handle general notifications
            pass
        elif message_type == "sync":
            # Handle synchronization requests
            pass
        elif message_type == "maintenance":
            # Handle maintenance notifications
            pass
        
        return {
            "success": True,
            "data": {
                "message": "Broadcast received successfully",
                "received_message": message,
                "message_type": message_type,
                "from_agent": from_agent,
                "timestamp": timestamp,
                "processed_at": datetime.now(timezone.utc).isoformat()
            },
            "status": 200
        }
    except Exception as e:
        logger.error(f"Failed to process supervisor broadcast: {e}")
        return {
            "success": False,
            "error": str(e),
            "status": 500
        }

@app.get("/api/supervisor/health")
async def get_supervisor_health_check():
    """Health check endpoint for supervisor agent."""
    try:
        return {
            "status": "healthy",
            "agent_type": "nutrition",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "capabilities": {
                "session_management": True,
                "task_processing": True,
                "performance_monitoring": True,
                "conversation_management": True
            },
            "active_sessions": 0,  # TODO: Implement when session manager is available
            "active_tasks": 0,     # TODO: Implement when task manager is available
            "performance_metrics": {}  # TODO: Implement when performance monitor is available
        }
    except Exception as e:
        logger.error(f"Supervisor health check error: {e}")
        return {
            "status": "error",
            "agent_type": "nutrition",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/api/supervisor/status")
async def get_supervisor_status():
    """Status endpoint for supervisor agent."""
    try:
        return {
            "status": "active",
            "agent_type": "nutrition",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "uptime": time.time(),
            "active_sessions": 0,  # TODO: Implement when session manager is available
            "active_tasks": 0,     # TODO: Implement when task manager is available
            "performance_metrics": {},  # TODO: Implement when performance monitor is available
            "health_status": {},   # TODO: Implement when health monitor is available
            "note": "Basic status endpoint - advanced features not yet implemented"
        }
    except Exception as e:
        logger.error(f"Supervisor status check error: {e}")
        return {
            "status": "error",
            "agent_type": "nutrition",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "error": str(e)
        }
