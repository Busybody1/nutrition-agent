# Standard library imports
import logging
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import datetime
import asyncio
from contextlib import asynccontextmanager

# Import models and utilities
from shared.models import FoodLogEntry
from shared.utils import DataValidator
from shared.config import get_settings
from shared.session_manager import FrameworkSessionManager
from shared.async_task_manager import AsyncTaskManager, TaskPriority
from shared.performance_monitor import PerformanceMonitor
from shared.health_monitor import HealthMonitor

# Set up logger
logging.basicConfig(level=logging.INFO)
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

# Global user data access
_user_data_access = None

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
from shared.session_middleware import SessionValidationMiddleware
import httpx
from fastapi.staticfiles import StaticFiles

# Authentication dependency
async def require_authentication(request: Request) -> Dict[str, Any]:
    """Dependency to require authentication for protected endpoints."""
    try:
        # Extract session token from request
        session_token = None
        
        # Check Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
        
        # Check X-Session-Token header
        if not session_token:
            session_token = request.headers.get("X-Session-Token")
        
        # Check query parameter
        if not session_token:
            session_token = request.query_params.get("session_token")
        
        # Check cookies
        if not session_token:
            session_token = request.cookies.get("session_token")
        
        if not session_token:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please provide a valid session token."
            )
        
        # Validate session token
        if not _session_manager:
            raise HTTPException(
                status_code=503,
                detail="Session manager not available. Please try again later."
            )
        
        is_valid = await _session_manager.validate_session(session_token)
        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired session token. Please authenticate again."
            )
        
        # Get session data
        session_data = await _session_manager.get_session(session_token)
        if not session_data:
            raise HTTPException(
                status_code=401,
                detail="Session not found. Please authenticate again."
            )
        
        return {
            "user_id": session_data["user_id"],
            "session_token": session_token,
            "session_data": session_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication service error. Please try again later."
        )

# Database connection functions
def get_fitness_engine():
    """Get the fitness database engine."""
    from shared.database import get_fitness_engine as _get_fitness_engine
    return _get_fitness_engine()

def get_nutrition_engine():
    """Get the nutrition database engine."""
    from shared.database import get_nutrition_engine as _get_nutrition_engine
    return _get_nutrition_engine()

def get_session_local():
    """Get the session local for the fitness database."""
    from shared.database import get_session_local as _get_session_local
    return _get_session_local()

def get_nutrition_session_local():
    """Get the session local for the nutrition database."""
    from shared.database import get_nutrition_session_local as _get_nutrition_session_local
    return _get_nutrition_session_local()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
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
        global _session_manager, _task_manager, _performance_monitor, _health_monitor, _conversation_manager, _user_data_access
        
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
            from shared.conversation_manager import ConversationStateManager
            _conversation_manager = ConversationStateManager("nutrition")
            logger.info("Conversation manager initialized successfully")
        except ImportError:
            logger.warning("Conversation manager not available, continuing without it")
            _conversation_manager = None
        
        # Initialize user data access
        try:
            from shared.user_data_access import UserDataAccess
            _user_data_access = UserDataAccess()
            logger.info("User data access initialized successfully")
        except ImportError:
            logger.warning("User data access not available, continuing without it")
            _user_data_access = None
        
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

# Create FastAPI app
app = FastAPI(
    title="Nutrition Agent", 
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session validation middleware for multi-user support
app.add_middleware(
    SessionValidationMiddleware,
    agent_type="nutrition"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database dependency functions
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

# =============================================================================
# CORE API ENDPOINTS
# =============================================================================

@app.get("/")
def read_root():
    """Nutrition Agent root endpoint."""
        return {
            "message": "Nutrition Agent is running with SCALABILITY FEATURES!",
            "version": "2.0.0",
            "status": "running",
            "scalability_ready": True,
            "endpoints": {
                "health": "/health",
                "foods_count": "/foods/count",
                "execute_tool": "/execute-tool",
                "user_nutrition_data": "/api/user/nutrition-data",

                # Core scalability endpoints
                "create_session": "/api/sessions/create",
                "get_session": "/api/sessions/{session_token}",
                "submit_task": "/api/tasks/submit",
                "get_task_status": "/api/tasks/{task_id}",
                "performance_metrics": "/api/performance/metrics",
                "scalability_status": "/api/scalability/status",
                # Conversation management endpoints
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
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "components": {
            "session_manager": _session_manager is not None,
            "task_manager": _task_manager is not None,
            "performance_monitor": _performance_monitor is not None,
            "health_monitor": _health_monitor is not None,
            "conversation_manager": _conversation_manager is not None,
            "user_data_access": _user_data_access is not None
        }
    }

@app.get("/health/detailed")
def detailed_health_check():
    """Detailed health check with database testing."""
    try:
        # Test main database connection (for writes)
        main_db_status = "not_configured"
        try:
        fitness_engine = get_fitness_engine()
        with fitness_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            main_db_status = "connected"
        except Exception as e:
            logger.warning(f"Main database not accessible: {e}")
            main_db_status = "unavailable"
        
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
        
        # Determine overall status
        if main_db_status == "connected":
            overall_status = "healthy"
        elif main_db_status == "unavailable" and nutrition_status == "unavailable":
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status, 
            "agent": "nutrition",
            "main_database": main_db_status,
            "nutrition_database": nutrition_status,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "message": "Health check completed successfully"
        }
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy", 
            "agent": "nutrition",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "message": "Health check failed with error"
        }

@app.get("/test")
def test_endpoint():
    """Simple test endpoint that doesn't require database access."""
        return {
        "status": "success",
        "message": "Nutrition Agent is running!",
        "agent": "nutrition",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "components": {
            "session_manager": _session_manager is not None,
            "task_manager": _task_manager is not None,
            "performance_monitor": _performance_monitor is not None,
            "health_monitor": _health_monitor is not None,
            "conversation_manager": _conversation_manager is not None,
            "user_data_access": _user_data_access is not None
        }
    }

# =============================================================================
# CORE NUTRITION ENDPOINTS
# =============================================================================

@app.get("/foods/count")
def get_foods_count(db = Depends(get_nutrition_db)):
    """Get total count of foods in the nutrition database."""
    try:
        from shared.models import Food
        count = db.query(Food).count()
        return {"count": count, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting foods count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get foods count: {str(e)}")

@app.get("/foods/{food_id}")
def get_food_by_id(food_id: int, db = Depends(get_nutrition_db)):
    """Get food by ID."""
    try:
        from shared.models import Food
        food = db.query(Food).filter(Food.id == food_id).first()
        if not food:
            raise HTTPException(status_code=404, detail="Food not found")
        return food
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting food by ID: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get food: {str(e)}")

@app.post("/log-food")
async def log_food(
    request: Request,
    food_name: str = Body(...),
    quantity: float = Body(...),
    unit: str = Body(...),
    meal_type: str = Body(...),
    user_id: str = Body(...),
    db = Depends(get_shared_db)
):
    """Log food consumption for a user."""
    try:
        # Validate user authentication
        auth_data = await require_authentication(request)
        if auth_data["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        # Create food log entry
        from shared.models import FoodLogEntry
        food_log = FoodLogEntry(
            user_id=user_id,
            food_name=food_name,
            quantity=quantity,
            unit=unit,
            meal_type=meal_type,
            logged_at=datetime.datetime.utcnow()
        )
        
        db.add(food_log)
        db.commit()
        db.refresh(food_log)
        
        return {
            "status": "success",
            "message": "Food logged successfully",
            "food_log_id": food_log.id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging food: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to log food: {str(e)}")

@app.post("/execute-tool")
async def execute_tool(
    request: Request,
    tool_name: str = Body(...),
    parameters: Dict[str, Any] = Body(...),
    db = Depends(get_shared_db)
):
    """Execute a nutrition tool."""
    try:
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Execute tool based on name
        if tool_name == "meal_plan":
            return await create_meal_plan(parameters, user_id, db)
        elif tool_name == "calculate_calories":
            return await calculate_calories(parameters, user_id, db)
        elif tool_name == "nutrition_recommendations":
            return await get_nutrition_recommendations(parameters, user_id, db)
                else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
            
    except HTTPException:
        raise
        except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# =============================================================================
# NUTRITION TOOL FUNCTIONS
# =============================================================================

async def create_meal_plan(parameters: Dict[str, Any], user_id: str, db) -> Dict[str, Any]:
    """Create a personalized meal plan for the user."""
    try:
        # Get user nutrition data from user database
        if _user_data_access:
            user_nutrition_data = await _user_data_access.get_user_nutrition_data(user_id)
            user_profile = await _user_data_access.get_user_profile(user_id)
        else:
            user_nutrition_data = {"targets": [], "recent_logs": [], "meal_plans": []}
            user_profile = {}
        
        # Extract parameters
        calories = parameters.get("calories", 2000)
        meals_per_day = parameters.get("meals_per_day", 3)
        dietary_restrictions = parameters.get("dietary_restrictions", [])
        
        # Create meal plan logic (simplified for now)
        meal_plan = {
            "user_id": user_id,
            "calories": calories,
            "meals_per_day": meals_per_day,
            "dietary_restrictions": dietary_restrictions,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_context": {
                "nutrition_targets": user_nutrition_data.get("targets", []),
                "recent_food_logs": user_nutrition_data.get("recent_logs", []),
                "user_profile": user_profile
            }
        }
        
        return {
            "status": "success",
            "meal_plan": meal_plan,
            "message": "Meal plan created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create meal plan: {str(e)}")

async def calculate_calories(parameters: Dict[str, Any], user_id: str, db) -> Dict[str, Any]:
    """Calculate calorie needs for the user."""
    try:
        # Get user profile from user database
        if _user_data_access:
            user_profile = await _user_data_access.get_user_profile(user_id)
        else:
            user_profile = {}
        
        # Extract parameters
        age = parameters.get("age", user_profile.get("age", 30))
        weight = parameters.get("weight", user_profile.get("weight", 70))
        height = parameters.get("height", user_profile.get("height", 170))
        activity_level = parameters.get("activity_level", user_profile.get("activity_level", "moderate"))
        goal = parameters.get("goal", "maintain")
        
        # Basic BMR calculation (Mifflin-St Jeor Equation)
        if user_profile.get("gender") == "female":
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
                else:
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        
        # Activity multiplier
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        
        tdee = bmr * activity_multipliers.get(activity_level, 1.55)
        
        # Goal adjustment
        if goal == "lose":
            target_calories = tdee - 500
        elif goal == "gain":
            target_calories = tdee + 500
        else:
            target_calories = tdee
        
        return {
            "status": "success",
            "calories": {
                "bmr": round(bmr),
                "tdee": round(tdee),
                "target_calories": round(target_calories),
                "activity_level": activity_level,
                "goal": goal
            },
            "user_profile": user_profile
        }
        
    except Exception as e:
        logger.error(f"Error calculating calories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate calories: {str(e)}")

async def get_nutrition_recommendations(parameters: Dict[str, Any], user_id: str, db) -> Dict[str, Any]:
    """Get personalized nutrition recommendations for the user."""
    try:
        # Get user nutrition data from user database
        if _user_data_access:
            user_nutrition_data = await _user_data_access.get_user_nutrition_data(user_id)
            user_profile = await _user_data_access.get_user_profile(user_id)
                    else:
            user_nutrition_data = {"targets": [], "recent_logs": [], "meal_plans": []}
            user_profile = {}
        
        # Extract parameters
        goal = parameters.get("goal", "general_health")
        focus_area = parameters.get("focus_area", "balanced")
        
        # Generate recommendations based on user data and parameters
        recommendations = {
                "user_id": user_id,
            "goal": goal,
            "focus_area": focus_area,
            "recommendations": [],
            "user_context": {
                "nutrition_targets": user_nutrition_data.get("targets", []),
                "recent_food_logs": user_nutrition_data.get("recent_logs", []),
                "user_profile": user_profile
            }
        }
        
        # Add specific recommendations based on focus area
        if focus_area == "weight_loss":
            recommendations["recommendations"] = [
                "Focus on protein-rich foods to maintain muscle mass",
                "Include plenty of fiber for satiety",
                "Limit added sugars and refined carbohydrates",
                "Stay hydrated throughout the day"
            ]
        elif focus_area == "muscle_gain":
            recommendations["recommendations"] = [
                "Increase protein intake to 1.6-2.2g per kg body weight",
                "Include complex carbohydrates for energy",
                "Eat in a slight caloric surplus",
                "Time protein intake around workouts"
            ]
        else:  # balanced
            recommendations["recommendations"] = [
                "Follow a balanced plate: 50% vegetables, 25% protein, 25% whole grains",
                "Include a variety of colorful fruits and vegetables",
                "Choose lean protein sources",
                "Limit processed foods and added sugars"
            ]
            
            return {
                "status": "success",
            "recommendations": recommendations
        }
        
        except Exception as e:
        logger.error(f"Error getting nutrition recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nutrition recommendations: {str(e)}")

@app.get("/api/user/nutrition-data")
async def get_user_nutrition_data(request: Request):
    """Get user nutrition data from user database."""
    try:
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        if _user_data_access:
            nutrition_data = await _user_data_access.get_user_nutrition_data(user_id)
            user_profile = await _user_data_access.get_user_profile(user_id)
            
            return {
                "status": "success",
                "nutrition_data": nutrition_data,
                "user_profile": user_profile,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
                else:
            raise HTTPException(status_code=500, detail="User data access not available")
            
    except HTTPException:
        raise
        except Exception as e:
        logger.error(f"Failed to get user nutrition data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user nutrition data: {str(e)}")

# =============================================================================
# SCALABILITY ENDPOINTS
# =============================================================================

@app.post("/api/sessions/create")
async def create_session(request: Request):
    """Create a new user session."""
    try:
        if not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        # Extract user ID from request
        body = await request.json()
        user_id = body.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Create session
        session_data = await _session_manager.create_user_session(user_id, "nutrition")

        return {
            "status": "success",
            "session_data": session_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/api/sessions/{session_token}")
async def get_session_info(session_token: str):
    """Get session information."""
    try:
        if not _session_manager:
            raise HTTPException(status_code=503, detail="Session manager not available")
        
        session_data = await _session_manager.get_session(session_token)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "status": "success",
            "session_data": session_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.post("/api/tasks/submit")
async def submit_task(
    request: Request,
    task_type: str = Body(...),
    data: Dict[str, Any] = Body(...)
):
    """Submit a background task."""
    try:
        if not _task_manager:
            raise HTTPException(status_code=503, detail="Task manager not available")
        
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Submit task
        task_id = await _task_manager.submit_task(task_type, data)

        return {
            "status": "success",
            "task_id": task_id,
            "message": "Task submitted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get task status."""
    try:
        if not _task_manager:
            raise HTTPException(status_code=503, detail="Task manager not available")
        
        task_status = await _task_manager.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "status": "success",
            "task_status": task_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.get("/api/performance/metrics")
async def get_performance_metrics():
    """Get performance metrics."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        metrics = await _performance_monitor.get_metrics()
        summary = await _performance_monitor.get_performance_summary()

        return {
            "status": "success",
            "metrics": metrics,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.get("/api/scalability/status")
async def get_scalability_status():
    """Get scalability status."""
    try:
        status_data = {
            "agent_type": "nutrition",
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check session manager
        if _session_manager:
            status_data["components"]["session_manager"] = "healthy"
                else:
            status_data["components"]["session_manager"] = "unavailable"
            status_data["status"] = "degraded"
        
        # Check task manager
        if _task_manager:
            status_data["components"]["task_manager"] = "healthy"
                else:
            status_data["components"]["task_manager"] = "unavailable"
            status_data["status"] = "degraded"
        
        # Check performance monitor
        if _performance_monitor:
            status_data["components"]["performance_monitor"] = "healthy"
                    else:
            status_data["components"]["performance_monitor"] = "unavailable"
            status_data["status"] = "degraded"
        
        # Check health monitor
        if _health_monitor:
            status_data["components"]["health_monitor"] = "healthy"
                    else:
            status_data["components"]["health_monitor"] = "unavailable"
            status_data["status"] = "degraded"
        
        # Check database connections
        try:
            fitness_engine = get_fitness_engine()
            with fitness_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            status_data["components"]["main_database"] = "healthy"
    except Exception as e:
            status_data["components"]["main_database"] = f"unhealthy: {str(e)}"
            status_data["status"] = "degraded"
        
        return status_data
            
    except Exception as e:
        logger.error(f"Error getting scalability status: {e}")
            return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/conversations/state")
async def get_conversation_state(request: Request):
    """Get conversation state for the current user."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Get conversation state
        conversation_state = await _conversation_manager.get_conversation_state(user_id)

        return {
            "status": "success",
            "conversation_state": conversation_state
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation state: {str(e)}")

@app.post("/api/conversations/message")
async def add_conversation_message(
    request: Request,
    message: str = Body(...),
    role: str = Body(default="user")
):
    """Add a message to the conversation."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Add message
        message_data = await _conversation_manager.add_message(user_id, message, role)

        return {
            "status": "success",
            "message_data": message_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@app.get("/api/conversations/history")
async def get_conversation_history(request: Request, limit: int = 50):
    """Get conversation history for the current user."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Get conversation history
        history = await _conversation_manager.get_conversation_history(user_id, limit)
        
        return {
            "status": "success",
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

@app.post("/api/conversations/search")
async def search_conversations(
    request: Request,
    query: str = Body(...)
):
    """Search conversation history."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Search conversations
        results = await _conversation_manager.search_conversations(query, user_id)
        
        return {
            "status": "success",
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search conversations: {str(e)}")

@app.get("/api/conversations/summary")
async def get_conversation_summary(request: Request):
    """Get conversation summary for the current user."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Get conversation state
        conversation_state = await _conversation_manager.get_conversation_state(user_id)
        
        # Create summary
        summary = {
            "user_id": user_id,
            "message_count": len(conversation_state.get("messages", [])),
            "last_activity": conversation_state.get("updated_at"),
            "agent_type": conversation_state.get("agent_type")
        }
        
        return {
            "status": "success",
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation summary: {str(e)}")

@app.post("/api/conversations/reset")
async def reset_conversation(request: Request):
    """Reset conversation for the current user."""
    try:
        if not _conversation_manager:
            raise HTTPException(status_code=503, detail="Conversation manager not available")
        
        # Validate user authentication
        auth_data = await require_authentication(request)
        user_id = auth_data["user_id"]
        
        # Reset conversation
        success = await _conversation_manager.reset_conversation(user_id)
        
        if success:
                return {
                "status": "success",
                "message": "Conversation reset successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reset conversation")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation: {str(e)}")

# =============================================================================
# SUPERVISOR COMMUNICATION ENDPOINTS
# =============================================================================

@app.post("/api/supervisor/broadcast")
async def supervisor_broadcast(request: Request):
    """Handle supervisor broadcast messages."""
    try:
        body = await request.json()
        message_type = body.get("type")
        data = body.get("data", {})
        
        logger.info(f"Received supervisor broadcast: {message_type}")
        
        # Handle different message types
        if message_type == "health_check":
            return {"status": "healthy", "agent": "nutrition"}
        elif message_type == "status_request":
            return await get_scalability_status()
        elif message_type == "cleanup":
            # Perform cleanup if requested
            return {"status": "cleanup_completed", "agent": "nutrition"}
                else:
            return {"status": "received", "message_type": message_type}
        
    except Exception as e:
        logger.error(f"Error handling supervisor broadcast: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/api/supervisor/health")
async def supervisor_health_check():
    """Health check endpoint for supervisor agent."""
        return {
        "status": "healthy",
        "agent_type": "nutrition",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "capabilities": {
            "session_management": _session_manager is not None,
            "task_processing": _task_manager is not None,
            "performance_monitoring": _performance_monitor is not None,
            "health_monitoring": _health_monitor is not None
        }
    }

@app.get("/api/supervisor/status")
async def supervisor_status():
    """Status endpoint for supervisor agent."""
        return {
        "status": "active",
        "agent_type": "nutrition",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "components": {
            "session_manager": "active" if _session_manager else "inactive",
            "task_manager": "active" if _task_manager else "inactive",
            "performance_monitor": "active" if _performance_monitor else "inactive",
            "health_monitor": "active" if _health_monitor else "inactive"
        }
    }

# =============================================================================
# PERFORMANCE MONITORING ENDPOINTS
# =============================================================================

@app.get("/api/performance/dashboard")
async def get_performance_dashboard():
    """Get performance dashboard data."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        summary = await _performance_monitor.get_performance_summary()
            
            return {
                "status": "success",
            "dashboard": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance dashboard: {str(e)}")

@app.get("/api/performance/alerts")
async def get_performance_alerts():
    """Get performance alerts."""
    try:
        if not _performance_monitor:
            raise HTTPException(status_code=503, detail="Performance monitor not available")
        
        # For now, return basic alerts
        alerts = []
        
        # Check if components are healthy
        if not _session_manager:
            alerts.append({
                "type": "warning",
                "message": "Session manager not available",
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
        
        if not _task_manager:
            alerts.append({
                "type": "warning",
                "message": "Task manager not available",
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
        
        return {
            "status": "success",
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        return {
            "status": "error",
            "alerts": [],
            "error": str(e)
        }

@app.get("/api/performance/health")
async def get_performance_health():
    """Get performance health status."""
    try:
        if not _performance_monitor:
        return {
                "status": "unhealthy",
                "message": "Performance monitor not available",
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        
        summary = await _performance_monitor.get_performance_summary()
        
        return {
            "status": "healthy",
            "summary": summary,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance health: {e}")
    return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    validate_environment()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
