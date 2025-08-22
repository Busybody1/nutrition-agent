#!/usr/bin/env python3
"""
Simple Nutrition Agent - Groq AI Integration

This agent provides intelligent nutrition functionality using Groq AI.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SIMPLE NUTRITION AGENT
# =============================================================================

# Simple in-memory storage (for demo purposes)
nutrition_data = {}

# Initialize Groq client
groq_client = None

# Database connection status
database_status = {"main": False, "user": False, "nutrition": False, "workout": False}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class NutritionRequest(BaseModel):
    """Simple nutrition request model."""
    message: str
    user_id: str

class NutritionResponse(BaseModel):
    """Simple nutrition response model."""
    response: str
    user_id: str
    agent: str
    timestamp: str

# =============================================================================
# DATABASE INITIALIZATION AND CONNECTION
# =============================================================================

async def initialize_databases():
    """Initialize all database connections on startup."""
    global database_status
    
    try:
        # Test main database
        try:
            from shared.database import get_main_db
            db = get_main_db()
            db.execute(text("SELECT 1"))
            db.close()
            database_status["main"] = True
            logger.info("✅ Main database connected successfully")
        except Exception as e:
            logger.error(f"❌ Main database connection failed: {e}")
            database_status["main"] = False

        # Test user database
        try:
            from shared.database import get_user_db
            db = get_user_db()
            db.execute(text("SELECT 1"))
            db.close()
            database_status["user"] = True
            logger.info("✅ User database connected successfully")
        except Exception as e:
            logger.error(f"❌ User database connection failed: {e}")
            database_status["user"] = False

        # Test nutrition database
        try:
            from shared.database import get_nutrition_db
            db = get_nutrition_db()
            if db:
                db.execute(text("SELECT 1"))
                db.close()
                database_status["nutrition"] = True
                logger.info("✅ Nutrition database connected successfully")
            else:
                logger.warning("⚠️ Nutrition database not configured")
        except Exception as e:
            logger.error(f"❌ Nutrition database connection failed: {e}")
            database_status["nutrition"] = False

        # Test workout database
        try:
            from shared.database import get_workout_db
            db = get_workout_db()
            if db:
                db.execute(text("SELECT 1"))
                db.close()
                database_status["workout"] = True
                logger.info("✅ Workout database connected successfully")
            else:
                logger.warning("⚠️ Workout database not configured")
        except Exception as e:
            logger.error(f"❌ Workout database connection failed: {e}")
            database_status["workout"] = False

    except Exception as e:
        logger.error(f"Database initialization error: {e}")

@asynccontextmanager
async def get_db():
    """Get database session - simple and reliable."""
    try:
        from shared.database import get_db as _get_db
        db = _get_db()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")

# =============================================================================
# AI INITIALIZATION
# =============================================================================

async def initialize_ai():
    """Initialize AI clients on startup."""
    global groq_client
    
    try:
        # Initialize Groq AI
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                groq_client = Groq(api_key=groq_api_key)
                # Test connection with a simple request
                test_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    temperature=0.1
                )
                logger.info("✅ Groq AI client initialized and tested successfully")
            except Exception as e:
                logger.error(f"❌ Groq AI initialization failed: {e}")
                groq_client = None
        else:
            logger.warning("⚠️ GROQ_API_KEY not set, AI nutrition features will not work")
            groq_client = None
            
    except Exception as e:
        logger.error(f"AI initialization error: {e}")
        groq_client = None

# =============================================================================
# USER VALIDATION FUNCTIONS
# =============================================================================

async def validate_user_exists(user_id: str) -> bool:
    """Validate that the user exists in the database."""
    try:
        # Allow test users for development/testing
        if user_id == "default_user" or user_id.startswith("test_"):
            logger.info(f"Allowing test user: {user_id}")
            return True
            
        # Check against the user database using USER_DATABASE_URI
        try:
            from shared.database import get_user_db
            db = get_user_db()
            try:
                # Check if user exists in users table using correct schema
                result = db.execute(
                    text("SELECT user_id, email, account_status FROM users WHERE user_id = :user_id AND account_status = 'active'"),
                    {"user_id": user_id}
                ).fetchone()
                
                if not result:
                    logger.warning(f"User validation failed: User {user_id} does not exist or is inactive")
                    return False
                    
                logger.info(f"User validation successful: User {user_id} ({result.email}) is valid")
                return True
                
            finally:
                db.close()
                
        except Exception as db_error:
            logger.error(f"Database error during user validation: {db_error}")
            # If database fails, allow the user to proceed for now
            logger.warning(f"Database validation failed for user {user_id}, allowing access")
            return True
        
    except Exception as e:
        logger.error(f"Error validating user {user_id}: {e}")
        return False

async def require_valid_user(user_id: str):
    """Require a valid user_id or raise HTTPException."""
    if not await validate_user_exists(user_id):
        raise HTTPException(
            status_code=404, 
            detail=f"User with ID '{user_id}' does not exist in the database. Please provide a valid user ID."
        )

# =============================================================================
# NUTRITION FUNCTIONS WITH GROQ AI
# =============================================================================

async def log_meal(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Log a meal with AI nutrition analysis."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (required) and other parameters (optional)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what you ate.")
        
        # Optional parameters with defaults
        meal_type = parameters.get("meal_type", "snack")
        estimated_calories = parameters.get("estimated_calories", 0)
        mood = parameters.get("mood", "good")
        notes = parameters.get("notes", "")
        
        meal_log = {
            "user_id": user_id,
            "description": description,
            "meal_type": meal_type,
            "estimated_calories": estimated_calories,
            "mood": mood,
            "notes": notes,
            "logged_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Add AI nutrition insights if Groq is available
        if groq_client:
            try:
                insight_prompt = f"""Analyze this meal description and provide nutrition insights:
- User description: {description}
- Meal type: {meal_type}
- Estimated calories: {estimated_calories}
- Mood: {mood}
- Additional notes: {notes}

Provide:
1. Nutritional analysis of the meal
2. Health benefits and considerations
3. Suggestions for balance and variety
4. Portion control recommendations
5. Timing and meal planning tips
6. Any specific insights based on their description

Keep it helpful and educational."""

                insight_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": insight_prompt}],
                    max_tokens=400,
                    temperature=0.7
                )
                
                meal_log["ai_insights"] = insight_response.choices[0].message.content
                meal_log["ai_model"] = "llama3-70b-8192"
                
            except Exception as e:
                logger.warning(f"Failed to generate AI insights: {e}")
        
        return {
            "status": "success",
            "meal_log": meal_log,
            "message": "Meal logged successfully with AI nutrition insights"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging meal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log meal: {str(e)}")

async def get_nutrition_summary(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Get nutrition summary with AI analysis."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (required)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what kind of nutrition summary you need.")
        
        # Optional parameters
        days = parameters.get("days", 7)
        goals = parameters.get("goals", "general health")
        
        # Generate AI-powered summary if Groq is available
        if groq_client:
            try:
                summary_prompt = f"""Based on this user request, provide a comprehensive nutrition summary:
- User request: {description}
- Time period: {days} days
- User goals: {goals}

Provide:
1. Summary of typical nutrition patterns
2. Nutrient balance analysis
3. Goal achievement assessment
4. Personalized recommendations
5. Meal planning suggestions
6. Next steps for improvement

Make it personalized and actionable based on their description."""

                summary_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                
                ai_summary = summary_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI summary: {e}")
                ai_summary = "AI summary generation failed, but here's your basic summary."
        else:
            ai_summary = "AI features are currently unavailable."
        
        return {
            "status": "success",
            "user_id": user_id,
            "summary": {
                "period_days": days,
                "goals": goals,
                "ai_analysis": ai_summary,
                "generated_at": datetime.now(timezone.utc).isoformat()
            },
            "message": "Nutrition summary generated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating nutrition summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

async def plan_meal(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Plan a meal with AI recommendations."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description (required)
        description = parameters.get("description", "")
        if not description:
            raise HTTPException(status_code=400, detail="Description parameter is required. Please describe what kind of meal you want to plan.")
        
        # Optional parameters
        meal_type = parameters.get("meal_type", "dinner")
        dietary_restrictions = parameters.get("dietary_restrictions", "none")
        calorie_target = parameters.get("calorie_target", 600)
        
        # Generate AI-powered meal plan if Groq is available
        if groq_client:
            try:
                meal_prompt = f"""Create a personalized meal plan:
- User request: {description}
- Meal type: {meal_type}
- Dietary restrictions: {dietary_restrictions}
- Calorie target: {calorie_target}

Provide:
1. Complete meal plan with ingredients
2. Nutritional breakdown
3. Preparation instructions
4. Alternative options
5. Shopping list
6. Time-saving tips
7. Health benefits

Make it realistic, delicious, and aligned with their preferences."""

                meal_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": meal_prompt}],
                    max_tokens=600,
                    temperature=0.7
                )
                
                ai_meal_plan = meal_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI meal plan: {e}")
                ai_meal_plan = "AI meal planning failed, but here are some general suggestions."
        else:
            ai_meal_plan = "AI features are currently unavailable."
        
        return {
            "status": "success",
            "user_id": user_id,
            "meal_plan": {
                "description": description,
                "meal_type": meal_type,
                "dietary_restrictions": dietary_restrictions,
                "calorie_target": calorie_target,
                "ai_plan": ai_meal_plan,
                "planned_at": datetime.now(timezone.utc).isoformat()
            },
            "message": "Meal plan created successfully with AI recommendations"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error planning meal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to plan meal: {str(e)}")

async def general_nutrition_response(parameters: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """General nutrition response with AI."""
    try:
        # Validate user exists first
        await require_valid_user(user_id)
        
        # Get description/message (required)
        message = parameters.get("description", "") or parameters.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="Description or message parameter is required.")
        
        # Generate AI response if Groq is available
        if groq_client:
            try:
                ai_prompt = f"""You are a nutrition and health expert. The user asks: {message}

Provide a helpful, encouraging response that:
1. Addresses their specific nutrition question or concern
2. Offers practical advice and tips
3. Motivates them to make healthy choices
4. Suggests next steps or alternatives
5. Maintains a positive, supportive tone

Keep it concise but comprehensive."""

                ai_response = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": ai_prompt}],
                    max_tokens=400,
                    temperature=0.7
                )
                
                response_text = ai_response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"Failed to generate AI response: {e}")
                response_text = "I'm here to help with nutrition! I can log meals, get summaries, and plan meals. Please provide a description of what you need."
        else:
            response_text = "I'm here to help with nutrition! I can log meals, get summaries, and plan meals. Please provide a description of what you need."
        
        return {
            "status": "success",
            "response": response_text,
            "user_id": user_id,
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating nutrition response: {e}")
        # Fallback to basic response if AI fails
        return {
            "status": "success",
            "response": "I'm here to help with nutrition! I can log meals, get summaries, and plan meals. Please provide a description of what you need.",
            "user_id": user_id,
            "agent": "nutrition",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Nutrition Agent - Groq AI Integration",
    description="Intelligent nutrition agent with Groq AI for personalized meal planning and nutrition advice",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================================================================
# LIFESPAN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting Nutrition Agent...")
    
    # Initialize databases
    await initialize_databases()
    
    # Initialize AI
    await initialize_ai()
    
    logger.info("✅ Nutrition Agent startup complete")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Serve the test interface HTML file."""
    try:
        with open("static/test_interface.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Test interface not found</h1>", status_code=404)

@app.get("/health")
async def health_check():
    """Comprehensive health check with all service statuses."""
    return {
        "status": "healthy",
        "agent": "nutrition",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "groq_ai": "connected" if groq_client else "disconnected",
            "main_database": "connected" if database_status["main"] else "disconnected",
            "user_database": "connected" if database_status["user"] else "disconnected",
            "nutrition_database": "connected" if database_status["nutrition"] else "disconnected",
            "workout_database": "connected" if database_status["workout"] else "disconnected"
        },
        "message": "Nutrition agent is running smoothly"
    }

@app.get("/test-db")
async def test_database():
    """Test database connection."""
    try:
        async with get_db() as db:
            result = db.execute(text("SELECT 1")).fetchone()
            return {
                "status": "success",
                "database": "connected",
                "test_query": result[0] if result else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return {
            "status": "error",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.post("/execute-tool")
async def execute_tool(
    tool_name: str = Body(...),
    parameters: Dict[str, Any] = Body(...)
):
    """Execute a nutrition tool."""
    try:
        user_id = parameters.get("user_id", "default_user")
        
        # Execute tool based on name
        if tool_name == "log_meal":
            return await log_meal(parameters, user_id)
        elif tool_name == "get_nutrition_summary":
            return await get_nutrition_summary(parameters, user_id)
        elif tool_name == "plan_meal":
            return await plan_meal(parameters, user_id)
        elif tool_name == "general_nutrition":
            # General nutrition response with AI
            return await general_nutrition_response(parameters, user_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
