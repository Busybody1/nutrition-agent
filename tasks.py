"""
Celery Tasks for Nutrition Agent
Handles background meal planning and nutrition analysis
"""

import logging
import time
import asyncio
from datetime import datetime
from celery import current_task
from celery_config import celery_app
from utils.redis_cache import cache

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def generate_meal_plan_task(self, user_id: str, preferences: dict, dietary_restrictions: list) -> dict:
    """
    Generate meal plan asynchronously.
    
    Args:
        user_id: User ID
        preferences: User preferences
        dietary_restrictions: List of dietary restrictions
        
    Returns:
        dict: Generated meal plan
    """
    try:
        # Update task state to show progress
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Analyzing dietary requirements', 'progress': 10}
        )
        
        # Check cache first
        cached_plan = cache.get_cached_meal_plan(user_id, preferences)
        if cached_plan:
            logger.info(f"✅ Using cached meal plan for user {user_id}")
            return {
                'status': 'completed',
                'meal_plan': cached_plan['meal_plan'],
                'cached': True,
                'task_id': self.request.id
            }
        
        # Step 1: Analyze requirements
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Analyzing dietary requirements', 'progress': 20}
        )
        
        # Import here to avoid circular imports
        from main import get_ai_response
        
        # Step 2: Generate meals using AI
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating meal options', 'progress': 40}
        )
        
        from utils.prompts import build_celery_meal_plan_prompt
        prompt = build_celery_meal_plan_prompt(preferences, dietary_restrictions)
        
        # Get AI response (no timeout concern in worker!)
        ai_response, model = asyncio.run(get_ai_response(prompt, max_tokens=2000, temperature=0.7))
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Calculating nutrition', 'progress': 70}
        )
        
        # Step 3: Process and structure the response
        meal_plan = {
            'user_id': user_id,
            'preferences': preferences,
            'dietary_restrictions': dietary_restrictions,
            'ai_response': ai_response,
            'model_used': model,
            'generated_at': datetime.utcnow().isoformat(),
            'task_id': self.request.id
        }
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Finalizing meal plan', 'progress': 90}
        )
        
        # Cache the meal plan
        cache.cache_meal_plan(user_id, preferences, meal_plan)
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Complete', 'progress': 100}
        )
        
        logger.info(f"✅ Meal plan generated for user {user_id}")
        
        result = {
            'status': 'completed',
            'meal_plan': meal_plan,
            'cached': False,
            'task_id': self.request.id,
            'execution_time': time.time() - self.request.timestart
        }
        
        # Cache job result
        cache.cache_job_result(self.request.id, result)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Meal plan generation failed: {e}")
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            error_result = {
                'status': 'failed',
                'error': str(e),
                'user_id': user_id,
                'task_id': self.request.id
            }
            cache.cache_job_result(self.request.id, error_result)
            return error_result

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def analyze_nutrition_task(self, user_id: str, food_items: list) -> dict:
    """
    Analyze nutrition for food items asynchronously.
    
    Args:
        user_id: User ID
        food_items: List of food items to analyze
        
    Returns:
        dict: Nutrition analysis
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Analyzing food items', 'progress': 20}
        )
        
        # Check cache first
        cached_analysis = cache.get_cached_nutrition_analysis(food_items)
        if cached_analysis:
            logger.info(f"✅ Using cached nutrition analysis")
            return {
                'status': 'completed',
                'analysis': cached_analysis['analysis'],
                'cached': True,
                'task_id': self.request.id
            }
        
        # Import here to avoid circular imports
        from main import get_ai_response
        
        from utils.prompts import build_celery_nutrition_analysis_prompt
        prompt = build_celery_nutrition_analysis_prompt(food_items)
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Processing nutrition data', 'progress': 60}
        )
        
        # Get AI response
        ai_response, model = asyncio.run(get_ai_response(prompt, max_tokens=1500, temperature=0.7))
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Finalizing analysis', 'progress': 80}
        )
        
        # Structure the analysis
        analysis = {
            'food_items': food_items,
            'ai_response': ai_response,
            'model_used': model,
            'analyzed_at': datetime.utcnow().isoformat(),
            'task_id': self.request.id
        }
        
        # Cache the analysis
        cache.cache_nutrition_analysis(food_items, analysis)
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Complete', 'progress': 100}
        )
        
        logger.info(f"✅ Nutrition analysis completed")
        
        result = {
            'status': 'completed',
            'analysis': analysis,
            'cached': False,
            'task_id': self.request.id,
            'execution_time': time.time() - self.request.timestart
        }
        
        # Cache job result
        cache.cache_job_result(self.request.id, result)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Nutrition analysis failed: {e}")
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            error_result = {
                'status': 'failed',
                'error': str(e),
                'user_id': user_id,
                'task_id': self.request.id
            }
            cache.cache_job_result(self.request.id, error_result)
            return error_result

@celery_app.task(bind=True, max_retries=2)
def generate_shopping_list_task(self, user_id: str, meal_plan: dict) -> dict:
    """
    Generate shopping list from meal plan asynchronously.
    
    Args:
        user_id: User ID
        meal_plan: Meal plan data
        
    Returns:
        dict: Shopping list
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating shopping list', 'progress': 50}
        )
        
        # Import here to avoid circular imports
        from main import get_ai_response
        
        from utils.prompts import build_celery_shopping_list_prompt
        prompt = build_celery_shopping_list_prompt(meal_plan)
        
        # Get AI response
        ai_response, model = asyncio.run(get_ai_response(prompt, max_tokens=1000, temperature=0.7))
        
        shopping_list = {
            'user_id': user_id,
            'meal_plan': meal_plan,
            'shopping_list': ai_response,
            'model_used': model,
            'generated_at': datetime.utcnow().isoformat(),
            'task_id': self.request.id
        }
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Complete', 'progress': 100}
        )
        
        logger.info(f"✅ Shopping list generated for user {user_id}")
        
        result = {
            'status': 'completed',
            'shopping_list': shopping_list,
            'task_id': self.request.id,
            'execution_time': time.time() - self.request.timestart
        }
        
        # Cache job result
        cache.cache_job_result(self.request.id, result)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Shopping list generation failed: {e}")
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=e)
        except self.MaxRetriesExceededError:
            error_result = {
                'status': 'failed',
                'error': str(e),
                'user_id': user_id,
                'task_id': self.request.id
            }
            cache.cache_job_result(self.request.id, error_result)
            return error_result
