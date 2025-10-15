"""
Redis Cache Utility for Nutrition Agent
Provides caching functionality for meal plans and nutrition analysis
"""

import redis
import json
import hashlib
import os
import logging
from typing import Optional, Any, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NutritionRedisCache:
    """Redis cache manager optimized for nutrition operations."""
    
    def __init__(self):
        self.redis_url = os.getenv('REDISCLOUD_URL', os.getenv('REDIS_URL', 'redis://localhost:6379'))
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        self.namespace = "nutrition"
        
        # Test connection
        try:
            self.client.ping()
            logger.info("✅ Connected to Redis Cloud")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.client = None
    
    def _get_key(self, key: str, namespace: str = None) -> str:
        """Generate namespaced cache key."""
        ns = namespace or self.namespace
        return f"{ns}:{key}"
    
    def set(self, key: str, value: Any, ttl: int = 3600, namespace: str = None) -> bool:
        """Set cache value with TTL."""
        if not self.client:
            return False
        
        try:
            cache_key = self._get_key(key, namespace)
            serialized_value = json.dumps(value)
            return self.client.setex(cache_key, ttl, serialized_value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str, namespace: str = None) -> Optional[Any]:
        """Get cache value."""
        if not self.client:
            return None
        
        try:
            cache_key = self._get_key(key, namespace)
            value = self.client.get(cache_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def cache_meal_plan(self, user_id: str, preferences: Dict[str, Any], meal_plan: Dict[str, Any], ttl: int = 86400) -> bool:
        """Cache meal plan (24 hours TTL)."""
        cache_key = f"meal_plan:{hashlib.md5(f'{user_id}:{json.dumps(preferences, sort_keys=True)}'.encode()).hexdigest()}"
        value = {
            'meal_plan': meal_plan,
            'user_id': user_id,
            'preferences': preferences,
            'timestamp': datetime.utcnow().isoformat(),
            'cache_key': cache_key
        }
        return self.set(cache_key, value, ttl, 'meal_plans')
    
    def get_cached_meal_plan(self, user_id: str, preferences: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached meal plan."""
        cache_key = f"meal_plan:{hashlib.md5(f'{user_id}:{json.dumps(preferences, sort_keys=True)}'.encode()).hexdigest()}"
        return self.get(cache_key, 'meal_plans')
    
    def cache_nutrition_analysis(self, food_items: list, analysis: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache nutrition analysis (30 minutes TTL)."""
        cache_key = f"nutrition:{hashlib.md5(json.dumps(sorted(food_items), sort_keys=True).encode()).hexdigest()}"
        value = {
            'analysis': analysis,
            'food_items': food_items,
            'timestamp': datetime.utcnow().isoformat(),
            'cache_key': cache_key
        }
        return self.set(cache_key, value, ttl, 'nutrition_analysis')
    
    def get_cached_nutrition_analysis(self, food_items: list) -> Optional[Dict[str, Any]]:
        """Get cached nutrition analysis."""
        cache_key = f"nutrition:{hashlib.md5(json.dumps(sorted(food_items), sort_keys=True).encode()).hexdigest()}"
        return self.get(cache_key, 'nutrition_analysis')
    
    def cache_ai_response(self, prompt: str, response: str, model: str, ttl: int = 1800) -> bool:
        """Cache AI response (30 minutes TTL)."""
        cache_key = f"ai:{hashlib.md5(prompt.encode()).hexdigest()}"
        value = {
            'response': response,
            'model': model,
            'timestamp': datetime.utcnow().isoformat(),
            'prompt_hash': cache_key
        }
        return self.set(cache_key, value, ttl, 'ai_responses')
    
    def get_cached_ai_response(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached AI response."""
        cache_key = f"ai:{hashlib.md5(prompt.encode()).hexdigest()}"
        return self.get(cache_key, 'ai_responses')
    
    def cache_job_result(self, job_id: str, result: Any, ttl: int = 3600) -> bool:
        """Cache job result."""
        return self.set(f"job:{job_id}", result, ttl, 'job_results')
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get cached job result."""
        return self.get(f"job:{job_id}", 'job_results')
    
    def delete(self, key: str, namespace: str = None) -> bool:
        """Delete cache key."""
        if not self.client:
            return False
        
        try:
            cache_key = self._get_key(key, namespace)
            return bool(self.client.delete(cache_key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.client:
            return {'connected': False}
        
        try:
            info = self.client.info()
            return {
                'connected': True,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {'connected': False, 'error': str(e)}

# Global cache instance
cache = NutritionRedisCache()
