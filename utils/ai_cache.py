"""
AI Response Cache for Cost Optimization
Provides intelligent caching of AI responses to reduce duplicate calls.
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AIResponseCache:
    """Intelligent cache for AI responses with TTL and invalidation."""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def _generate_key(self, prompt: str, max_tokens: int, temperature: float, function_name: str) -> str:
        """Generate cache key for a request."""
        content = f"{prompt}_{max_tokens}_{temperature}_{function_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() > cache_entry.get("expires_at", 0)
    
    def get(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, function_name: str = "") -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        self.stats["total_requests"] += 1
        
        cache_key = self._generate_key(prompt, max_tokens, temperature, function_name)
        
        if cache_key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        cache_entry = self.cache[cache_key]
        
        if self._is_expired(cache_entry):
            del self.cache[cache_key]
            self.stats["evictions"] += 1
            self.stats["misses"] += 1
            return None
        
        self.stats["hits"] += 1
        return cache_entry.get("response")
    
    def set(self, prompt: str, response: Dict[str, Any], max_tokens: int = 1000, 
            temperature: float = 0.7, function_name: str = "", ttl: Optional[int] = None) -> None:
        """Cache a response with TTL."""
        cache_key = self._generate_key(prompt, max_tokens, temperature, function_name)
        
        if ttl is None:
            ttl = self._get_ttl_for_function(function_name)
        
        cache_entry = {
            "response": response,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
            "function_name": function_name,
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8]
        }
        
        self.cache[cache_key] = cache_entry
        logger.debug(f"Cached response for {function_name} (TTL: {ttl}s)")
    
    def _get_ttl_for_function(self, function_name: str) -> int:
        """Get TTL based on function type."""
        ttl_map = {
            "create_meal_plan": 24 * 3600,      # 24 hours
            "create_workout_plan": 12 * 3600,   # 12 hours
            "get_nutrition_summary": 6 * 3600,  # 6 hours
            "get_workout_summary": 6 * 3600,    # 6 hours
            "get_activity_summary": 4 * 3600,   # 4 hours
            "general_nutrition_response": 2 * 3600,  # 2 hours
            "general_workout_response": 2 * 3600,    # 2 hours
            "general_activity_response": 2 * 3600,   # 2 hours
            "intelligent_routing": 1 * 3600,    # 1 hour
            "analyze_food_image": 12 * 3600,    # 12 hours
            "analyze_exercise_image": 12 * 3600, # 12 hours
        }
        
        return ttl_map.get(function_name, self.default_ttl)
    
    def invalidate(self, function_name: str = None, user_id: str = None) -> int:
        """Invalidate cache entries by function name or user ID."""
        invalidated = 0
        
        if function_name:
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if entry.get("function_name") == function_name
            ]
        elif user_id:
            # This would require storing user_id in cache entries
            # For now, we'll implement a simple pattern match
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if user_id in str(entry.get("response", {}))
            ]
        else:
            # Clear all cache
            keys_to_remove = list(self.cache.keys())
        
        for key in keys_to_remove:
            del self.cache[key]
            invalidated += 1
        
        logger.info(f"Invalidated {invalidated} cache entries")
        return invalidated
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        self.stats["evictions"] += len(expired_keys)
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self.stats["hits"] / max(self.stats["total_requests"], 1) * 100
        )
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of cache in MB."""
        total_size = 0
        for entry in self.cache.values():
            total_size += len(json.dumps(entry, default=str))
        return total_size / (1024 * 1024)  # Convert to MB
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        logger.info("AI response cache cleared")

# Global cache instance
ai_cache = AIResponseCache()
