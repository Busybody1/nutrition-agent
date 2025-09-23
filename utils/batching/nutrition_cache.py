"""
Nutrition Agent Cache
Optimized caching for nutrition data with nutrition-specific TTL strategies.
"""

import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class NutritionCacheEntry:
    """Nutrition-specific cache entry."""
    key: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    ttl: float = 1800  # Default 30 minutes
    nutrition_type: str = "general"
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

class NutritionCache:
    """Nutrition-specific cache with optimized TTL strategies."""
    
    def __init__(self, max_size: int = 800, enable_nutrition_optimization: bool = True):
        self.max_size = max_size
        self.enable_nutrition_optimization = enable_nutrition_optimization
        
        # Cache storage
        self.cache: Dict[str, NutritionCacheEntry] = {}
        self.nutrition_index: Dict[str, List[str]] = {}  # Index by nutrition type
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "total_accesses": 0,
            "cache_size": 0,
            "hit_rate": 0.0
        }
        
        # Nutrition-specific TTL configurations
        self.nutrition_ttl_config = {
            "meal_plan": 3600,        # 1 hour - meal plans
            "nutrition_analysis": 1800,  # 30 minutes - analysis
            "food_suggestion": 1800,  # 30 minutes - suggestions
            "dietary_restriction": 7200,  # 2 hours - dietary restrictions
            "general": 1800,         # 30 minutes - general queries
            "user_profile": 7200,    # 2 hours - user profiles
            "food_database": 3600,   # 1 hour - food database
            "calorie_calculation": 1800  # 30 minutes - calorie calculations
        }
    
    def _generate_key(self, prompt: str, max_tokens: int, temperature: float, 
                     function_name: str, nutrition_type: str = "general") -> str:
        """Generate cache key for nutrition data."""
        content = f"{nutrition_type}:{function_name}:{max_tokens}:{temperature}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_nutrition_ttl(self, nutrition_type: str) -> float:
        """Get TTL for specific nutrition type."""
        return self.nutrition_ttl_config.get(nutrition_type, 1800)
    
    def _is_expired(self, entry: NutritionCacheEntry) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry.timestamp > entry.ttl
    
    def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = []
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries
        to_remove = len(self.cache) - self.max_size + 1
        for key, _ in sorted_entries[:to_remove]:
            self._remove_entry(key)
            self.stats["evictions"] += 1
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and indexes."""
        if key in self.cache:
            entry = self.cache[key]
            
            # Remove from nutrition index
            if entry.nutrition_type in self.nutrition_index:
                if key in self.nutrition_index[entry.nutrition_type]:
                    self.nutrition_index[entry.nutrition_type].remove(key)
            
            del self.cache[key]
            self.stats["cache_size"] = len(self.cache)
    
    def _update_access_stats(self, entry: NutritionCacheEntry):
        """Update access statistics."""
        entry.access_count += 1
        entry.last_accessed = time.time()
        self.stats["total_accesses"] += 1
        self.stats["hit_rate"] = self.stats["hits"] / max(self.stats["total_accesses"], 1)
    
    def get(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
            function_name: str = "", nutrition_type: str = "general") -> Optional[Dict[str, Any]]:
        """Get cached nutrition data."""
        key = self._generate_key(prompt, max_tokens, temperature, function_name, nutrition_type)
        
        if key in self.cache:
            entry = self.cache[key]
            
            if not self._is_expired(entry):
                self.stats["hits"] += 1
                self._update_access_stats(entry)
                return entry.data
            else:
                # Remove expired entry
                self._remove_entry(key)
        
        self.stats["misses"] += 1
        return None
    
    def set(self, prompt: str, data: Dict[str, Any], max_tokens: int = 1000, 
            temperature: float = 0.7, function_name: str = "", 
            nutrition_type: str = "general", custom_ttl: Optional[float] = None):
        """Set cached nutrition data."""
        key = self._generate_key(prompt, max_tokens, temperature, function_name, nutrition_type)
        
        # Determine TTL
        ttl = custom_ttl if custom_ttl is not None else self._get_nutrition_ttl(nutrition_type)
        
        # Create cache entry
        entry = NutritionCacheEntry(
            key=key,
            data=data,
            ttl=ttl,
            nutrition_type=nutrition_type
        )
        
        # Clean up if needed
        self._evict_expired()
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Add to cache
        self.cache[key] = entry
        
        # Update nutrition index
        if nutrition_type not in self.nutrition_index:
            self.nutrition_index[nutrition_type] = []
        self.nutrition_index[nutrition_type].append(key)
        
        self.stats["sets"] += 1
        self.stats["cache_size"] = len(self.cache)
    
    def clear_nutrition_type(self, nutrition_type: str):
        """Clear all entries for a specific nutrition type."""
        if nutrition_type in self.nutrition_index:
            keys_to_remove = self.nutrition_index[nutrition_type].copy()
            for key in keys_to_remove:
                self._remove_entry(key)
            del self.nutrition_index[nutrition_type]
    
    def clear_expired(self):
        """Clear all expired entries."""
        self._evict_expired()
    
    def clear_all(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.nutrition_index.clear()
        self.stats["cache_size"] = 0
    
    def get_nutrition_stats(self, nutrition_type: str) -> Dict[str, Any]:
        """Get statistics for a specific nutrition type."""
        if nutrition_type not in self.nutrition_index:
            return {
                "entries": 0,
                "total_accesses": 0,
                "hit_rate": 0.0,
                "avg_ttl": 0.0
            }
        
        entries = [self.cache[key] for key in self.nutrition_index[nutrition_type] if key in self.cache]
        
        if not entries:
            return {
                "entries": 0,
                "total_accesses": 0,
                "hit_rate": 0.0,
                "avg_ttl": 0.0
            }
        
        total_accesses = sum(entry.access_count for entry in entries)
        avg_ttl = sum(entry.ttl for entry in entries) / len(entries)
        
        return {
            "entries": len(entries),
            "total_accesses": total_accesses,
            "hit_rate": total_accesses / max(len(entries), 1),
            "avg_ttl": avg_ttl
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            **self.stats,
            "nutrition_types": {
                nutrition_type: self.get_nutrition_stats(nutrition_type)
                for nutrition_type in self.nutrition_index.keys()
            },
            "max_size": self.max_size,
            "nutrition_optimization_enabled": self.enable_nutrition_optimization
        }
