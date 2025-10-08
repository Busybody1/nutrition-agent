"""
Nutrition Agent Batch Manager
Optimized for meal planning with balanced response times and efficient batching.
"""

import asyncio
import time
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class NutritionPriority(Enum):
    """Priority levels for nutrition agent calls."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4  # Meal planning requests
    CRITICAL = 5  # Dietary restrictions or allergies

class NutritionBatchStrategy(Enum):
    """Batching strategies optimized for nutrition management."""
    MEAL_PLANNING_FIRST = "meal_planning_first"  # Prioritize meal planning
    BALANCED = "balanced"  # Balance between speed and efficiency
    EFFICIENT = "efficient"  # Prioritize batching efficiency
    ANALYSIS_FOCUSED = "analysis_focused"  # Optimize for nutrition analysis

@dataclass
class NutritionCall:
    """Nutrition-specific AI call request."""
    id: str
    prompt: str
    max_tokens: int
    temperature: float
    function_name: str
    user_id: str
    priority: NutritionPriority = NutritionPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    use_cache: bool = True
    ai_client_func: Callable = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 25.0  # Longer timeout for complex meal planning
    nutrition_type: str = "general"  # meal_plan, analysis, suggestion, etc.
    dietary_restrictions: bool = False  # Whether this involves dietary restrictions
    
    def __lt__(self, other):
        """Priority queue ordering (higher priority first)."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        if self.dietary_restrictions != other.dietary_restrictions:
            return self.dietary_restrictions > other.dietary_restrictions
        return self.timestamp < other.timestamp

@dataclass
class NutritionResponse:
    """Nutrition-specific AI response."""
    id: str
    content: str
    model: str
    cached: bool = False
    response_time: float = 0.0
    cost: float = 0.0
    tokens_used: int = 0
    batch_id: str = ""
    timestamp: float = field(default_factory=time.time)
    nutrition_type: str = "general"

class NutritionBatchManager:
    """Nutrition-specific batch manager optimized for meal planning."""
    
    def __init__(self, 
                 max_batch_size: int = 8,  # Optimized for nutrition analysis
                 max_wait_time: float = 0.6,  # Balanced wait time
                 strategy: NutritionBatchStrategy = NutritionBatchStrategy.BALANCED,
                 enable_dietary_priority: bool = True,
                 enable_nutrition_optimization: bool = True):
        
        # Core configuration optimized for nutrition management
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.strategy = strategy
        self.enable_dietary_priority = enable_dietary_priority
        self.enable_nutrition_optimization = enable_nutrition_optimization
        
        # Queues and processing
        self.pending_calls: List[NutritionCall] = []
        self.dietary_priority_calls: List[NutritionCall] = []  # Separate queue for dietary restrictions
        self.processing_batches: Dict[str, List[NutritionCall]] = {}
        self.batch_lock = asyncio.Lock()
        
        # Nutrition-specific caching
        self.cache: Dict[str, NutritionResponse] = {}
        self.cache_ttl: Dict[str, float] = {}
        self.nutrition_cache: Dict[str, List[NutritionResponse]] = defaultdict(list)
        
        # Performance tracking
        self.stats = {
            "total_calls": 0,
            "batched_calls": 0,
            "dietary_priority_calls": 0,
            "cache_hits": 0,
            "total_savings": 0.0,
            "average_response_time": 0.0,
            "average_batch_size": 0.0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "nutrition_efficiency": 0.0
        }
        
        # Nutrition-specific metrics
        self.nutrition_metrics = {
            "meal_plan_calls": 0,
            "nutrition_analysis_calls": 0,
            "food_suggestion_calls": 0,
            "dietary_restriction_calls": 0,
            "peak_meal_planning_hour": 0,
            "average_meal_complexity": 0.0
        }
        
        # Batch processing task
        self.batch_processor_task = None
        self.is_running = False
        
        # Don't start the batch processor during initialization
        # It will be started when first needed
    
    def _start_batch_processor(self):
        """Start the background batch processor."""
        if not self.is_running and self.batch_processor_task is None:
            try:
                self.is_running = True
                self.batch_processor_task = asyncio.create_task(self._batch_processor())
            except RuntimeError:
                # No event loop running, will start later
                self.is_running = False
                self.batch_processor_task = None
    
    async def _batch_processor(self):
        """Background processor optimized for nutrition management."""
        while self.is_running:
            try:
                # Process dietary priority calls first
                await self._process_dietary_priority_calls()
                
                # Then process regular batches
                await self._process_pending_batches()
                
                await asyncio.sleep(0.1)  # Check every 100ms
            except Exception as e:
                logger.error(f"Nutrition batch processor error: {e}")
                await asyncio.sleep(0.5)
    
    async def _process_dietary_priority_calls(self):
        """Process dietary restriction calls with priority."""
        if not self.dietary_priority_calls:
            return
        
        async with self.batch_lock:
            calls_to_process = self.dietary_priority_calls.copy()
            self.dietary_priority_calls.clear()
        
        # Process dietary priority calls immediately
        for call in calls_to_process:
            asyncio.create_task(self._process_single_call(call, "dietary_priority"))
    
    async def _process_pending_batches(self):
        """Process pending batches based on nutrition strategy."""
        async with self.batch_lock:
            if not self.pending_calls:
                return
            
            # Determine batch size based on strategy
            batch_size = self._calculate_nutrition_batch_size()
            wait_time = self._calculate_nutrition_wait_time()
            
            # Check if we should process a batch
            should_process = self._should_process_nutrition_batch(batch_size, wait_time)
            
            if should_process:
                # Create batch
                batch = self._create_nutrition_batch(batch_size)
                if batch:
                    asyncio.create_task(self._process_batch(batch))
    
    def _calculate_nutrition_batch_size(self) -> int:
        """Calculate optimal batch size for nutrition management."""
        if self.strategy == NutritionBatchStrategy.MEAL_PLANNING_FIRST:
            return max(1, int(self.max_batch_size * 0.6))
        elif self.strategy == NutritionBatchStrategy.ANALYSIS_FOCUSED:
            return self.max_batch_size
        elif self.strategy == NutritionBatchStrategy.EFFICIENT:
            return self.max_batch_size
        else:  # BALANCED
            return max(1, int(self.max_batch_size * 0.8))
    
    def _calculate_nutrition_wait_time(self) -> float:
        """Calculate wait time optimized for nutrition management."""
        if self.strategy == NutritionBatchStrategy.MEAL_PLANNING_FIRST:
            return self.max_wait_time * 0.5
        elif self.strategy == NutritionBatchStrategy.ANALYSIS_FOCUSED:
            return self.max_wait_time * 1.2
        elif self.strategy == NutritionBatchStrategy.EFFICIENT:
            return self.max_wait_time * 1.5
        else:  # BALANCED
            return self.max_wait_time
    
    def _should_process_nutrition_batch(self, batch_size: int, wait_time: float) -> bool:
        """Determine if we should process a batch now."""
        if len(self.pending_calls) >= batch_size:
            return True
        
        # Check if oldest call has been waiting too long
        if self.pending_calls:
            oldest_call = min(self.pending_calls, key=lambda x: x.timestamp)
            if time.time() - oldest_call.timestamp > wait_time:
                return True
        
        return False
    
    def _create_nutrition_batch(self, target_size: int) -> List[NutritionCall]:
        """Create an optimal batch for nutrition management."""
        if not self.pending_calls:
            return []
        
        # Sort by priority and nutrition type
        sorted_calls = sorted(self.pending_calls, key=lambda x: (
            -x.priority.value,  # Higher priority first
            x.nutrition_type,   # Group by nutrition type
            x.timestamp        # Older calls first
        ))
        
        # Select optimal batch
        batch = []
        for call in sorted_calls[:target_size]:
            if self._is_nutrition_compatible(call, batch):
                batch.append(call)
                self.pending_calls.remove(call)
        
        return batch
    
    def _is_nutrition_compatible(self, call: NutritionCall, batch: List[NutritionCall]) -> bool:
        """Check if a call is compatible with existing batch for nutrition management."""
        if not batch:
            return True
        
        # Group by nutrition type for better batching
        batch_types = {c.nutrition_type for c in batch}
        if call.nutrition_type not in batch_types and len(batch_types) > 1:
            return False
        
        return True
    
    async def _process_batch(self, batch: List[NutritionCall]):
        """Process a batch of nutrition calls."""
        if not batch:
            return
        
        batch_id = f"nutrition_batch_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Process calls in parallel
            tasks = []
            for call in batch:
                task = asyncio.create_task(self._process_single_call(call, batch_id))
                tasks.append(task)
            
            # Wait for all calls to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update metrics
            processing_time = time.time() - start_time
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            
            self._update_nutrition_batch_metrics(batch_id, len(batch), processing_time, 
                                               success_count / len(batch), processing_time / len(batch))
            
        except Exception as e:
            logger.error(f"Nutrition batch processing error: {e}")
        finally:
            # Clean up
            if batch_id in self.processing_batches:
                del self.processing_batches[batch_id]
    
    async def _process_single_call(self, call: NutritionCall, batch_id: str):
        """Process a single nutrition call."""
        try:
            # Check cache first
            if call.use_cache:
                cached_response = self._get_cached_nutrition_response(call)
                if cached_response:
                    call.future.set_result((cached_response.content, cached_response.model))
                    return
            
            # Execute AI call
            start_time = time.time()
            response, model = await call.ai_client_func(
                call.prompt, call.max_tokens, call.temperature
            )
            response_time = time.time() - start_time
            
            # Create response object
            nutrition_response = NutritionResponse(
                id=call.id,
                content=response,
                model=model,
                response_time=response_time,
                batch_id=batch_id,
                tokens_used=call.max_tokens,
                nutrition_type=call.nutrition_type
            )
            
            # Cache the response
            if call.use_cache:
                self._cache_nutrition_response(call, nutrition_response)
            
            # Set result
            call.future.set_result((response, model))
            
        except Exception as e:
            logger.error(f"Nutrition call processing error: {e}")
            call.future.set_exception(e)
    
    def _get_cached_nutrition_response(self, call: NutritionCall) -> Optional[NutritionResponse]:
        """Get cached response for nutrition call."""
        cache_key = self._generate_nutrition_cache_key(call)
        
        # Check direct cache
        if cache_key in self.cache:
            cached_response = self.cache[cache_key]
            if time.time() < self.cache_ttl.get(cache_key, 0):
                self.stats["cache_hits"] += 1
                return cached_response
        
        # Check nutrition-specific cache
        nutrition_key = f"{call.nutrition_type}:{call.max_tokens}:{call.temperature}"
        if nutrition_key in self.nutrition_cache:
            for cached_response in self.nutrition_cache[nutrition_key]:
                if self._is_nutrition_similar(call.prompt, cached_response.content):
                    self.stats["cache_hits"] += 1
                    return cached_response
        
        return None
    
    def _is_nutrition_similar(self, prompt1: str, response1: str, threshold: float = 0.8) -> bool:
        """Check if two nutrition prompts are similar."""
        words1 = set(prompt1.lower().split())
        words2 = set(response1.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union >= threshold
    
    def _cache_nutrition_response(self, call: NutritionCall, response: NutritionResponse):
        """Cache a nutrition response."""
        cache_key = self._generate_nutrition_cache_key(call)
        self.cache[cache_key] = response
        
        # Set TTL based on nutrition type
        ttl = self._get_nutrition_cache_ttl(call.nutrition_type)
        self.cache_ttl[cache_key] = time.time() + ttl
        
        # Add to nutrition-specific cache
        nutrition_key = f"{call.nutrition_type}:{call.max_tokens}:{call.temperature}"
        self.nutrition_cache[nutrition_key].append(response)
        
        # Limit nutrition cache size
        if len(self.nutrition_cache[nutrition_key]) > 40:
            self.nutrition_cache[nutrition_key] = self.nutrition_cache[nutrition_key][-20:]
    
    def _get_nutrition_cache_ttl(self, nutrition_type: str) -> float:
        """Get cache TTL based on nutrition type."""
        ttl_map = {
            "meal_plan": 3600,        # 1 hour for meal plans
            "nutrition_analysis": 1800,  # 30 minutes for analysis
            "food_suggestion": 1800,  # 30 minutes for suggestions
            "dietary_restriction": 7200,  # 2 hours for dietary restrictions
            "general": 1800          # 30 minutes default
        }
        return ttl_map.get(nutrition_type, 1800)
    
    def _generate_nutrition_cache_key(self, call: NutritionCall) -> str:
        """Generate cache key for nutrition call."""
        content = f"{call.nutrition_type}:{call.function_name}:{call.max_tokens}:{call.temperature}:{call.prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_nutrition_batch_metrics(self, batch_id: str, size: int, processing_time: float, 
                                      success_rate: float, avg_response_time: float):
        """Update nutrition batch processing metrics."""
        # Update stats
        self.stats["batched_calls"] += size
        self.stats["average_batch_size"] = (
            (self.stats["average_batch_size"] * (self.stats["batched_calls"] - size) + size) 
            / self.stats["batched_calls"]
        )
        self.stats["success_rate"] = (
            (self.stats["success_rate"] * (self.stats["batched_calls"] - size) + success_rate * size)
            / self.stats["batched_calls"]
        )
    
    async def get_nutrition_response(self, 
                                   prompt: str, 
                                   max_tokens: int = 1000, 
                                   temperature: float = 0.7,
                                   function_name: str = "",
                                   user_id: str = "",
                                   priority: NutritionPriority = NutritionPriority.NORMAL,
                                   use_cache: bool = True,
                                   ai_client_func = None,
                                   nutrition_type: str = "general",
                                   dietary_restrictions: bool = False) -> Tuple[str, str]:
        """Get AI response optimized for nutrition management."""
        
        # Start batch processor if not already running
        if not self.is_running:
            self._start_batch_processor()
        
        # Create nutrition call
        call_id = f"nutrition_{function_name}_{user_id}_{int(time.time() * 1000)}"
        call = NutritionCall(
            id=call_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            function_name=function_name,
            user_id=user_id,
            priority=priority,
            use_cache=use_cache,
            ai_client_func=ai_client_func,
            nutrition_type=nutrition_type,
            dietary_restrictions=dietary_restrictions
        )
        
        # Handle dietary restriction calls
        if dietary_restrictions and self.enable_dietary_priority:
            async with self.batch_lock:
                self.dietary_priority_calls.append(call)
                self.stats["dietary_priority_calls"] += 1
                self.stats["total_calls"] += 1
        else:
            # Add to pending calls
            async with self.batch_lock:
                self.pending_calls.append(call)
                self.stats["total_calls"] += 1
        
        # Wait for result
        try:
            result = await asyncio.wait_for(call.future, timeout=call.timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Nutrition call timeout: {call_id}")
            raise
        except Exception as e:
            logger.error(f"Nutrition call error: {call_id}, {e}")
            raise
    
    def get_nutrition_stats(self) -> Dict[str, Any]:
        """Get nutrition-specific statistics."""
        return {
            **self.stats,
            "nutrition_metrics": self.nutrition_metrics,
            "pending_calls": len(self.pending_calls),
            "dietary_priority_calls": len(self.dietary_priority_calls),
            "processing_batches": len(self.processing_batches),
            "cache_size": len(self.cache),
            "nutrition_cache_size": sum(len(v) for v in self.nutrition_cache.values()),
            "strategy": self.strategy.value,
            "dietary_priority_enabled": self.enable_dietary_priority
        }
    
    async def shutdown(self):
        """Shutdown the nutrition batch manager."""
        self.is_running = False
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
