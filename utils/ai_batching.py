"""
AI Batching Utilities for Cost Optimization
Provides parallel processing, caching, and smart batching for AI calls across all agents.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class AICall:
    """Represents a single AI call request."""
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    function_name: str = ""
    user_id: str = ""
    priority: int = 1  # 1=high, 2=medium, 3=low
    cache_key: Optional[str] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if not self.cache_key:
            self.cache_key = self._generate_cache_key()
    
    def _generate_cache_key(self) -> str:
        """Generate cache key based on prompt and parameters."""
        content = f"{self.prompt}_{self.max_tokens}_{self.temperature}_{self.function_name}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class AIResponse:
    """Represents an AI response."""
    content: str
    model_used: str
    tokens_used: int = 0
    cost: float = 0.0
    response_time: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None

class AIBatchManager:
    """Manages batching of AI calls for cost optimization."""
    
    def __init__(self, max_batch_size: int = 5, max_wait_time: float = 0.5):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_calls: List[AICall] = []
        self.batch_lock = asyncio.Lock()
        self.cache: Dict[str, AIResponse] = {}
        self.cache_ttl: Dict[str, float] = {}
        self.stats = {
            "total_calls": 0,
            "batched_calls": 0,
            "cache_hits": 0,
            "total_savings": 0.0,
            "average_response_time": 0.0
        }
    
    async def get_ai_response(self, 
                            prompt: str, 
                            max_tokens: int = 1000, 
                            temperature: float = 0.7,
                            function_name: str = "",
                            user_id: str = "",
                            priority: int = 1,
                            use_cache: bool = True,
                            ai_client_func = None) -> Tuple[str, str]:
        """
        Get AI response with batching support.
        
        Args:
            prompt: The prompt to send to AI
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            function_name: Name of the calling function (for monitoring)
            user_id: User ID (for caching)
            priority: Priority level (1=high, 2=medium, 3=low)
            use_cache: Whether to use caching
            ai_client_func: Function to call for AI response
            
        Returns:
            Tuple of (response_content, model_used)
        """
        if not ai_client_func:
            raise ValueError("ai_client_func is required")
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(prompt, max_tokens, temperature, function_name)
            cached_response = await self._get_from_cache(cache_key)
            if cached_response:
                self.stats["cache_hits"] += 1
                return cached_response.content, cached_response.model_used
        
        # Create AI call
        ai_call = AICall(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            function_name=function_name,
            user_id=user_id,
            priority=priority
        )
        
        # For high priority calls, process immediately
        if priority == 1:
            return await self._process_single_call(ai_call, ai_client_func)
        
        # Add to batch
        async with self.batch_lock:
            self.pending_calls.append(ai_call)
            
            # Process batch if it's full or if we've waited long enough
            if (len(self.pending_calls) >= self.max_batch_size or 
                (self.pending_calls and time.time() - self.pending_calls[0].created_at >= self.max_wait_time)):
                calls_to_process = self.pending_calls.copy()
                self.pending_calls.clear()
                
                # Process batch in parallel
                return await self._process_batch(calls_to_process, ai_client_func)
        
        # If we get here, the call is still pending
        # Wait for it to be processed
        return await self._wait_for_call_completion(ai_call, ai_client_func)
    
    async def _process_single_call(self, ai_call: AICall, ai_client_func) -> Tuple[str, str]:
        """Process a single AI call immediately."""
        start_time = time.time()
        
        try:
            response, model = await ai_client_func(
                ai_call.prompt, 
                ai_call.max_tokens, 
                ai_call.temperature
            )
            
            response_time = time.time() - start_time
            self.stats["total_calls"] += 1
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_calls"] - 1) + response_time) / 
                self.stats["total_calls"]
            )
            
            # Cache the response
            await self._cache_response(ai_call.cache_key, response, model, response_time)
            
            return response, model
            
        except Exception as e:
            logger.error(f"Single AI call failed: {e}")
            return f"AI call failed: {str(e)}", "error"
    
    async def _process_batch(self, calls: List[AICall], ai_client_func) -> Tuple[str, str]:
        """Process a batch of AI calls in parallel."""
        if not calls:
            return "No calls to process", "none"
        
        start_time = time.time()
        
        try:
            # Create tasks for parallel execution
            tasks = []
            for call in calls:
                task = asyncio.create_task(
                    ai_client_func(call.prompt, call.max_tokens, call.temperature)
                )
                tasks.append((call, task))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results
            for i, (call, result) in enumerate(zip(calls, results)):
                if isinstance(result, Exception):
                    logger.error(f"Batch call failed for {call.function_name}: {result}")
                    # Cache error response
                    await self._cache_response(
                        call.cache_key, 
                        f"AI call failed: {str(result)}", 
                        "error", 
                        0
                    )
                else:
                    response, model = result
                    response_time = time.time() - start_time
                    await self._cache_response(call.cache_key, response, model, response_time)
            
            # Update stats
            self.stats["total_calls"] += len(calls)
            self.stats["batched_calls"] += len(calls)
            
            # Return the first result (for backward compatibility)
            if results and not isinstance(results[0], Exception):
                return results[0]
            else:
                return "Batch processing completed", "batch"
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return f"Batch processing failed: {str(e)}", "error"
    
    async def _wait_for_call_completion(self, ai_call: AICall, ai_client_func) -> Tuple[str, str]:
        """Wait for a call to be completed by batch processing."""
        # This is a simplified implementation
        # In a real implementation, you'd use a more sophisticated waiting mechanism
        await asyncio.sleep(0.1)  # Small delay to allow batch processing
        
        # Check if the call was processed
        cached_response = await self._get_from_cache(ai_call.cache_key)
        if cached_response:
            return cached_response.content, cached_response.model_used
        
        # If not cached, process immediately
        return await self._process_single_call(ai_call, ai_client_func)
    
    async def _get_from_cache(self, cache_key: str) -> Optional[AIResponse]:
        """Get response from cache if available and not expired."""
        if cache_key not in self.cache:
            return None
        
        # Check TTL
        if cache_key in self.cache_ttl and time.time() > self.cache_ttl[cache_key]:
            del self.cache[cache_key]
            del self.cache_ttl[cache_key]
            return None
        
        return self.cache[cache_key]
    
    async def _cache_response(self, cache_key: str, content: str, model: str, response_time: float):
        """Cache an AI response."""
        response = AIResponse(
            content=content,
            model_used=model,
            response_time=response_time,
            cache_hit=False
        )
        
        self.cache[cache_key] = response
        # Set TTL based on function type
        ttl = self._get_cache_ttl(cache_key)
        self.cache_ttl[cache_key] = time.time() + ttl
    
    def _get_cache_ttl(self, cache_key: str) -> float:
        """Get cache TTL based on function type."""
        # Different TTLs for different types of responses
        if "meal_plan" in cache_key:
            return 24 * 3600  # 24 hours
        elif "workout_plan" in cache_key:
            return 12 * 3600  # 12 hours
        elif "nutrition_summary" in cache_key:
            return 6 * 3600   # 6 hours
        elif "general" in cache_key:
            return 2 * 3600   # 2 hours
        else:
            return 3600       # 1 hour default
    
    def _generate_cache_key(self, prompt: str, max_tokens: int, temperature: float, function_name: str) -> str:
        """Generate cache key for a request."""
        content = f"{prompt}_{max_tokens}_{temperature}_{function_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "pending_calls": len(self.pending_calls),
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(self.stats["total_calls"], 1) * 100
            )
        }
    
    async def clear_cache(self):
        """Clear all cached responses."""
        self.cache.clear()
        self.cache_ttl.clear()
        logger.info("AI batching cache cleared")

# Global batch manager instance
batch_manager = AIBatchManager()
