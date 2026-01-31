"""
Nutrition Agent Optimizer
ML-based optimization for nutrition management batching and performance.
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    """Optimization targets for nutrition management."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    COST_EFFICIENCY = "cost_efficiency"
    CACHE_HIT_RATE = "cache_hit_rate"
    MEAL_PLANNING_EFFICIENCY = "meal_planning_efficiency"

class NutritionPattern(Enum):
    """Nutrition patterns for optimization."""
    MORNING_MEAL_PLANNING = "morning_meal_planning"  # 6-9 AM
    LUNCH_PLANNING = "lunch_planning"                # 11 AM - 1 PM
    DINNER_PLANNING = "dinner_planning"             # 5-7 PM
    NUTRITION_ANALYSIS = "nutrition_analysis"       # Throughout day
    BATCH_PROCESSING = "batch_processing"           # Off-peak hours

@dataclass
class NutritionMetrics:
    """Nutrition-specific performance metrics."""
    timestamp: float = field(default_factory=time.time)
    nutrition_type: str = "general"
    response_time: float = 0.0
    batch_size: int = 1
    cache_hit: bool = False
    dietary_restrictions: bool = False
    success: bool = True
    cost: float = 0.0
    tokens_used: int = 0
    user_id: str = ""

@dataclass
class OptimizationRule:
    """Optimization rule for nutrition management."""
    pattern: NutritionPattern
    target: OptimizationTarget
    condition: str
    action: str
    priority: int = 1
    enabled: bool = True

class NutritionOptimizer:
    """ML-based optimizer for nutrition management performance."""
    
    def __init__(self, enable_ml_optimization: bool = True, 
                 learning_rate: float = 0.1, history_size: int = 1000):
        self.enable_ml_optimization = enable_ml_optimization
        self.learning_rate = learning_rate
        self.history_size = history_size
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=history_size)
        self.nutrition_patterns: Dict[NutritionPattern, List[NutritionMetrics]] = defaultdict(list)
        
        # Optimization rules
        self.optimization_rules: List[OptimizationRule] = []
        self._initialize_default_rules()
        
        # Performance models
        self.performance_models: Dict[str, Dict[str, Any]] = {}
        self._initialize_performance_models()
        
        # Statistics
        self.stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_improvement": 0.0,
            "optimization_accuracy": 0.0,
            "patterns_detected": 0,
            "rules_triggered": 0
        }
    
    def _initialize_default_rules(self):
        """Initialize default optimization rules for nutrition management."""
        self.optimization_rules = [
            OptimizationRule(
                pattern=NutritionPattern.MORNING_MEAL_PLANNING,
                target=OptimizationTarget.RESPONSE_TIME,
                condition="hour >= 6 and hour < 9",
                action="reduce_batch_size",
                priority=1
            ),
            OptimizationRule(
                pattern=NutritionPattern.LUNCH_PLANNING,
                target=OptimizationTarget.THROUGHPUT,
                condition="hour >= 11 and hour < 13",
                action="increase_batch_size",
                priority=2
            ),
            OptimizationRule(
                pattern=NutritionPattern.DINNER_PLANNING,
                target=OptimizationTarget.MEAL_PLANNING_EFFICIENCY,
                condition="hour >= 17 and hour < 19",
                action="optimize_meal_batching",
                priority=1
            ),
            OptimizationRule(
                pattern=NutritionPattern.NUTRITION_ANALYSIS,
                target=OptimizationTarget.CACHE_HIT_RATE,
                condition="nutrition_type == 'nutrition_analysis'",
                action="increase_cache_ttl",
                priority=2
            ),
            OptimizationRule(
                pattern=NutritionPattern.BATCH_PROCESSING,
                target=OptimizationTarget.COST_EFFICIENCY,
                condition="hour >= 22 or hour < 6",
                action="maximize_batching",
                priority=3
            )
        ]
    
    def _initialize_performance_models(self):
        """Initialize performance prediction models."""
        self.performance_models = {
            "response_time": {
                "base_time": 1.0,
                "batch_factor": 0.2,
                "cache_factor": -0.5,
                "load_factor": 0.3,
                "accuracy": 0.85
            },
            "throughput": {
                "base_throughput": 6.0,
                "batch_factor": 2.2,
                "cache_factor": 1.4,
                "load_factor": -0.7,
                "accuracy": 0.80
            },
            "cost_efficiency": {
                "base_cost": 0.02,
                "batch_factor": -0.4,
                "cache_factor": -0.7,
                "load_factor": 0.2,
                "accuracy": 0.90
            },
            "meal_planning_efficiency": {
                "base_efficiency": 0.8,
                "batch_factor": 0.3,
                "cache_factor": 0.4,
                "load_factor": -0.2,
                "accuracy": 0.75
            }
        }
    
    def record_metrics(self, metrics: NutritionMetrics):
        """Record nutrition metrics for optimization."""
        self.metrics_history.append(metrics)
        
        # Categorize by nutrition pattern
        pattern = self._detect_nutrition_pattern(metrics)
        self.nutrition_patterns[pattern].append(metrics)
        
        # Update performance models
        if self.enable_ml_optimization:
            self._update_performance_models(metrics)
    
    def _detect_nutrition_pattern(self, metrics: NutritionMetrics) -> NutritionPattern:
        """Detect nutrition pattern from metrics."""
        hour = datetime.fromtimestamp(metrics.timestamp).hour
        
        if 6 <= hour < 9:
            return NutritionPattern.MORNING_MEAL_PLANNING
        elif 11 <= hour < 13:
            return NutritionPattern.LUNCH_PLANNING
        elif 17 <= hour < 19:
            return NutritionPattern.DINNER_PLANNING
        elif metrics.nutrition_type == "nutrition_analysis":
            return NutritionPattern.NUTRITION_ANALYSIS
        elif hour >= 22 or hour < 6:
            return NutritionPattern.BATCH_PROCESSING
        else:
            return NutritionPattern.NUTRITION_ANALYSIS  # Default
    
    def _update_performance_models(self, metrics: NutritionMetrics):
        """Update performance prediction models."""
        for model_name, model in self.performance_models.items():
            if model_name == "response_time":
                predicted = self._predict_response_time(metrics)
                actual = metrics.response_time
                self._update_model_accuracy(model, predicted, actual)
            elif model_name == "throughput":
                predicted = self._predict_throughput(metrics)
                actual = 1.0 / max(metrics.response_time, 0.001)  # Approximate throughput
                self._update_model_accuracy(model, predicted, actual)
            elif model_name == "cost_efficiency":
                predicted = self._predict_cost_efficiency(metrics)
                actual = metrics.cost / max(metrics.tokens_used, 1)
                self._update_model_accuracy(model, predicted, actual)
            elif model_name == "meal_planning_efficiency":
                predicted = self._predict_meal_planning_efficiency(metrics)
                actual = 0.9 if metrics.success and metrics.response_time < 2.0 else 0.7
                self._update_model_accuracy(model, predicted, actual)
    
    def _predict_response_time(self, metrics: NutritionMetrics) -> float:
        """Predict response time based on metrics."""
        model = self.performance_models["response_time"]
        
        predicted = (model["base_time"] + 
                    model["batch_factor"] * metrics.batch_size +
                    model["cache_factor"] * (1 if metrics.cache_hit else 0) +
                    model["load_factor"] * self._get_current_load())
        
        return max(predicted, 0.3)  # Minimum response time
    
    def _predict_throughput(self, metrics: NutritionMetrics) -> float:
        """Predict throughput based on metrics."""
        model = self.performance_models["throughput"]
        
        predicted = (model["base_throughput"] + 
                    model["batch_factor"] * metrics.batch_size +
                    model["cache_factor"] * (1 if metrics.cache_hit else 0) +
                    model["load_factor"] * self._get_current_load())
        
        return max(predicted, 1.0)  # Minimum throughput
    
    def _predict_cost_efficiency(self, metrics: NutritionMetrics) -> float:
        """Predict cost efficiency based on metrics."""
        model = self.performance_models["cost_efficiency"]
        
        predicted = (model["base_cost"] + 
                    model["batch_factor"] * metrics.batch_size +
                    model["cache_factor"] * (1 if metrics.cache_hit else 0) +
                    model["load_factor"] * self._get_current_load())
        
        return max(predicted, 0.001)  # Minimum cost
    
    def _predict_meal_planning_efficiency(self, metrics: NutritionMetrics) -> float:
        """Predict meal planning efficiency based on metrics."""
        model = self.performance_models["meal_planning_efficiency"]
        
        predicted = (model["base_efficiency"] + 
                    model["batch_factor"] * metrics.batch_size +
                    model["cache_factor"] * (1 if metrics.cache_hit else 0) +
                    model["load_factor"] * self._get_current_load())
        
        return max(min(predicted, 1.0), 0.0)  # Clamp to 0-1 range
    
    def _get_current_load(self) -> float:
        """Get current system load."""
        if not self.metrics_history:
            return 0.0
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
        
        # Normalize to 0-1 range
        return min(avg_response_time / 3.0, 1.0)
    
    def _update_model_accuracy(self, model: Dict[str, Any], predicted: float, actual: float):
        """Update model accuracy based on prediction vs actual."""
        error = abs(predicted - actual)
        relative_error = error / max(actual, 0.001)
        
        # Update accuracy (exponential moving average)
        model["accuracy"] = (model["accuracy"] * 0.9 + 
                            (1.0 - min(relative_error, 1.0)) * 0.1)
    
    def optimize_batch_size(self, nutrition_type: str, current_load: float, 
                           target: OptimizationTarget = OptimizationTarget.RESPONSE_TIME) -> int:
        """Optimize batch size for nutrition management."""
        if not self.enable_ml_optimization:
            return self._get_default_batch_size(nutrition_type)
        
        # Get relevant metrics
        relevant_metrics = self._get_relevant_metrics(nutrition_type)
        if not relevant_metrics:
            return self._get_default_batch_size(nutrition_type)
        
        # Apply optimization rules
        optimized_size = self._apply_optimization_rules(nutrition_type, current_load, target)
        
        # Update statistics
        self.stats["total_optimizations"] += 1
        if optimized_size != self._get_default_batch_size(nutrition_type):
            self.stats["successful_optimizations"] += 1
            self.stats["rules_triggered"] += 1
        
        return optimized_size
    
    def _get_relevant_metrics(self, nutrition_type: str) -> List[NutritionMetrics]:
        """Get metrics relevant to nutrition type."""
        relevant = []
        for metrics in self.metrics_history:
            if metrics.nutrition_type == nutrition_type:
                relevant.append(metrics)
        return relevant[-50:]  # Last 50 relevant metrics
    
    def _apply_optimization_rules(self, nutrition_type: str, current_load: float, 
                                target: OptimizationTarget) -> int:
        """Apply optimization rules to determine batch size."""
        hour = datetime.now().hour
        base_size = self._get_default_batch_size(nutrition_type)
        
        for rule in sorted(self.optimization_rules, key=lambda x: x.priority):
            if not rule.enabled:
                continue
            
            # Check condition
            if self._evaluate_condition(rule.condition, hour, current_load, nutrition_type):
                # Apply action
                if rule.action == "reduce_batch_size":
                    return max(1, int(base_size * 0.5))
                elif rule.action == "increase_batch_size":
                    return min(12, int(base_size * 1.5))
                elif rule.action == "optimize_meal_batching":
                    return min(10, int(base_size * 1.2))
                elif rule.action == "maximize_batching":
                    return min(15, int(base_size * 2.0))
        
        return base_size
    
    def _evaluate_condition(self, condition: str, hour: int, load: float, nutrition_type: str) -> bool:
        """Evaluate optimization rule condition."""
        try:
            # Simple condition evaluation
            context = {
                "hour": hour,
                "load": load,
                "nutrition_type": nutrition_type,
                "dietary_restrictions": nutrition_type in ["dietary_restriction", "meal_plan"]
            }
            
            # Replace variables in condition
            for var, value in context.items():
                condition = condition.replace(var, str(value))
            
            return eval(condition)
        except:
            return False
    
    def _get_default_batch_size(self, nutrition_type: str) -> int:
        """Get default batch size for nutrition type."""
        defaults = {
            "meal_plan": 8,
            "nutrition_analysis": 6,
            "food_suggestion": 5,
            "dietary_restriction": 3,
            "general": 6
        }
        return defaults.get(nutrition_type, 6)
    
    def optimize_cache_ttl(self, nutrition_type: str, hit_rate: float) -> float:
        """Optimize cache TTL for nutrition type."""
        if not self.enable_ml_optimization:
            return self._get_default_cache_ttl(nutrition_type)
        
        # Adjust TTL based on hit rate
        base_ttl = self._get_default_cache_ttl(nutrition_type)
        
        if hit_rate > 0.8:  # High hit rate
            return base_ttl * 1.5  # Increase TTL
        elif hit_rate < 0.3:  # Low hit rate
            return base_ttl * 0.5  # Decrease TTL
        else:
            return base_ttl
    
    def _get_default_cache_ttl(self, nutrition_type: str) -> float:
        """Get default cache TTL for nutrition type."""
        defaults = {
            "meal_plan": 3600,        # 1 hour
            "nutrition_analysis": 1800,  # 30 minutes
            "food_suggestion": 1800,  # 30 minutes
            "dietary_restriction": 7200,  # 2 hours
            "general": 1800          # 30 minutes
        }
        return defaults.get(nutrition_type, 1800)
    
    def get_optimization_recommendations(self, nutrition_type: str) -> Dict[str, Any]:
        """Get optimization recommendations for nutrition type."""
        relevant_metrics = self._get_relevant_metrics(nutrition_type)
        if not relevant_metrics:
            return {"recommendations": [], "confidence": 0.0}
        
        recommendations = []
        confidence = 0.0
        
        # Analyze patterns
        avg_response_time = statistics.mean([m.response_time for m in relevant_metrics])
        hit_rate = sum(1 for m in relevant_metrics if m.cache_hit) / len(relevant_metrics)
        dietary_restriction_rate = sum(1 for m in relevant_metrics if m.dietary_restrictions) / len(relevant_metrics)
        
        # Generate recommendations
        if avg_response_time > 2.0:
            recommendations.append({
                "type": "reduce_batch_size",
                "reason": "High response time detected",
                "impact": "high"
            })
            confidence += 0.3
        
        if hit_rate < 0.4:
            recommendations.append({
                "type": "increase_cache_ttl",
                "reason": "Low cache hit rate",
                "impact": "medium"
            })
            confidence += 0.2
        
        if dietary_restriction_rate > 0.3:
            recommendations.append({
                "type": "enable_dietary_priority",
                "reason": "High dietary restriction rate",
                "impact": "high"
            })
            confidence += 0.4
        
        if len(relevant_metrics) > 20:
            recommendations.append({
                "type": "enable_ml_optimization",
                "reason": "Sufficient data for ML optimization",
                "impact": "high"
            })
            confidence += 0.3
        
        return {
            "recommendations": recommendations,
            "confidence": min(confidence, 1.0),
            "metrics_analyzed": len(relevant_metrics),
            "avg_response_time": avg_response_time,
            "cache_hit_rate": hit_rate,
            "dietary_restriction_rate": dietary_restriction_rate
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            **self.stats,
            "performance_models": {
                name: {
                    "accuracy": model["accuracy"],
                    "last_updated": time.time()
                }
                for name, model in self.performance_models.items()
            },
            "nutrition_patterns": {
                pattern.value: len(metrics)
                for pattern, metrics in self.nutrition_patterns.items()
            },
            "optimization_rules": len(self.optimization_rules),
            "enabled_rules": sum(1 for rule in self.optimization_rules if rule.enabled),
            "ml_optimization_enabled": self.enable_ml_optimization
        }
