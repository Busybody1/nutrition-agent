#!/usr/bin/env python3
"""
Comprehensive test script for Nutrition Agent endpoints
"""

import requests
import json
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NutritionAgentTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def test_endpoint(self, method: str, endpoint: str, data: Dict[str, Any] = None, expected_status: int = 200) -> Dict[str, Any]:
        """Test a single endpoint and return results."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            duration = time.time() - start_time
            success = response.status_code == expected_status
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "success": success,
                "duration": round(duration, 2),
                "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            }
            
            if success:
                logger.info(f"✓ {method} {endpoint} - {response.status_code} ({duration:.2f}s)")
            else:
                logger.error(f"✗ {method} {endpoint} - {response.status_code} ({duration:.2f}s)")
                logger.error(f"  Response: {result['response']}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": None,
                "expected_status": expected_status,
                "success": False,
                "duration": round(duration, 2),
                "error": str(e),
                "response": None
            }
            logger.error(f"✗ {method} {endpoint} - ERROR ({duration:.2f}s): {e}")
            return result
    
    def run_all_tests(self):
        """Run all endpoint tests."""
        logger.info("Starting comprehensive Nutrition Agent endpoint tests...")
        
        # Health checks
        self.test_endpoint("GET", "/health")
        self.test_endpoint("GET", "/health/detailed")
        self.test_endpoint("GET", "/debug/database")
        
        # Food endpoints
        self.test_endpoint("GET", "/foods/count")
        self.test_endpoint("GET", "/foods/1")
        
        # Tool execution
        tool_tests = [
            {
                "tool": "search_food_by_name",
                "params": {"name": "chicken"}
            },
            {
                "tool": "get_food_nutrition",
                "params": {"food_id": "550e8400-e29b-41d4-a716-446655440001"}
            },
            {
                "tool": "log_food_to_calorie_log",
                "params": {
                    "entry": {
                        "id": "550e8400-e29b-41d4-a716-446655440010",
                        "user_id": "550e8400-e29b-41d4-a716-446655440000",
                        "food_item_id": "550e8400-e29b-41d4-a716-446655440001",
                        "quantity_g": 100.0,
                        "meal_type": "lunch",
                        "consumed_at": "2024-01-01T12:00:00Z",
                        "actual_nutrition": {
                            "calories": 165.0,
                            "protein_g": 31.0,
                            "carbs_g": 0.0,
                            "fat_g": 3.6
                        },
                        "notes": "Test food log"
                    }
                }
            },
            {
                "tool": "get_user_calorie_history",
                "params": {"user_id": "550e8400-e29b-41d4-a716-446655440000"}
            },
            {
                "tool": "search_food_fuzzy",
                "params": {"name": "rice"}
            },
            {
                "tool": "calculate_calories",
                "params": {"foods": [{"name": "chicken", "quantity": 100}]}
            },
            {
                "tool": "get_meal_suggestions",
                "params": {"user_id": "550e8400-e29b-41d4-a716-446655440000", "meal_type": "lunch"}
            },
            {
                "tool": "get_nutrition_recommendations",
                "params": {"user_id": "550e8400-e29b-41d4-a716-446655440000"}
            },
            {
                "tool": "track_nutrition_goals",
                "params": {
                    "user_id": "550e8400-e29b-41d4-a716-446655440000",
                    "goal_type": "calories",
                    "target_value": 2000.0
                }
            }
        ]
        
        for test in tool_tests:
            self.test_endpoint("POST", "/execute-tool", {"tool": test["tool"], "params": test["params"]})
        
        # Direct endpoint tests
        self.test_endpoint("POST", "/meal-plan", {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "daily_calories": 2000,
            "meal_count": 3,
            "dietary_restrictions": []
        })
        
        self.test_endpoint("POST", "/calculate-calories", {
            "foods": [
                {"name": "chicken breast", "quantity": 100},
                {"name": "brown rice", "quantity": 50}
            ]
        })
        
        self.test_endpoint("POST", "/nutrition-recommendations", {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "history": [],
            "user_profile": {"daily_calorie_target": 2000}
        })
        
        self.test_endpoint("POST", "/fuzzy-search", {"name": "chicken"})
        
        self.test_endpoint("POST", "/nutrition-goals", {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "goal_type": "calories",
            "target_value": 2000.0,
            "current_value": 1500.0
        })
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get("success", False))
        failed_tests = total_tests - successful_tests
        
        logger.info("\n" + "="*60)
        logger.info("NUTRITION AGENT COMPREHENSIVE TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success rate: {(successful_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            logger.info("\nFailed tests:")
            for result in self.test_results:
                if not result.get("success", False):
                    logger.error(f"  {result['method']} {result['endpoint']} - {result.get('error', result.get('response', 'Unknown error'))}")
        
        logger.info("="*60)

def main():
    """Main test function."""
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    
    tester = NutritionAgentTester(base_url)
    tester.run_all_tests()

if __name__ == "__main__":
    main() 