"""
Nutrition Database Service for Nutrition Agent.

This service provides direct database access to nutrition database for food search and macro verification.
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

# Fix import path to use absolute import
from utils.config import get_nutrition_db_uri

logger = logging.getLogger(__name__)

class NutritionDatabaseService:
    """Service for interacting with nutrition database directly."""
    
    def __init__(self):
        self.db_uri = get_nutrition_db_uri()
        if not self.db_uri:
            logger.warning("NUTRITION_DB_URI not configured - nutrition database features will be disabled")
    
    def _get_connection(self):
        """Get database connection with error handling."""
        if not self.db_uri:
            raise ValueError("NUTRITION_DB_URI not configured")
        
        try:
            # Fix postgres:// to postgresql:// if needed
            if self.db_uri.startswith('postgres://'):
                fixed_uri = self.db_uri.replace('postgres://', 'postgresql://', 1)
            else:
                fixed_uri = self.db_uri
            
            conn = psycopg2.connect(
                fixed_uri,
                cursor_factory=RealDictCursor,
                connect_timeout=10
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to nutrition database: {e}")
            raise
    
    def search_foods_by_name(self, name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search foods by name with intelligent ranking.
        
        Args:
            name: Food name to search for
            limit: Maximum number of results to return
            
        Returns:
            List of food dictionaries with nutrition data
        """
        if not self.db_uri:
            logger.warning("NUTRITION_DB_URI not configured - returning empty results")
            return []
        
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Use intelligent ranking: exact matches first, then starts with, then contains
            query = """
                SELECT f.id, f.name, f.description, f.brand_id, f.category_id,
                       f.serving_size, f.serving_unit, f.serving,
                       b.name as brand_name, c.name as category_name,
                       f.created_at,
                       CASE 
                           WHEN LOWER(f.name) = LOWER(%s) THEN 1
                           WHEN LOWER(f.name) LIKE LOWER(%s) THEN 2
                           WHEN LOWER(f.name) LIKE LOWER(%s) THEN 3
                           ELSE 4
                       END as search_rank
                FROM foods f
                LEFT JOIN brands b ON f.brand_id = b.id
                LEFT JOIN categories c ON f.category_id = c.id
                WHERE LOWER(f.name) LIKE LOWER(%s)
                ORDER BY search_rank, f.name
                LIMIT %s
            """
            
            search_term = f"%{name}%"
            params = [name, f"{name}%", f"%{name}%", search_term, limit]
            
            cursor.execute(query, params)
            foods = cursor.fetchall()
            
            # Convert to list of dictionaries
            result = [dict(food) for food in foods]
            
            logger.info(f"Found {len(result)} foods matching name '{name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error searching foods by name: {e}")
            return []
        finally:
            # Ensure proper cleanup
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_food_nutrients(self, food_id: int) -> List[Dict[str, Any]]:
        """
        Get comprehensive nutrition data for a specific food.
        
        Args:
            food_id: ID of the food
            
        Returns:
            List of nutrient dictionaries with amounts
        """
        if not self.db_uri:
            logger.warning("NUTRITION_DB_URI not configured - returning empty results")
            return []
        
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    n.id as nutrient_id,
                    n.name as nutrient_name,
                    n.unit as nutrient_unit,
                    n.description as nutrient_description,
                    fn.amount,
                    fn.created_at
                FROM food_nutrients fn
                JOIN nutrients n ON fn.nutrient_id = n.id
                WHERE fn.food_id = %s
                ORDER BY n.name
            """
            
            cursor.execute(query, (food_id,))
            nutrients = cursor.fetchall()
            
            result = [dict(nutrient) for nutrient in nutrients]
            
            logger.info(f"Found {len(result)} nutrients for food ID {food_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting food nutrients: {e}")
            return []
        finally:
            # Ensure proper cleanup
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def search_ingredients_for_meal(self, ingredient_names: List[str]) -> List[Dict[str, Any]]:
        """
        Search for multiple ingredients to verify they exist in the database.
        
        Args:
            ingredient_names: List of ingredient names to search for
            
        Returns:
            List of found ingredients with their nutrition data
        """
        if not self.db_uri:
            logger.warning("NUTRITION_DB_URI not configured - returning empty results")
            return []
        
        if not ingredient_names:
            return []
        
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            found_ingredients = []
            
            for ingredient_name in ingredient_names:
                # Search for exact or close matches
                query = """
                    SELECT f.id, f.name, f.description, f.serving_size, f.serving_unit, f.serving,
                           b.name as brand_name, c.name as category_name
                    FROM foods f
                    LEFT JOIN brands b ON f.brand_id = b.id
                    LEFT JOIN categories c ON f.category_id = c.id
                    WHERE LOWER(f.name) = LOWER(%s) 
                       OR LOWER(f.name) LIKE LOWER(%s)
                       OR LOWER(f.name) LIKE LOWER(%s)
                    ORDER BY 
                        CASE 
                            WHEN LOWER(f.name) = LOWER(%s) THEN 1
                            WHEN LOWER(f.name) LIKE LOWER(%s) THEN 2
                            ELSE 3
                        END,
                        f.name
                    LIMIT 1
                """
                
                search_term = f"%{ingredient_name}%"
                params = [ingredient_name, f"{ingredient_name}%", search_term, ingredient_name, f"{ingredient_name}%"]
                
                cursor.execute(query, params)
                food = cursor.fetchone()
                
                if food:
                    food_dict = dict(food)
                    # Get nutrition data for this food
                    nutrients = self.get_food_nutrients(food_dict['id'])
                    food_dict['nutrients'] = nutrients
                    found_ingredients.append(food_dict)
                    logger.info(f"Found ingredient: {food_dict['name']}")
                else:
                    logger.warning(f"Ingredient not found in database: {ingredient_name}")
            
            logger.info(f"Found {len(found_ingredients)} out of {len(ingredient_names)} ingredients")
            return found_ingredients
            
        except Exception as e:
            logger.error(f"Error searching ingredients: {e}")
            return []
        finally:
            # Ensure proper cleanup
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def calculate_meal_macros(self, ingredients: List[Dict[str, Any]], servings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate total macros for a meal based on ingredients and servings.
        
        Args:
            ingredients: List of ingredient dictionaries with nutrients
            servings: Optional dictionary mapping ingredient names to serving amounts
            
        Returns:
            Dictionary with total macros and breakdown
        """
        if not ingredients:
            return {
                "total_macros": {},
                "ingredient_breakdown": [],
                "total_calories": 0
            }
        
        try:
            total_macros = {}
            ingredient_breakdown = []
            total_calories = 0
            
            for ingredient in ingredients:
                ingredient_name = ingredient['name']
                serving_size = ingredient.get('serving_size', 1)
                serving_unit = ingredient.get('serving_unit', 'g')
                
                # Get serving multiplier if provided
                serving_multiplier = 1.0
                if servings and ingredient_name in servings:
                    serving_multiplier = servings[ingredient_name] / serving_size
                
                ingredient_macros = {}
                ingredient_calories = 0
                
                # Calculate macros for this ingredient
                for nutrient in ingredient.get('nutrients', []):
                    nutrient_name = nutrient['nutrient_name'].lower()
                    amount = nutrient['amount'] * serving_multiplier
                    unit = nutrient['nutrient_unit']
                    
                    ingredient_macros[nutrient_name] = {
                        'amount': round(amount, 2),
                        'unit': unit
                    }
                    
                    # Add to total macros
                    if nutrient_name not in total_macros:
                        total_macros[nutrient_name] = {'amount': 0, 'unit': unit}
                    total_macros[nutrient_name]['amount'] += amount
                    
                    # Calculate calories (assuming 4 cal/g for protein/carbs, 9 cal/g for fat)
                    if nutrient_name in ['protein', 'carbohydrate', 'total fat']:
                        if nutrient_name in ['protein', 'carbohydrate']:
                            ingredient_calories += amount * 4
                        else:  # fat
                            ingredient_calories += amount * 9
                
                # Round total macros
                for nutrient_name in total_macros:
                    total_macros[nutrient_name]['amount'] = round(total_macros[nutrient_name]['amount'], 2)
                
                ingredient_breakdown.append({
                    'name': ingredient_name,
                    'serving': serving_size * serving_multiplier,
                    'unit': serving_unit,
                    'macros': ingredient_macros,
                    'calories': round(ingredient_calories, 2)
                })
                
                total_calories += ingredient_calories
            
            return {
                "total_macros": total_macros,
                "ingredient_breakdown": ingredient_breakdown,
                "total_calories": round(total_calories, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating meal macros: {e}")
            return {
                "total_macros": {},
                "ingredient_breakdown": [],
                "total_calories": 0,
                "error": str(e)
            }
    
    def get_database_status(self) -> Dict[str, Any]:
        """
        Get nutrition database connection status.
        
        Returns:
            Dictionary with connection status information
        """
        if not self.db_uri:
            return {
                "status": "not_configured",
                "message": "NUTRITION_DB_URI not configured",
                "available": False
            }
        
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Test connection with simple queries
            cursor.execute("SELECT COUNT(*) as count FROM foods")
            food_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM nutrients")
            nutrient_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM food_nutrients")
            food_nutrient_count = cursor.fetchone()['count']
            
            return {
                "status": "connected",
                "message": "Successfully connected to nutrition database",
                "available": True,
                "food_count": food_count,
                "nutrient_count": nutrient_count,
                "food_nutrient_count": food_nutrient_count
            }
            
        except Exception as e:
            logger.error(f"Database status check failed: {e}")
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "available": False
            }
        finally:
            # Ensure proper cleanup
            if cursor:
                cursor.close()
            if conn:
                conn.close()

# Global instance
nutrition_db_service = NutritionDatabaseService()
