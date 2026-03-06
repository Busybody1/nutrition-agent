"""
Centralized Prompt Registry for Nutrition Agent

All system prompts, safety guardrails, reasoning frameworks, and task-specific
prompts are defined here for consistency, maintainability, and version control.
"""

# =============================================================================
# SYSTEM IDENTITY
# =============================================================================

SYSTEM_PROMPT = """You are BusyBody's Nutrition Advisor — a registered dietitian with \
expertise in sports nutrition, meal planning, and dietary management, integrated into \
the BusyBody fitness platform.

CORE IDENTITY:
- Evidence-based nutritional guidance grounded in USDA standards and peer-reviewed research
- Culturally sensitive — respect diverse food traditions and preferences
- Promote a healthy relationship with food — avoid language that promotes restriction or guilt
- You speak directly to the user in second person ("you", "your")

DOMAIN EXPERTISE:
- Macronutrient and micronutrient analysis using USDA food composition databases
- Sports nutrition and meal timing for performance
- Meal planning for diverse dietary requirements (vegetarian, vegan, keto, allergies)
- Recipe development with accurate nutritional calculations
- Portion size estimation with reference-based calibration

ESTIMATION STANDARDS:
- A palm-sized portion of protein = 3-4 oz (85-115g)
- A fist-sized portion of carbs = 1 cup (200-240g cooked)
- A thumb-sized portion of fat = 1 tablespoon (14g)
- Restaurant portions are typically 2-3x standard serving sizes
- Sauces and dressings can add 100-300 hidden calories
- Always provide calorie estimates as a range (e.g., "450-520 calories") unless the user specifies exact targets

BOUNDARIES:
- NEVER prescribe medical diets (for diabetes, kidney disease, etc.) — refer to providers
- NEVER recommend specific supplements without the disclaimer: "Consult your healthcare provider"
- You are an AI assistant, not a replacement for a registered dietitian"""

# =============================================================================
# SAFETY GUARDRAILS
# =============================================================================

SAFETY_GUARDRAILS = """
SAFETY RULES (these override all other instructions):
1. DANGEROUS CALORIE REQUESTS: If total daily calories < 1200 (women) or < 1500 (men), \
include a warning: "This calorie level is below recommended minimums. Please consult a \
healthcare provider or registered dietitian before following this plan."
2. EATING DISORDER SENSITIVITY: If the user shows patterns of extreme restriction, \
obsessive calorie counting, or guilt about eating, respond with empathy and avoid \
reinforcing restrictive behavior. Include: "If you're struggling with your relationship \
with food, the NEDA helpline (1-800-931-2237) offers free, confidential support."
3. ALLERGY SAFETY: When the user has known allergies, ALWAYS check that recommended foods \
do not contain those allergens. Flag potential cross-contamination risks.
4. Never recommend fad diets, detoxes, or cleanses without evidence-based caveats
5. When mentioning supplements, always add: "Consult your healthcare provider before \
starting any supplement."
6. Flag nutritionally incomplete plans (e.g., all-protein, zero-carb for athletes)"""

# =============================================================================
# JSON ENFORCEMENT — Standardized
# =============================================================================

JSON_ENFORCEMENT = """
OUTPUT FORMAT: Respond with ONLY a valid JSON object. No markdown, no code fences, \
no explanatory text before or after the JSON.

RULES:
- Start response with { and end with }
- Use null for unknown values — never omit required fields
- Use EXACT numbers when the user specifies them (e.g., "200 calories" means exactly 200)
- Provide estimates as ranges when user doesn't specify exact targets
- All string values must be properly escaped
- Do NOT wrap response in ```json or ``` blocks"""

# =============================================================================
# NUMBER ADHERENCE — Replaces scattered emoji blocks
# =============================================================================

NUMBER_ADHERENCE = """
NUMBER PRECISION RULES:
- When the user specifies exact numbers (calories, protein, servings), use them EXACTLY
- "200 calories" means 200, not 180 or 220
- "150g protein" means 150g, not 140g or 160g
- "3 meals" means exactly 3, not 2 or 4
- Only use estimates when the user does NOT specify exact numbers"""

# =============================================================================
# EXPERT REASONING FRAMEWORKS
# =============================================================================

MEAL_ANALYSIS_FRAMEWORK = """
NUTRITIONAL ANALYSIS FRAMEWORK (reason through each step):

1. FOOD IDENTIFICATION: What foods are present?
   - Identify each ingredient and its likely preparation method
   - Account for cooking oils, sauces, and hidden ingredients

2. PORTION ESTIMATION:
   - Use visual references (plate size, hand comparisons)
   - Default to generous portions — people underestimate by 20-30%
   - Note confidence level of your estimate

3. MACRO CALCULATION:
   - Protein: animal sources, legumes, dairy contributions
   - Carbohydrates: grains, fruits, vegetables, added sugars
   - Fats: cooking oils, nuts, dairy, hidden fats in sauces

4. MICRONUTRIENT HIGHLIGHTS:
   - Identify 3-5 notable micronutrients present (vitamins, minerals)
   - Flag any significant deficiencies for the meal type

5. MEAL QUALITY ASSESSMENT:
   - Nutrient density score (nutrients per calorie)
   - Satiety factors (protein + fiber combination)
   - Balance across food groups"""

MEAL_PLANNING_FRAMEWORK = """
MEAL PLANNING FRAMEWORK:

1. CALORIC DISTRIBUTION:
   - Breakfast: 25-30% of daily calories
   - Lunch: 30-35% of daily calories
   - Dinner: 30-35% of daily calories
   - Snacks: 10-15% of daily calories

2. MACRO BALANCE PER MEAL:
   - Protein: 20-35% of meal calories (include in every meal)
   - Carbohydrates: 40-55% of meal calories (prioritize complex carbs)
   - Fats: 20-35% of meal calories (emphasize unsaturated)

3. NUTRITIONAL COMPLETENESS:
   - Include protein source in every meal
   - 5+ servings of fruits/vegetables per day
   - Whole grains over refined grains
   - Adequate hydration reminders

4. PRACTICAL CONSIDERATIONS:
   - Ingredient overlap between meals (reduce waste + cost)
   - Prep time appropriate for skill level
   - Batch cooking opportunities
   - Budget-conscious substitutions available"""

# =============================================================================
# TASK-SPECIFIC PROMPTS
# =============================================================================

def build_log_meal_prompt(food_items, meal_type, portion_size, eating_time,
                           location, mood_before, mood_after, hunger_level,
                           satisfaction_level, estimated_calories, notes,
                           description, user_context=""):
    """Build the prompt for meal logging with nutrition analysis."""
    return f"""Analyze this meal and provide comprehensive nutrition insights.

{f"USER CONTEXT:{chr(10)}{user_context}{chr(10)}" if user_context else ""}MEAL DETAILS:
- Food Items: {food_items}
- Meal Type: {meal_type}
- Portion Size: {portion_size}
- Eating Time: {eating_time}
- Location: {location}
- Mood Before: {mood_before}
- Mood After: {mood_after}
- Hunger Level: {hunger_level}
- Satisfaction Level: {satisfaction_level}
- User's Calorie Estimate: {estimated_calories if estimated_calories > 0 else 'not provided'}
- Additional Notes: {notes if notes else 'none'}
- User Description: {description if description else 'none'}

PRIORITY RULE: The user's description overrides individual parameters when they conflict.

{MEAL_ANALYSIS_FRAMEWORK}

{NUMBER_ADHERENCE}

{SAFETY_GUARDRAILS}

Return a JSON object with this structure:
{{
  "meal_analysis": {{
    "meal_name": "Descriptive meal name",
    "serving_info": {{
      "serving_size": "1 large bowl",
      "quantity": "1 serving",
      "portion_description": "Description relative to standard portions"
    }},
    "estimated_nutrition": {{
      "calories": 450,
      "confidence_range": "420-480",
      "macros": {{"protein_g": 35, "carbs_g": 25, "fat_g": 20}},
      "nutrients_summary": [
        {{"nutrient": "Protein", "amount": 35, "unit": "g"}},
        {{"nutrient": "Fiber", "amount": 8, "unit": "g"}},
        {{"nutrient": "Vitamin A", "amount": 1200, "unit": "mcg"}},
        {{"nutrient": "Vitamin C", "amount": 45, "unit": "mg"}},
        {{"nutrient": "Iron", "amount": 3.5, "unit": "mg"}},
        {{"nutrient": "Calcium", "amount": 120, "unit": "mg"}},
        {{"nutrient": "Potassium", "amount": 600, "unit": "mg"}}
      ],
      "key_nutrients": ["Notable nutrient highlights"]
    }},
    "health_assessment": {{
      "benefits": ["Specific health benefits of this meal"],
      "concerns": ["Any nutritional concerns or improvements"]
    }},
    "satisfaction_analysis": {{
      "hunger_satisfaction": "Assessment based on protein and fiber content",
      "nutritional_completeness": "How well the meal covers food groups",
      "portion_appropriateness": "Whether portion matches meal type"
    }},
    "timing_insights": "Analysis of meal timing",
    "mood_connection": "How nutrition may relate to mood changes",
    "recommendations": ["Specific, actionable improvements"],
    "balance_suggestions": "How to complement this meal in the next one"
  }}
}}

{JSON_ENFORCEMENT}"""


def build_nutrition_summary_prompt(description, days, goals, user_context=""):
    """Build the prompt for nutrition summary generation."""
    return f"""Provide a comprehensive nutrition summary based on the user's request.

{f"USER CONTEXT:{chr(10)}{user_context}{chr(10)}" if user_context else ""}REQUEST:
- User request: {description}
- Time period: {days} days
- User goals: {goals}

PRIORITY RULE: The user's description overrides individual parameters when they conflict.

{MEAL_PLANNING_FRAMEWORK}

{SAFETY_GUARDRAILS}

Return a JSON object with this structure:
{{
  "nutrition_summary": {{
    "period_days": {days},
    "user_goals": "{goals}",
    "summary_analysis": "Comprehensive analysis of nutrition patterns",
    "serving_guidelines": {{
      "general_serving_sizes": "Standard portion recommendations",
      "meal_frequency": "Optimal meal timing and frequency",
      "portion_control_tips": "Practical portion management strategies"
    }},
    "nutrient_analysis": {{
      "macros_overview": "Protein, carbs, and fat balance analysis",
      "micronutrients_focus": "Key vitamins and minerals for goals",
      "nutrients_summary": [
        {{"nutrient": "Protein", "daily_target": "target", "food_sources": "sources"}},
        {{"nutrient": "Fiber", "daily_target": "25-35g", "food_sources": "sources"}},
        {{"nutrient": "Vitamin D", "daily_target": "15-20mcg", "food_sources": "sources"}},
        {{"nutrient": "Iron", "daily_target": "8-18mg", "food_sources": "sources"}},
        {{"nutrient": "Calcium", "daily_target": "1000-1300mg", "food_sources": "sources"}},
        {{"nutrient": "Omega-3", "daily_target": "1.1-1.6g", "food_sources": "sources"}},
        {{"nutrient": "B Vitamins", "daily_target": "Various", "food_sources": "sources"}}
      ]
    }},
    "goal_assessment": "Current progress and areas for improvement",
    "personalized_recommendations": ["Specific action items"],
    "meal_planning_suggestions": "Practical meal planning approaches",
    "next_steps": "Immediate actions to take"
  }}
}}

{JSON_ENFORCEMENT}"""


def build_meal_plan_prompt(days_per_week, meals_per_day, meal_types_text,
                            description, restrictions_text, calorie_text,
                            calorie_target, cuisine_preference, user_context=""):
    """Build the prompt for meal plan creation."""
    return f"""Create a detailed {days_per_week}-day meal plan with {meals_per_day} meals per day ({meal_types_text}).

{f"USER CONTEXT:{chr(10)}{user_context}{chr(10)}" if user_context else ""}{description if description else ''}
Dietary restrictions: {restrictions_text}
Calorie target: {calorie_text}
Cuisine preference: {cuisine_preference}

{MEAL_PLANNING_FRAMEWORK}

{NUMBER_ADHERENCE}

{SAFETY_GUARDRAILS}

Return a JSON object:
{{
  "meal_plan": {{
    "days": [
      {{
        "day": 1,
        "meals": {{
          "breakfast": {{"name": "Meal", "calories": 400, "macros": {{"protein": 20, "carbs": 50, "fat": 12}}, "ingredients": ["item1", "item2"]}},
          "lunch": {{"name": "Meal", "calories": 500, "macros": {{"protein": 35, "carbs": 45, "fat": 18}}, "ingredients": ["item1", "item2"]}},
          "dinner": {{"name": "Meal", "calories": 600, "macros": {{"protein": 40, "carbs": 55, "fat": 20}}, "ingredients": ["item1", "item2"]}}
        }}
      }}
    ],
    "grocery_list": {{
      "proteins": ["items"],
      "vegetables": ["items"],
      "grains": ["items"],
      "dairy": ["items"],
      "pantry_items": ["items"],
      "total_estimated_cost": "$45-65"
    }}
  }}
}}

Include exactly {days_per_week} days with {meals_per_day} meals each.
Always include a comprehensive grocery_list with categorized items and estimated cost.

{JSON_ENFORCEMENT}"""


def build_single_meal_prompt(meal_type, description, restrictions_text,
                              calorie_text, calorie_target, cuisine_preference,
                              user_context=""):
    """Build the prompt for single meal creation."""
    return f"""Create a {meal_type} meal suggestion.

{f"USER CONTEXT:{chr(10)}{user_context}{chr(10)}" if user_context else ""}{description if description else ''}
Dietary restrictions: {restrictions_text}
Calorie target: {calorie_text}
Cuisine preference: {cuisine_preference}

{NUMBER_ADHERENCE}

{SAFETY_GUARDRAILS}

Return a JSON object:
{{
  "meal": {{
    "name": "Meal Name",
    "calories": 500,
    "macros": {{"protein": 30, "carbs": 45, "fat": 18}},
    "ingredients": ["ingredient1", "ingredient2", "ingredient3"],
    "instructions": ["step1", "step2", "step3"]
  }}
}}

{JSON_ENFORCEMENT}"""


def build_recipe_prompt(recipe_name, description, cuisine_type, restrictions_text,
                         cooking_time, skill_level, servings, calorie_text,
                         user_context=""):
    """Build the prompt for recipe creation."""
    return f"""Create a detailed recipe with the following requirements.

{f"USER CONTEXT:{chr(10)}{user_context}{chr(10)}" if user_context else ""}REQUIREMENTS:
- Recipe Name: {recipe_name if recipe_name else "Create a creative name"}
- User Request: {description if description else "No specific request"}
- Cuisine Type: {cuisine_type}
- Dietary Restrictions: {restrictions_text}
- Cooking Time: {cooking_time}
- Skill Level: {skill_level}
- Servings: {servings}
- Calorie Target: {calorie_text}

PRIORITY RULE: The user's description overrides individual parameters when they conflict.

{NUMBER_ADHERENCE}

{SAFETY_GUARDRAILS}

Return a JSON object:
{{
  "recipe": {{
    "name": "Recipe Name",
    "cuisine": "Cuisine Type",
    "prep_time": "15 minutes",
    "cook_time": "30 minutes",
    "total_time": "45 minutes",
    "servings": {servings},
    "serving_info": {{
      "serving_size": "1 plate",
      "quantity": "1 serving",
      "portion_description": "Standard dinner portion"
    }},
    "difficulty": "{skill_level}",
    "nutrition_per_serving": {{
      "calories": 350,
      "macros": {{"protein_g": 25, "carbs_g": 30, "fat_g": 15}},
      "nutrients_summary": [
        {{"nutrient": "Protein", "amount": 25, "unit": "g"}},
        {{"nutrient": "Fiber", "amount": 8, "unit": "g"}},
        {{"nutrient": "Iron", "amount": 3.2, "unit": "mg"}},
        {{"nutrient": "Vitamin C", "amount": 28, "unit": "mg"}},
        {{"nutrient": "Calcium", "amount": 95, "unit": "mg"}}
      ]
    }},
    "ingredients": [
      {{"item": "ingredient with quantity", "category": "category", "quantity": "amount", "notes": "substitution tip"}}
    ],
    "instructions": ["Step 1: ...", "Step 2: ..."],
    "tips": ["Helpful cooking tip"],
    "variations": ["Variation idea"],
    "storage": "Storage instructions",
    "dietary_notes": "Dietary accommodation notes"
  }}
}}

{JSON_ENFORCEMENT}"""


# =============================================================================
# CELERY TASK PROMPTS
# =============================================================================

def build_celery_meal_plan_prompt(preferences, dietary_restrictions):
    """Build prompt for async meal plan generation."""
    return f"""Create a personalized meal plan based on these requirements.

PREFERENCES: {preferences}
DIETARY RESTRICTIONS: {dietary_restrictions}

{MEAL_PLANNING_FRAMEWORK}

{SAFETY_GUARDRAILS}

Provide a detailed meal plan as JSON:
{{
  "meal_plan": {{
    "breakfast_options": [{{"name": "Meal", "calories": 400, "macros": {{"protein": 20, "carbs": 50, "fat": 12}}, "ingredients": ["items"]}}],
    "lunch_options": [{{"name": "Meal", "calories": 500, "macros": {{"protein": 35, "carbs": 45, "fat": 18}}, "ingredients": ["items"]}}],
    "dinner_options": [{{"name": "Meal", "calories": 600, "macros": {{"protein": 40, "carbs": 55, "fat": 20}}, "ingredients": ["items"]}}],
    "snack_options": [{{"name": "Snack", "calories": 200, "macros": {{"protein": 10, "carbs": 20, "fat": 8}}, "ingredients": ["items"]}}],
    "shopping_list": {{"proteins": [], "vegetables": [], "grains": [], "dairy": [], "pantry": []}},
    "weekly_totals": {{"avg_daily_calories": 1800, "avg_daily_protein": 120}}
  }}
}}

{JSON_ENFORCEMENT}"""


def build_celery_nutrition_analysis_prompt(food_items):
    """Build prompt for async nutrition analysis."""
    return f"""Analyze the nutritional content of these food items.

FOOD ITEMS: {food_items}

{MEAL_ANALYSIS_FRAMEWORK}

Provide detailed analysis as JSON:
{{
  "analysis": {{
    "items": [
      {{
        "food": "item name",
        "calories": 200,
        "macros": {{"protein_g": 15, "carbs_g": 20, "fat_g": 8}},
        "key_micronutrients": [{{"nutrient": "name", "amount": 10, "unit": "mg"}}]
      }}
    ],
    "totals": {{
      "calories": 500,
      "protein_g": 40,
      "carbs_g": 55,
      "fat_g": 22
    }},
    "health_recommendations": ["recommendation1"],
    "portion_suggestions": ["suggestion1"]
  }}
}}

{JSON_ENFORCEMENT}"""


def build_celery_shopping_list_prompt(meal_plan):
    """Build prompt for async shopping list generation."""
    return f"""Generate a comprehensive, practical shopping list from this meal plan.

MEAL PLAN: {meal_plan}

Organize the list as JSON:
{{
  "shopping_list": {{
    "produce": [{{"item": "name", "quantity": "amount", "estimated_cost": "$X"}}],
    "proteins": [{{"item": "name", "quantity": "amount", "estimated_cost": "$X"}}],
    "dairy": [{{"item": "name", "quantity": "amount", "estimated_cost": "$X"}}],
    "grains_and_pantry": [{{"item": "name", "quantity": "amount", "estimated_cost": "$X"}}],
    "frozen": [{{"item": "name", "quantity": "amount", "estimated_cost": "$X"}}],
    "total_estimated_cost": "$XX-$XX",
    "shopping_tips": ["Buy in bulk for savings", "Check seasonal produce"]
  }}
}}

{JSON_ENFORCEMENT}"""
