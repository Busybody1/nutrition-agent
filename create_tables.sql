-- Database initialization script for Nutrition Agent
-- Creates all required tables and indexes

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    age INTEGER,
    height_cm FLOAT,
    weight_kg FLOAT,
    primary_goal VARCHAR(50),
    daily_calorie_target INTEGER,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE
);

-- Create foods table (not food_items as expected by nutrition agent)
CREATE TABLE IF NOT EXISTS foods (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    brand_id VARCHAR(255),
    category_id VARCHAR(100) NOT NULL,
    serving_size FLOAT NOT NULL,
    serving_unit VARCHAR(50) NOT NULL,
    serving VARCHAR(100),
    calories FLOAT NOT NULL,
    protein_g FLOAT NOT NULL,
    carbs_g FLOAT NOT NULL,
    fat_g FLOAT NOT NULL,
    fiber_g FLOAT DEFAULT 0,
    sugar_g FLOAT DEFAULT 0,
    source VARCHAR(50) DEFAULT 'user_input',
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create food_logs table with additional serving columns
CREATE TABLE IF NOT EXISTS food_logs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    food_item_id UUID NOT NULL,
    quantity_g FLOAT NOT NULL,
    meal_type VARCHAR(20) NOT NULL,
    consumed_at TIMESTAMP NOT NULL,
    calories FLOAT NOT NULL,
    protein_g FLOAT NOT NULL,
    carbs_g FLOAT NOT NULL,
    fat_g FLOAT NOT NULL,
    serving_size FLOAT,
    serving_unit VARCHAR(50),
    serving VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP NOT NULL
);

-- Create workout_logs table
CREATE TABLE IF NOT EXISTS workout_logs (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    duration_minutes INTEGER NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    calories_burned INTEGER,
    notes TEXT,
    created_at TIMESTAMP NOT NULL
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_foods_name ON foods(name);
CREATE INDEX IF NOT EXISTS idx_foods_category_id ON foods(category_id);
CREATE INDEX IF NOT EXISTS idx_food_logs_user_id ON food_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_food_logs_consumed_at ON food_logs(consumed_at);
CREATE INDEX IF NOT EXISTS idx_workout_logs_user_id ON workout_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_workout_logs_started_at ON workout_logs(started_at);

-- Insert some sample data for testing
INSERT INTO foods (id, name, brand_id, category_id, serving_size, serving_unit, serving, calories, protein_g, carbs_g, fat_g, created_at, updated_at) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Chicken Breast', 'brand_001', 'protein', 100.0, 'g', '1 serving', 165.0, 31.0, 0.0, 3.6, NOW(), NOW()),
('550e8400-e29b-41d4-a716-446655440002', 'Brown Rice', 'brand_002', 'grains', 100.0, 'g', '1/2 cup', 111.0, 2.6, 23.0, 0.9, NOW(), NOW()),
('550e8400-e29b-41d4-a716-446655440003', 'Broccoli', 'brand_003', 'vegetables', 100.0, 'g', '1 cup', 34.0, 2.8, 7.0, 0.4, NOW(), NOW()),
('550e8400-e29b-41d4-a716-446655440004', 'Salmon', 'brand_004', 'protein', 100.0, 'g', '1 fillet', 208.0, 25.0, 0.0, 12.0, NOW(), NOW()),
('550e8400-e29b-41d4-a716-446655440005', 'Sweet Potato', 'brand_005', 'vegetables', 100.0, 'g', '1 medium', 86.0, 1.6, 20.0, 0.1, NOW(), NOW())
ON CONFLICT (id) DO NOTHING; 