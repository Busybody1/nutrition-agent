# Nutrition AI Agent Setup Instructions

## 🚀 Quick Start

### 1. Environment Setup

Copy the environment template and configure your variables:
```bash
cp env_template.txt .env
# Edit .env with your actual values
```

**Required Environment Variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `GROQ_API_KEY` - Your Groq API key

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Database Setup
```bash
# The agent will automatically create tables on first run
# Make sure your PostgreSQL database is running and accessible
```

### 4. Run the Agent
```bash
python main.py
# Or with uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8001
```

## 🔧 Detailed Configuration

### Database Configuration

#### Option 1: Single Database (Recommended for Development)
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/fitness_db
```

#### Option 2: Multi-Database Setup (Production)
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/fitness_db
NUTRITION_DB_URI=postgresql://username:password@localhost:5432/nutrition_db
```

### AI Service Configuration

#### Groq API Setup
1. Sign up at [groq.com](https://groq.com)
2. Get your API key
3. Set in environment:
```bash
GROQ_API_KEY=your_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama3-70b-8192
```

### Redis Configuration (Optional but Recommended)
```bash
REDIS_URL=redis://localhost:6379
# Or individual settings:
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

## 🗄️ Database Schema

The agent will automatically create these tables:

### Core Tables
- `users` - User accounts and profiles
- `foods` - Food items with nutrition data
- `food_logs` - User food consumption logs
- `user_sessions` - User session management
- `agent_conversation_sessions` - Conversation state
- `inter_agent_communications` - Cross-agent communication

### Nutrition Tables
- `food_nutrients` - Detailed nutrient information
- `nutrients` - Nutrient definitions
- `brands` - Food brand information
- `categories` - Food categories

## 🧪 Testing the Agent

### 1. Health Check
```bash
curl http://localhost:8001/health
```

### 2. Test Database Connection
```bash
curl http://localhost:8001/test/database
```

### 3. Test Food Search
```bash
curl -X POST http://localhost:8001/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool": "search_food_by_name", "params": {"name": "apple"}}'
```

### 4. Test AI Meal Planning
```bash
curl -X POST http://localhost:8001/meal-plan \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "daily_calories": 2000,
    "meal_count": 3,
    "dietary_restrictions": ["vegetarian"]
  }'
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Database Connection Failed
- Check `DATABASE_URL` format
- Ensure PostgreSQL is running
- Verify network connectivity
- Check firewall settings

#### 2. Groq API Errors
- Verify `GROQ_API_KEY` is correct
- Check API rate limits
- Ensure network connectivity to Groq

#### 3. Missing Tables
- The agent creates tables automatically on startup
- Check database permissions
- Review startup logs for errors

#### 4. Redis Connection Issues
- Redis is optional - the agent will work without it
- Check `REDIS_URL` format
- Ensure Redis service is running

### Debug Endpoints

The agent provides several debug endpoints:

- `/debug/database` - Database connection status
- `/debug/schema` - Database schema information
- `/debug/foods` - Sample food data
- `/debug/search-test` - Search functionality testing

### Logs

Enable detailed logging by setting:
```bash
DEBUG=true
```

## 🚀 Production Deployment

### Heroku
1. Set environment variables in Heroku dashboard
2. Ensure `DATABASE_URL` and `REDIS_URL` are set
3. Deploy using Heroku Git integration

### Docker
```bash
docker build -t nutrition-agent .
docker run -p 8001:8001 --env-file .env nutrition-agent
```

### Environment Variables for Production
```bash
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=https://yourdomain.com
JWT_SECRET_KEY=your_secure_secret_key
```

## 📊 Monitoring

### Health Endpoints
- `/health` - Basic health check
- `/health/detailed` - Detailed health with database testing

### Performance Metrics
- `/api/performance/metrics` - Performance monitoring data
- `/api/scalability/status` - Scalability infrastructure status

## 🔐 Security

### Authentication
- Session-based authentication
- JWT token support
- Rate limiting enabled

### Data Validation
- Input sanitization
- SQL injection protection
- XSS protection

## 📈 Scaling

The agent is designed to handle:
- 100+ concurrent users
- Background task processing
- Redis-based caching
- Connection pooling
- Load balancing support

## 🆘 Support

If you encounter issues:
1. Check the debug endpoints
2. Review application logs
3. Verify environment variables
4. Test database connectivity
5. Check API service status

## 📝 Notes

- The agent automatically falls back to rule-based meal planning if AI fails
- Database tables are created automatically on first run
- Redis is optional but recommended for production
- All endpoints include proper error handling and validation
- The agent supports both single and multi-database configurations
