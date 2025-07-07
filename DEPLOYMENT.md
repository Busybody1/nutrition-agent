# Nutrition Agent Deployment Guide

## Quick Deployment

The nutrition-agent has been optimized for fast Heroku deployment with minimal boot time.

### Prerequisites

1. **Heroku CLI installed**
2. **Heroku account with PostgreSQL addon**
3. **Git repository initialized**

### Deployment Steps

1. **Navigate to nutrition_agent directory:**
   ```bash
   cd nutrition_agent
   ```

2. **Initialize git (if not already done):**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. **Set branch to main:**
   ```bash
   git branch -M main
   ```

4. **Add Heroku remote:**
   ```bash
   heroku git:remote -a your-app-name
   ```

5. **Deploy:**
   ```bash
   git push heroku main --force
   ```

### Or use the deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

## Optimizations Made

### Boot Time Optimizations
- ✅ Removed blocking database operations from startup
- ✅ Reduced logging verbosity during startup
- ✅ Made database engine creation lazy
- ✅ Added fast health check endpoint
- ✅ Optimized uvicorn configuration

### Error Handling
- ✅ Comprehensive error handling for database operations
- ✅ Fallback mechanisms for all critical operations
- ✅ Graceful degradation when services are unavailable

### Configuration
- ✅ Automatic PostgreSQL URL conversion
- ✅ Environment variable handling
- ✅ Lazy initialization of all components

## Monitoring

### Check deployment status:
```bash
heroku logs --tail
```

### Test endpoints:
- `/health` - Fast health check
- `/health/detailed` - Detailed health check with database
- `/foods/count` - Get food count

## Troubleshooting

### R10 Boot Timeout
If you still get boot timeout errors:
1. Check that all environment variables are set
2. Verify the Procfile is correct
3. Check the logs for any blocking operations

### Database Connection Issues
The app will work even if the database is not available initially. The database connection will be established when first needed.

## Environment Variables

Make sure these are set in Heroku:
- `DATABASE_URL` - PostgreSQL connection string
- `GROQ_API_KEY` - Your Groq API key
- `REDIS_URL` - (Optional) Redis connection string 