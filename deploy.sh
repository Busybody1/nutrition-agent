#!/bin/bash
# Deployment script for nutrition-agent

echo "🚀 Starting nutrition-agent deployment..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the nutrition_agent directory."
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Set branch to main
echo "🌿 Setting branch to main..."
git branch -M main

# Check if Heroku remote exists
if ! git remote | grep -q heroku; then
    echo "❌ Error: Heroku remote not found. Please add it with:"
    echo "   heroku git:remote -a your-app-name"
    exit 1
fi

# Deploy to Heroku
echo "🚀 Deploying to Heroku..."
git push heroku main --force

echo "✅ Deployment complete!"
echo "📊 Check the logs with: heroku logs --tail" 