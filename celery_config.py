"""
Celery Configuration for Nutrition Agent
Handles background meal planning and nutrition analysis
"""

import os
from celery import Celery
from kombu import Exchange, Queue

# Initialize Celery
celery_app = Celery('nutrition_agent')

# Configuration
celery_app.conf.update(
    broker_url=os.getenv('REDISCLOUD_URL', os.getenv('REDIS_URL', 'redis://localhost:6379')),
    result_backend=os.getenv('REDISCLOUD_URL', os.getenv('REDIS_URL', 'redis://localhost:6379')),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    
    # Task routing
    task_routes={
        'tasks.nutrition.*': {'queue': 'nutrition'},
    },
    
    # Timeouts - Workers have 5 minutes max
    task_time_limit=300,  # 5 minutes hard limit
    task_soft_time_limit=240,  # 4 minutes soft limit
    
    # Retries
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Result expiration
    result_expires=3600,  # 1 hour
    
    # Worker settings
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=50,
    
    # Task execution settings
    task_track_started=True,
    task_ignore_result=False,
)

# Define queues
celery_app.conf.task_default_queue = 'nutrition'
celery_app.conf.task_queues = (
    Queue('nutrition', Exchange('nutrition'), routing_key='nutrition'),
)

# Auto-discover tasks
celery_app.autodiscover_tasks(['tasks'])
