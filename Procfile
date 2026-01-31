web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 90
worker: celery -A celery_config.celery_app worker --loglevel=info --concurrency=2 --queues=nutrition