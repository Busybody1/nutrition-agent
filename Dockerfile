FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set pip timeout for slow networks
ENV PIP_DEFAULT_TIMEOUT=100

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

# Copy shared modules and agent code
COPY shared /app/shared
COPY main.py /app/main.py

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT", "--workers", "1"] 