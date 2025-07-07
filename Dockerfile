FROM python:3.12-slim
WORKDIR /app
COPY shared /app/shared
COPY . /app
COPY requirements.txt /app/requirements.txt
COPY nutrition_agent/main.py /app/main.py

# Set pip timeout for slow networks
ENV PIP_DEFAULT_TIMEOUT=100
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"] 