# NOTE: written but NOT build-tested in the development sandbox (no Docker available there).
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app BACKEND_URL=http://127.0.0.1:8000
EXPOSE 8000 8501
CMD ["python", "run.py", "--api-port", "8000", "--ui-port", "8501"]
