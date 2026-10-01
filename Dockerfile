FROM python:3.10-slim
WORKDIR /workspace
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app/ ./app/
COPY results/minute_2023_2025/models/ ./results/minute_2023_2025/models/
COPY results/minute_2023_2025/model_registry.csv ./results/minute_2023_2025/model_registry.csv
COPY results/minute_2023_2025/status.json ./results/minute_2023_2025/status.json
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
