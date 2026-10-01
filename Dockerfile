FROM python:3.10-slim
WORKDIR /workspace
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app/ ./app/
COPY results/models/ ./results/models/
COPY results/model_registry.csv ./results/model_registry.csv
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
