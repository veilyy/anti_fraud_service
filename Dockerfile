FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/xgb_model.pkl /app/models/xgb_model.pkl

RUN mkdir -p /app/models

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.service:app --host 0.0.0.0 --port 8000 & streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0"]