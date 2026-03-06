# Fraud Detection System

ML-сервис для обнаружения мошеннических транзакций
## О проекте

Сервис определяет мошеннические транзакции по 8 ключевым признакам. 
Модель XGBoost оптимизирована с помощью Optuna.

**Метрики** (порог 0.3):
- Precision: 93.8% (точность предсказания fraud)
- Recall: 81.8% (сколько мошенников поймали)
- F1-score: 0.87

## Стек
- Python 3.11
- XGBoost
- FastAPI (REST API)
- Streamlit (Web UI)
- Docker
- Optuna (оптимизация)
- Scikit-learn (метрики, RFE)

### Docker

```bash
# Собрать образ
docker build -t fraud_service .

# Запустить контейнер
docker run -p 8000:8000 -p 8501:8501 fraud_service
```

После запуска:
- API: http://localhost:8000/docs
- UI: http://localhost:8501
