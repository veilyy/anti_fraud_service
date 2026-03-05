import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os      

def main():
    df = pd.read_csv('../data/raw/creditcard.csv')
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42,
        stratify=y  # важно в данном датасете
    )
    
    from xgboost import XGBClassifier

    # Подобранные параметры с помощью библиотеки optuna, которые показали наилучшие результаты

    best_params = {
    'n_estimators': 175,
    'max_depth': 3,
    'learning_rate': 0.10172153971660594,
    'subsample': 0.6570274135897809,
    'colsample_bytree': 0.9698906731478989,
    'gamma': 0.7133886458627672,
    'reg_alpha': 0.9721614563882643,
    'reg_lambda': 1.6909440502708735,
    'random_state': 42,
    'eval_metric': 'aucpr'}

    features = ['V10', 'V14', 'V17', 'V4', 'V12', 'V7', 'V6', 'Amount']

    xgb = XGBClassifier(**best_params)
    xgb.fit(X_train[features], y_train)
    
    # Получаем вероятности
    y_proba = xgb.predict_proba(X_test[features])[:, 1]
    
    # Применяем порог 0.3
    THRESHOLD = 0.3
    y_pred = (y_proba >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Порог классификации: {THRESHOLD}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # Сохраняем метрики в текстовый файл
    os.makedirs('../models', exist_ok=True)
    
    with open('../models/xgb_model_metrics.txt', 'w') as f:
        f.write(f"Порог классификации: {THRESHOLD}\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"[[{cm[0,0]} {cm[0,1]}]\n")
        f.write(f" [{cm[1,0]} {cm[1,1]}]]\n")

    print("\n Метрики сохранены в models/xgb_model_metrics.txt")
    
    # Сохраняем модель
    joblib.dump(xgb, '../models/xgb_model.pkl')
    print(" Модель сохранена в models/xgb_model.pkl")

if __name__ == "__main__":
    main()