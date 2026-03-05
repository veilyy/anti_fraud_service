import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection")
st.title("Fraud Detection")

st.write("Введите 8 признаков для проверки транзакции:")

v10 = st.number_input("V10", value=0.0, format="%.4f")
v14 = st.number_input("V14", value=0.0, format="%.4f")
v17 = st.number_input("V17", value=0.0, format="%.4f")
v4 = st.number_input("V4", value=0.0, format="%.4f")
v12 = st.number_input("V12", value=0.0, format="%.4f")
v7 = st.number_input("V7", value=0.0, format="%.4f")
v6 = st.number_input("V6", value=0.0, format="%.4f")
amount = st.number_input("Сумма транзакции", value=100.0)

if st.button("Проверить"):
    data = {
        "V10": v10, "V14": v14, "V17": v17,
        "V4": v4, "V12": v12, "V7": v7, "V6": v6,
        "Amount": amount
    }
    
    try:
        # Запрос к API
        response = requests.post("http://localhost:8000/fraud", json=data)
        result = response.json()
        
        if result["fraud"]:
            st.error(f" МОШЕННИЧЕСТВО! (вероятность: {result['probability']:.2%})")
        else:
            st.success(f"Безопасно (вероятность: {result['probability']:.2%})")
            
    except Exception as e:
        st.error(f"Ошибка: {e}")