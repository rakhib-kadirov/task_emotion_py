import os
import joblib
import streamlit as st

from preprocess import clean_text
from labels import LABEL_MAP

# ===== Пути к файлам =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ===== Интерфейс =====
st.title("Emotion Detection")
st.write("Введите предложение, и модель определит эмоцию")

user_text = st.text_area("Текст:")

if st.button("Определить эмоцию"):
    if user_text.strip() == "":
        st.warning("Введите текст")
    else:
        cleaned = clean_text(user_text)
        vectorized = vectorizer.transform([cleaned])
        prediction_num = model.predict(vectorized)[0]
        emotion = LABEL_MAP[prediction_num]

        st.success(f"Определённая эмоция: **{emotion}**")
