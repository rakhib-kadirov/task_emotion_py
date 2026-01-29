import pandas as pd
import joblib

from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ===== Загрузка данных =====
train_df = pd.read_csv("data/training.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

# ===== Предобработка текста =====
for df in (train_df, val_df, test_df):
    df["clean_text"] = df["text"].apply(clean_text)

# ===== Разделение =====
X_train = train_df["clean_text"]
y_train = train_df["label"]

X_val = val_df["clean_text"]
y_val = val_df["label"]

X_test = test_df["clean_text"]
y_test = test_df["label"]

# ===== TF-IDF =====
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# ===== Модель =====
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ===== Оценка =====
print("===== VALIDATION RESULTS =====")
print(classification_report(y_val, model.predict(X_val_vec)))

print("===== TEST RESULTS =====")
print(classification_report(y_test, model.predict(X_test_vec)))

# ===== Сохранение =====
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Модель и векторизатор сохранены.")