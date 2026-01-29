import pandas as pd
import matplotlib.pyplot as plt

from labels import LABEL_MAP

# ===== Загрузка данных =====
train_df = pd.read_csv("data/training.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print("TRAIN SHAPE:", train_df.shape)
print("VALIDATION SHAPE:", val_df.shape)
print("TEST SHAPE:", test_df.shape)

# ===== Проверка структуры =====
print("\nTRAIN COLUMNS:")
print(train_df.columns)

print("\nПервые 5 строк:")
print(train_df.head())

# ===== Проверка пропусков =====
print("\nПропущенные значения:")
print(train_df.isnull().sum())

# ===== Распределение эмоций =====
train_df["emotion"] = train_df["label"].map(LABEL_MAP)

emotion_counts = train_df["emotion"].value_counts()
print("\nРаспределение эмоций:")
print(emotion_counts)

# ===== Визуализация =====
plt.figure()
emotion_counts.plot(kind="bar")
plt.title("Распределение эмоций в TRAIN")
plt.xlabel("Эмоция")
plt.ylabel("Количество")
plt.show()

# ===== Анализ длины текстов =====
train_df["text_length"] = train_df["text"].apply(len)

print("\nСтатистика длины текста:")
print(train_df["text_length"].describe())

plt.figure()
plt.hist(train_df["text_length"], bins=30)
plt.title("Распределение длины текстов")
plt.xlabel("Длина текста (символы)")
plt.ylabel("Количество")
plt.show()