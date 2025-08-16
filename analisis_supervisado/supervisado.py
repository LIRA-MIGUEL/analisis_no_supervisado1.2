import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("resenas_superman.csv", encoding="utf-8")

# Procesar datos
df["rating"] = df["rating"].str.split("/").str[0]
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating", "review"])
df = df[df["review"].str.strip() != ""]
df["review"] = df["review"].str.lower()

# Asignar sentimiento
df["sentimiento"] = df["rating"].apply(
    lambda r: "positivo" if r >= 7 else "neutral" if r >= 5 else "negativo"
)

# Vectorizar textos
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = vectorizer.fit_transform(df["review"])
y = df["sentimiento"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Predicciones en test
y_pred = model.predict(X_test)

# Mostrar ejemplos de predicción
print("Ejemplos de predicciones:\n")
for i, (real, pred) in enumerate(zip(y_test[:10], y_pred[:10]), 1):
    print(f"{i:>2}. Real: {real:<9} | Predicho: {pred:<9}")

# Precisión
print(f"\n Precisión del modelo: {accuracy_score(y_test, y_pred):.2%}\n")

# Reporte de clasificación
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=["positivo", "neutral", "negativo"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["positivo", "neutral", "negativo"],
            yticklabels=["positivo", "neutral", "negativo"])
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# Clasificar todas las reseñas
df["pred_sentimiento"] = model.predict(vectorizer.transform(df["review"]))

# Mostrar todas las reseñas con clasificación real y predicha
print("\n Reseñas clasificadas:\n")
for idx, row in df.iterrows():
    print(f"Reseña: {row['review']}")
    print(f"Rating: {row['rating']} → Real: {row['sentimiento']} | Predicho: {row['pred_sentimiento']}")
    print("-" * 80)
