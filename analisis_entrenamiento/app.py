import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

spanish_stopwords = stopwords.words("spanish")

data = pd.read_csv('resenas_superman.csv', encoding='utf-8')

data['rating'] = data['rating'].str.split('/').str[0]
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

data = data.dropna(subset=['rating', 'review'])
data = data[data['review'].str.strip() != '']

data['review'] = data['review'].str.lower()

def etiqueta(rating):
    if rating >= 7:
        return 'positivo'
    elif rating >= 5:
        return 'neutral'
    else:
        return 'negativo'

data['sentimiento'] = data['rating'].apply(etiqueta)

vectorizador = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000)
X = vectorizador.fit_transform(data['review'])

# --- Análisis supervisado: Clasificación ---
print("\n" + "="*30)
print("ANÁLISIS SUPERVISADO - Clasificación")
print("="*30)

y = data['sentimiento']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo_clasif = LogisticRegression(max_iter=1000)
modelo_clasif.fit(X_train, y_train)
y_pred = modelo_clasif.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))

resultados_sup = data.iloc[y_test.index][['review', 'sentimiento']].copy()
resultados_sup['prediccion'] = y_pred
resultados_sup.to_csv('resultados_supervisado.csv', index=False)

# --- Análisis no supervisado: Clustering ---
print("\n" + "="*30)
print("ANÁLISIS NO SUPERVISADO - Clustering")
print("="*30)

k = 4
modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
modelo_kmeans.fit(X)
data['cluster'] = modelo_kmeans.labels_

print("\n--- Primeras 10 reseñas con etiqueta supervisada y cluster asignado ---")
for i, row in data.head(10).iterrows():
    print(f"Reseña: {row['review'][:80]}...")
    print(f"Etiqueta supervisada: {row['sentimiento']}")
    print(f"Cluster asignado: {row['cluster']}")
    print("-"*50)

# Aquí imprimimos la distribución de etiquetas por cluster
print("\nDistribución de etiquetas supervisadas por cluster:")
for c in sorted(data['cluster'].unique()):
    subset = data[data['cluster'] == c]
    print(f"\nCluster {c}: {len(subset)} reseñas")
    print(subset['sentimiento'].value_counts())

data[['review', 'sentimiento', 'cluster']].to_csv('resultados_clustering.csv', index=False)

cluster_counts = data['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Cantidad de reseñas')
plt.title('Distribución de reseñas por cluster')
plt.xticks(cluster_counts.index)
plt.show()
