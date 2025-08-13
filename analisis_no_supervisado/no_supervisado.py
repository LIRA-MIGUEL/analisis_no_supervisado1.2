import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk
import os

# Configuración NLTK
nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)
spanish_stopwords = list(stopwords.words("spanish"))

def cargar_datos(path_csv='resenas_superman.csv'):
    data = pd.read_csv(path_csv, encoding='utf-8')
    data = data.dropna(subset=['review'])
    data = data[data['review'].str.strip() != '']
    data['review'] = data['review'].str.lower()
    return data

def entrenar_modelo_no_supervisado(data, k=4):
    vectorizador = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000, ngram_range=(1,3))
    X = vectorizador.fit_transform(data['review'])
    modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo_kmeans.fit(X)
    data['cluster'] = modelo_kmeans.labels_
    return vectorizador, modelo_kmeans, data

def predecir_cluster(texto, vectorizador, modelo_kmeans):
    texto_proc = texto.lower()
    X_new = vectorizador.transform([texto_proc])
    return modelo_kmeans.predict(X_new)[0]

def nombre_cluster(cluster_num):
    nombres = {
        0: "Cluster Tecnología",
        1: "Cluster Opiniones Positivas",
        2: "Cluster Quejas",
        3: "Cluster Neutro",
    }
    return nombres.get(cluster_num, f"Cluster {cluster_num}")

def guardar_csv(data, filename="reseñas_cluster.csv"):
    data.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n✅ Archivo guardado: {filename}")

if __name__ == "__main__":
    print("Cargando datos y entrenando modelo KMeans...")
    data = cargar_datos()
    vectorizador, modelo_kmeans, data = entrenar_modelo_no_supervisado(data, k=4)
    print("Entrenamiento completado.")

    # Preparamos dataframe para guardar nuevas reseñas
    # Añadimos columnas rating (float) y cluster (int)
    if 'rating' not in data.columns:
        data['rating'] = None
    if 'cluster' not in data.columns:
        data['cluster'] = None

    while True:
        print("\nMenú:")
        print("1. Agregar reseña con rating")
        print("2. Salir")
        opcion = input("Elige una opción (1-2): ").strip()

        if opcion == '1':
            texto = input("Ingresa la reseña: ").strip()
            if texto == '':
                print("⚠️ La reseña no puede estar vacía.")
                continue

            # Validar rating
            rating_input = input("Ingresa el rating (1-10): ").strip()
            try:
                rating = float(rating_input)
                if rating < 1 or rating > 10:
                    raise ValueError
            except ValueError:
                print("⚠️ Rating inválido. Debe ser un número entre 1 y 10.")
                continue

            cluster = predecir_cluster(texto, vectorizador, modelo_kmeans)
            nombre = nombre_cluster(cluster)
            print(f"\n➡️ Cluster asignado: {cluster} - {nombre}")

            # Agregar al dataframe para guardar
            nueva_fila = pd.DataFrame({
                "review": [texto],
                "rating": [rating],
                "cluster": [cluster]
            })
            data = pd.concat([data, nueva_fila], ignore_index=True)

            # Guardar cada vez que se agrega una reseña
            guardar_csv(data)

        elif opcion == '2':
            print("Saliendo...")
            break

        else:
            print("Opción inválida. Intenta de nuevo.")
