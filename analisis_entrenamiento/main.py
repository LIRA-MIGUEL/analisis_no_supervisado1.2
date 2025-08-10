import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk

# Configuración NLTK y stopwords
nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

# Usamos todas las stopwords oficiales sin modificaciones
spanish_stopwords = list(stopwords.words("spanish"))  # <-- aquí la corrección clave

def cargar_datos(path_csv='resenas_superman.csv'):
    data = pd.read_csv(path_csv, encoding='utf-8')
    data['rating'] = data['rating'].str.split('/').str[0]
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating', 'review'])
    data = data[data['review'].str.strip() != '']
    data['review'] = data['review'].str.lower()
    data['sentimiento'] = data['rating'].apply(lambda r: 'positivo' if r >= 7 else 'neutral' if r >=5 else 'negativo')
    return data

def entrenar_modelos(data):
    vectorizador = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000, ngram_range=(1,2))
    X = vectorizador.fit_transform(data['review'])

    y = data['sentimiento']
    modelo_clasif = LogisticRegression(max_iter=1000, class_weight='balanced')
    modelo_clasif.fit(X, y)

    k = 4
    modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo_kmeans.fit(X)

    data['cluster'] = modelo_kmeans.labels_

    return vectorizador, modelo_clasif, modelo_kmeans, data

def predecir(texto, vectorizador, modelo_clasif, modelo_kmeans):
    texto_proc = texto.lower()
    X_new = vectorizador.transform([texto_proc])
    pred_sentimiento = modelo_clasif.predict(X_new)[0]
    pred_cluster = modelo_kmeans.predict(X_new)[0]
    return pred_sentimiento, pred_cluster

def guardar_nueva_reseña(path_csv, texto, rating, sentimiento, cluster):
    import os
    nueva_fila = pd.DataFrame({
        'rating': [rating],
        'review': [texto],
        'sentimiento': [sentimiento],
        'cluster': [cluster]
    })
    if os.path.exists(path_csv):
        nueva_fila.to_csv(path_csv, mode='a', header=False, index=False)
    else:
        nueva_fila.to_csv(path_csv, index=False)

def menu_interactivo():
    data = cargar_datos()
    vectorizador, modelo_clasif, modelo_kmeans, data = entrenar_modelos(data)
    dataset_path = 'resenas_superman_ampliado.csv'

    print("=== Análisis de Sentimientos y Clustering interactivo ===")

    while True:
        print("\nOpciones:")
        print("1. Ingresar nueva reseña")
        print("2. Salir")
        opcion = input("Elige una opción: ").strip()

        if opcion == '1':
            texto = input("Escribe la reseña: ").strip()
            rating = input("Escribe el rating (numérico del 1 al 10): ").strip()
            try:
                rating_num = float(rating)
                if rating_num < 1 or rating_num > 10:
                    raise ValueError
            except ValueError:
                print("Rating inválido. Debe ser un número entre 1 y 10.")
                continue

            sentimiento, cluster = predecir(texto, vectorizador, modelo_clasif, modelo_kmeans)
            print(f"Predicción de sentimiento: {sentimiento}")
            print(f"Predicción de cluster: {cluster}")

            guardar_nueva_reseña(dataset_path, texto, rating_num, sentimiento, cluster)
            print(f"Reseña guardada en '{dataset_path}' para ampliar dataset.")

        elif opcion == '2':
            print("Saliendo...")
            break
        else:
            print("Opción no válida, intenta de nuevo.")

if __name__ == "__main__":
    menu_interactivo()
