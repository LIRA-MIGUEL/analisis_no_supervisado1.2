import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import nltk
import os

nltk_data_path = "nltk_data"
nltk.data.path.append(nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

spanish_stopwords = list(stopwords.words("spanish"))

def cargar_datos(path_csv='resenas_superman.csv'):
    data = pd.read_csv(path_csv, encoding='utf-8')
    data['rating'] = data['rating'].str.split('/').str[0]
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating', 'review'])
    data = data[data['review'].str.strip() != '']
    data['review'] = data['review'].str.lower()
    data['sentimiento'] = data['rating'].apply(lambda r: 'positivo' if r >= 7 else 'neutral' if r >= 5 else 'negativo')
    return data

def entrenar_modelos(data):
    vectorizador = TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000, ngram_range=(1,3))
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
    pred_proba = modelo_clasif.predict_proba(X_new)[0]
    clases = modelo_clasif.classes_
    probs = dict(zip(clases, pred_proba))

    print(f"Probabilidades predicción: {probs}")

    # Umbral mínimo para considerar válida la predicción (puedes ajustar)
    umbral_minimo = 0.3

    # Filtrar clases que tengan probabilidad mayor al umbral
    clases_filtradas = [(clase, prob) for clase, prob in probs.items() if prob >= umbral_minimo]

    if not clases_filtradas:
        # Si ninguna supera el umbral, tomar la clase con mayor probabilidad
        pred_sentimiento = clases[pred_proba.argmax()]
    else:
        # Si varias pasan el umbral, elegir la que tenga mayor probabilidad
        pred_sentimiento = max(clases_filtradas, key=lambda x: x[1])[0]

    pred_cluster = modelo_kmeans.predict(X_new)[0]
    return pred_sentimiento, pred_cluster

def guardar_reseña(path_csv, texto, rating, sentimiento_texto, sentimiento_rating, cluster):
    nueva_fila = pd.DataFrame({
        'rating': [rating],
        'review': [texto],
        'sentimiento_texto': [sentimiento_texto],
        'sentimiento_rating': [sentimiento_rating],
        'cluster': [cluster]
    })
    if os.path.exists(path_csv):
        nueva_fila.to_csv(path_csv, mode='a', header=False, index=False)
    else:
        nueva_fila.to_csv(path_csv, index=False)

if __name__ == "__main__":
    print("Cargando datos y entrenando modelos...")
    data = cargar_datos()
    vectorizador, modelo_clasif, modelo_kmeans, data = entrenar_modelos(data)
    print("Listo para usar.")

    dataset_guardado = 'resenas_superman_ampliado.csv'

    while True:
        opcion = input("\n1. Ingresar nueva reseña\n2. Salir\nElige una opción: ").strip()
        if opcion == '1':
            texto = input("Escribe la reseña: ").strip()
            rating = input("Escribe el rating (1-10): ").strip()
            try:
                rating_num = float(rating)
                if not (1 <= rating_num <= 10):
                    raise ValueError
            except:
                print("Rating inválido. Intenta de nuevo.")
                continue

            sentimiento_rating = 'positivo' if rating_num >= 7 else 'neutral' if rating_num >= 5 else 'negativo'
            sentimiento_texto, cluster = predecir(texto, vectorizador, modelo_clasif, modelo_kmeans)

            print(f"\nSentimiento por texto: {sentimiento_texto}")
            print(f"Sentimiento por rating: {sentimiento_rating}")
            print(f"Cluster asignado: {cluster}")

            guardar_reseña(dataset_guardado, texto, rating_num, sentimiento_texto, sentimiento_rating, cluster)
            print(f"Reseña guardada en '{dataset_guardado}'.")

        elif opcion == '2':
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")
