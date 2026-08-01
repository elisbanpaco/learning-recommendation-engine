"""
Entrenamiento dinámico de K-Means para clustering de películas.
Detecta automáticamente las columnas de géneros.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os
import json
from pathlib import Path
import mlflow
import mlflow.sklearn

def train_clustering():
    """Entrena K-Means dinámicamente detectando columnas de géneros"""
    
    print(" Iniciando entrenamiento de Clustering...")
    
    # 1. Cargar datos procesados
    processed_dir = Path("data/processed")
    movies = pd.read_csv(processed_dir / 'movies_with_genres.csv')
    
    print(f" Cargadas {len(movies):,} películas")
    
    # 2. Detectar columnas de género dinámicamente
    genre_cols = [col for col in movies.columns if col not in ['movieId', 'title', 'genres']]
    print(f" Columnas de género detectadas: {len(genre_cols)}")
    print(f"   {genre_cols[:5]}...")
    
    # 3. Preparar datos
    X = movies[genre_cols].values
    
    # 4. Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Determinar número óptimo de clusters (Elbow Method)
    print("\n Determinando número óptimo de clusters...")
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Mostrar resultados
    print("\n Métricas por número de clusters:")
    for k, sil, iner in zip(k_range, silhouette_scores, inertias):
        print(f"   k={k}: Silhouette={sil:.3f}, Inercia={iner:.1f}")
    
    # Seleccionar el mejor k (mayor silhouette score)
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n Mejor k: {best_k} (Silhouette: {max(silhouette_scores):.3f})")
    
    # 6. Entrenar modelo final con el mejor k
    print(f"\n Entrenando K-Means con k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 7. Calcular métricas finales
    labels = kmeans.labels_
    final_silhouette = silhouette_score(X_scaled, labels)
    final_db = davies_bouldin_score(X_scaled, labels)
    
    print(f"\n Métricas finales:")
    print(f"   Silhouette Score: {final_silhouette:.3f}")
    print(f"   Davies-Bouldin: {final_db:.3f}")
    print(f"   Inercia: {kmeans.inertia_:.1f}")
    
    # 8. Guardar modelos
    models_dir = Path("src/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(kmeans, models_dir / 'movie_kmeans.pkl')
    joblib.dump(scaler, models_dir / 'movie_scaler.pkl')
    
    print(f"\n Modelos guardados en {models_dir}")
    
    # 9. Guardar películas con clusters
    movies['cluster'] = kmeans.labels_
    movies[['movieId', 'title', 'cluster'] + genre_cols[:5]].to_csv(
        processed_dir / 'movies_with_clusters.csv', 
        index=False
    )
    print(f"    movies_with_clusters.csv guardado")
    
    # 10. Mostrar distribución de clusters
    print("\n Distribución de clusters:")
    cluster_stats = movies.groupby('cluster').agg({
        'movieId': 'count',
        **{genre: 'sum' for genre in genre_cols[:3]}
    })
    for cluster in range(best_k):
        count = cluster_stats.loc[cluster, 'movieId']
        top_genres = movies[movies['cluster'] == cluster][genre_cols].sum().sort_values(ascending=False).head(3)
        print(f"\n   Cluster {cluster}: {count} películas")
        print(f"      Géneros predominantes: {', '.join(top_genres.index.tolist())}")
    
    # 11. Guardar métricas
    metrics = {
        'n_clusters': best_k,
        'silhouette_score': float(final_silhouette),
        'davies_bouldin_score': float(final_db),
        'inertia': float(kmeans.inertia_),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': genre_cols,
        'cluster_distribution': movies['cluster'].value_counts().to_dict()
    }
    
    with open(models_dir / 'clustering_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n Métricas guardadas en clustering_metrics.json")
    
    # MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "local")
    if tracking_uri != "local":
        mlflow.set_tracking_uri(tracking_uri)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "movie-recommender")
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run(run_name="K-Means_Clustering"):
        mlflow.log_params({
            'k_range_start': min(k_range),
            'k_range_end': max(k_range),
            'best_k': best_k,
            'random_state': 42
        })
        mlflow.log_metrics({
            'silhouette_score': float(final_silhouette),
            'davies_bouldin_score': float(final_db),
            'inertia': float(kmeans.inertia_)
        })
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        mlflow.sklearn.log_model(scaler, "scaler_model")
        mlflow.log_artifact(str(models_dir / 'clustering_metrics.json'))
        mlflow.log_artifact(str(processed_dir / 'movies_with_clusters.csv'))
    
    return kmeans, scaler, metrics

if __name__ == "__main__":
    train_clustering()