"""
Entorno para el agente RL de recomendación de películas.
Detecta automáticamente la estructura de datos.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class MovieRecommendationEnv:
    def __init__(self):
        """Inicializa el entorno"""
        
        # Cargar datos
        processed_dir = Path("data/processed")
        models_dir = Path("src/models")
        
        # Cargar películas con clusters
        self.movies = pd.read_csv(processed_dir / 'movies_with_clusters.csv')
        self.ratings = pd.read_csv(processed_dir / 'ratings_clean.csv')
        
        # Cargar modelo SVD
        self.svd = joblib.load(models_dir / 'svd_model.pkl')
        
        # Detectar número de clusters
        self.num_states = self.movies['cluster'].nunique()
        self.num_actions = 3  # Explotar, Explorar, Mezclar
        
        # Calcular estadísticas por cluster
        self._compute_cluster_stats()
        
        print(f" Entorno inicializado:")
        print(f"   Estados (clusters): {self.num_states}")
        print(f"   Acciones: {self.num_actions}")
        print(f"   Películas: {len(self.movies):,}")
        print(f"   Ratings: {len(self.ratings):,}")
    
    def _compute_cluster_stats(self):
        """Calcula estadísticas por cluster"""
        merged = self.ratings.merge(self.movies[['movieId', 'cluster']], on='movieId')
        self.cluster_stats = {
            'ratings_mean': merged.groupby('cluster')['rating'].mean().to_dict(),
            'ratings_count': merged.groupby('cluster')['rating'].count().to_dict(),
            'n_movies': self.movies.groupby('cluster')['movieId'].count().to_dict()
        }
    
    def get_reward(self, state: int, action: int, user_id: int) -> float:
        """
        Calcula recompensa para un estado-acción-usuario
        
        Args:
            state: Cluster actual
            action: 0=Explotar, 1=Explorar, 2=Mezclar
            user_id: ID del usuario
            
        Returns:
            Recompensa entre -1 y 1
        """
        # Obtener películas del cluster
        cluster_movies = self.movies[self.movies['cluster'] == state]['movieId'].tolist()
        if not cluster_movies:
            return 0.0
        
        # Seleccionar película según acción
        if action == 0:  # Explotar - mejor rating promedio
            movie_id = self._get_best_movie_in_cluster(state)
        elif action == 1:  # Explorar - película novedosa (menos ratings)
            movie_id = self._get_novel_movie_in_cluster(state)
        else:  # Mezclar - película de cluster similar
            movie_id = self._get_mix_movie(state)
        
        # Predecir rating usando SVD
        try:
            pred_rating = self.svd.predict(user_id, movie_id).est
        except:
            pred_rating = self.ratings['rating'].mean()
        
        # Normalizar recompensa (1-5 → 0-1)
        min_rating = self.ratings['rating'].min()
        max_rating = self.ratings['rating'].max()
        reward = (pred_rating - min_rating) / (max_rating - min_rating)
        
        # Bonus si es mejor que el promedio del cluster
        avg_cluster = self.cluster_stats['ratings_mean'].get(state, 3.0)
        if pred_rating > avg_cluster:
            reward += 0.2
        
        # Penalizar si es peor que el promedio
        elif pred_rating < avg_cluster - 0.5:
            reward -= 0.2
        
        return np.clip(reward, -1.0, 1.0)
    
    def _get_best_movie_in_cluster(self, cluster: int) -> int:
        """Retorna la película con mejor rating en el cluster"""
        movies_in_cluster = self.movies[self.movies['cluster'] == cluster]
        merged = movies_in_cluster.merge(self.ratings, on='movieId')
        if len(merged) == 0:
            return int(movies_in_cluster.iloc[0]['movieId'])
        best = merged.groupby('movieId')['rating'].mean().idxmax()
        return int(best)
    
    def _get_novel_movie_in_cluster(self, cluster: int) -> int:
        """Retorna la película menos conocida (menos ratings)"""
        movies_in_cluster = self.movies[self.movies['cluster'] == cluster]
        merged = movies_in_cluster.merge(self.ratings, on='movieId')
        if len(merged) == 0:
            return int(movies_in_cluster.iloc[0]['movieId'])
        counts = merged.groupby('movieId').size()
        novel = counts.idxmin() if len(counts) > 0 else int(movies_in_cluster.iloc[0]['movieId'])
        return int(novel)
    
    def _get_mix_movie(self, state: int) -> int:
        """Retorna película de cluster similar (saltos de ±1)"""
        # Elegir cluster vecino
        offset = np.random.choice([-1, 1])
        next_cluster = (state + offset) % self.num_states
        movies_in_cluster = self.movies[self.movies['cluster'] == next_cluster]
        if len(movies_in_cluster) == 0:
            return int(self.movies.iloc[0]['movieId'])
        return int(movies_in_cluster.sample(1).iloc[0]['movieId'])
    
    def get_state_for_user(self, user_id: int) -> int:
        """
        Determina el cluster preferido de un usuario basado en sus ratings
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        if len(user_ratings) == 0:
            return 0  # Usuario nuevo
        
        merged = user_ratings.merge(self.movies[['movieId', 'cluster']], on='movieId')
        if len(merged) == 0:
            return 0
        
        # Cluster con mejor rating promedio (ponderado por cantidad)
        cluster_preference = merged.groupby('cluster').agg({
            'rating': 'mean',
            'movieId': 'count'
        })
        cluster_preference['score'] = cluster_preference['rating'] * np.log(1 + cluster_preference['movieId'])
        
        return int(cluster_preference['score'].idxmax())