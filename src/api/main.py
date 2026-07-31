from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import json
from pathlib import Path
import sys
import os
import numpy as np
import traceback
import subprocess

# Añadir src al path
sys.path.append(str(Path(__file__).parent.parent))

from ml.reinforcement.agent import QLearningAgent
from api.schemas import UserData, FeedbackData

# Inicializar FastAPI
app = FastAPI(
    title="🎬 Movie Recommender with RL",
    description="Sistema de recomendación con Unsupervised Learning + Reinforcement Learning",
    version="1.0.0"
)

# Configurar rutas
BASE_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = BASE_DIR / "src" / "static"
TEMPLATES_DIR = BASE_DIR / "src" / "templates"

# Montar estáticos
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cargar modelos
models = {}
data = {}

def ensure_models_exist():
    """Verifica que los modelos existen y los entrena si es necesario"""
    models_dir = Path("src/models")
    processed_dir = Path("data/processed")
    
    # Verificar que los datos procesados existen
    if not (processed_dir / 'movies_with_genres.csv').exists():
        print(" Datos procesados no encontrados. Ejecutando preprocesamiento...")
        try:
            # Ejecutar preprocesamiento
            result = subprocess.run(
                ["uv", "run", "python", "scripts/preprocess_data.py"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f" Error en preprocesamiento: {result.stderr}")
                return False
            print(" Preprocesamiento completado")
        except Exception as e:
            print(f" Error ejecutando preprocesamiento: {e}")
            return False
    
    # Verificar que los modelos existen
    required_models = [
        'movie_kmeans.pkl',
        'movie_scaler.pkl',
        'svd_model.pkl',
        'q_table.json'
    ]
    
    missing_models = []
    for model_file in required_models:
        if not (models_dir / model_file).exists():
            missing_models.append(model_file)
    
    if missing_models:
        print(f" Modelos faltantes: {missing_models}")
        print(" Ejecutando entrenamiento de modelos...")
        
        try:
            # Entrenar clustering
            if 'movie_kmeans.pkl' in missing_models or 'movie_scaler.pkl' in missing_models:
                print("   Entrenando Clustering...")
                subprocess.run(
                    ["uv", "run", "python", "src/ml/clustering/train_clustering.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            # Entrenar SVD
            if 'svd_model.pkl' in missing_models:
                print("   Entrenando SVD...")
                subprocess.run(
                    ["uv", "run", "python", "src/ml/recommendation/train_svd.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            # Entrenar Agente RL
            if 'q_table.json' in missing_models:
                print("   Entrenando Agente RL...")
                subprocess.run(
                    ["uv", "run", "python", "src/ml/reinforcement/train_agent.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            print(" Todos los modelos entrenados")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f" Error entrenando modelos: {e.stderr}")
            return False
        except Exception as e:
            print(f" Error: {e}")
            return False
    
    return True

def load_models():
    """Carga todos los modelos y datos"""
    global models, data
    
    try:
        # Asegurar que los modelos existen
        if not ensure_models_exist():
            print("  Los modelos no pudieron ser entrenados automáticamente.")
            print("   Ejecuta manualmente:")
            print("   uv run python scripts/preprocess_data.py")
            print("   uv run python src/ml/clustering/train_clustering.py")
            print("   uv run python src/ml/recommendation/train_svd.py")
            print("   uv run python src/ml/reinforcement/train_agent.py")
            return
        
        models_dir = Path("src/models")
        processed_dir = Path("data/processed")
        
        print("🎬 Cargando modelos...")
        
        # Cargar datos
        data['movies'] = pd.read_csv(processed_dir / 'movies_with_clusters.csv')
        data['ratings'] = pd.read_csv(processed_dir / 'ratings_clean.csv')
        print(f" Datos cargados: {len(data['movies']):,} películas, {len(data['ratings']):,} ratings")
        
        # Cargar modelos
        models['kmeans'] = joblib.load(models_dir / 'movie_kmeans.pkl')
        models['scaler'] = joblib.load(models_dir / 'movie_scaler.pkl')
        models['svd'] = joblib.load(models_dir / 'svd_model.pkl')
        print(" Modelos sklearn cargados")
        
        # Cargar agente RL
        agent = QLearningAgent(num_states=1, num_actions=1)
        agent.load_model(str(models_dir / 'q_table.json'))
        models['agent'] = agent
        print(" Agente RL cargado")
        
        # Acciones
        models['action_map'] = {
            0: "⭐ Popular Choice (Exploit)",
            1: "🔍 New Discovery (Explore)",
            2: "🎭 Similar Taste (Mix)"
        }
        
        # Cargar métricas si existen
        metrics_files = ['clustering_metrics.json', 'svd_metrics.json', 'rl_metrics.json']
        for metric_file in metrics_files:
            if (models_dir / metric_file).exists():
                with open(models_dir / metric_file, 'r') as f:
                    models[metric_file.replace('.json', '')] = json.load(f)
        
        print(f" Modelos cargados: {len(data['movies']):,} películas, {len(data['ratings']):,} ratings")
        
    except Exception as e:
        print(f" ERROR cargando modelos: {e}")
        print(traceback.format_exc())
        raise

def load_models():
    global models, data
    
    try:
        models_dir = Path("src/models")
        processed_dir = Path("data/processed")
        
        print("🎬 Cargando modelos...")
        
        # Cargar datos
        data['movies'] = pd.read_csv(processed_dir / 'movies_with_clusters.csv')
        data['ratings'] = pd.read_csv(processed_dir / 'ratings_clean.csv')
        print(f" Datos cargados: {len(data['movies']):,} películas, {len(data['ratings']):,} ratings")
        
        # Cargar modelos
        models['kmeans'] = joblib.load(models_dir / 'movie_kmeans.pkl')
        models['scaler'] = joblib.load(models_dir / 'movie_scaler.pkl')
        models['svd'] = joblib.load(models_dir / 'svd_model.pkl')
        print(" Modelos sklearn cargados")
        
        # Cargar agente RL
        agent = QLearningAgent(num_states=1, num_actions=1)
        agent.load_model(str(models_dir / 'q_table.json'))
        models['agent'] = agent
        print(" Agente RL cargado")
        
        # Acciones
        models['action_map'] = {
            0: "⭐ Popular Choice (Exploit)",
            1: "🔍 New Discovery (Explore)",
            2: "🎭 Similar Taste (Mix)"
        }
        
        print(f" Modelos cargados: {len(data['movies']):,} películas, {len(data['ratings']):,} ratings")
        
    except Exception as e:
        print(f" ERROR cargando modelos: {e}")
        print(traceback.format_exc())
        raise

# Cargar modelos al inicio
load_models()

# ===== PÁGINA PRINCIPAL CON HTML DIRECTO =====
@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal con HTML directo (sin Jinja2)"""
    try:
        # Leer el archivo HTML directamente
        html_path = Path("src/templates/index.html")
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        print(f" Error en home: {e}")
        print(traceback.format_exc())
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

# ===== RUTA DE PRUEBA =====
@app.get("/test", response_class=HTMLResponse)
async def test_page():
    return """
    <html>
        <head><title>Test</title></head>
        <body>
            <h1>¡Funciona!</h1>
            <p>El servidor está corriendo correctamente.</p>
            <p>Prueba la API en <a href="/api/stats">/api/stats</a></p>
        </body>
    </html>
    """

# ===== ENDPOINTS API =====
@app.get("/api/stats")
async def get_stats():
    """Estadísticas del sistema"""
    return {
        "total_movies": int(len(data['movies'])) if data else 0,
        "total_users": int(data['ratings']['userId'].nunique()) if data else 0,
        "total_ratings": int(len(data['ratings'])) if data else 0,
        "n_clusters": int(data['movies']['cluster'].nunique()) if data else 0
    }

@app.post("/api/recommend")
async def get_recommendations(user_data: UserData):
    """Endpoint de recomendación con RL y diversidad"""
    try:
        user_id = user_data.user_id
        
        # Verificar que el usuario existe
        if user_id not in data['ratings']['userId'].values:
            return JSONResponse({
                "status": "error",
                "message": f"Usuario {user_id} no encontrado. Usa IDs entre 1 y {data['ratings']['userId'].max()}"
            })
        
        # Obtener cluster preferido del usuario
        user_ratings = data['ratings'][data['ratings']['userId'] == user_id]
        merged = user_ratings.merge(data['movies'][['movieId', 'cluster']], on='movieId')
        
        if len(merged) == 0:
            user_cluster = 0
        else:
            cluster_preference = merged.groupby('cluster')['rating'].mean()
            user_cluster = int(cluster_preference.idxmax())
        
        # El agente decide qué acción tomar
        agent = models['agent']
        agent.epsilon = 0.0
        action = agent.choose_action(state=user_cluster)
        
        # ===== NUEVO: Obtener el valor Q para mostrar =====
        q_value = agent.get_action_value(user_cluster, action)
        
        # ===== NUEVO: Obtener películas con diversidad =====
        # Obtener historial de recomendaciones para este usuario
        if user_id not in recommendation_history:
            recommendation_history[user_id] = []
        
        # Obtener todas las películas del cluster
        movies_in_cluster = data['movies'][data['movies']['cluster'] == user_cluster]
        
        # Si el cluster tiene pocas películas, incluir clusters cercanos
        if len(movies_in_cluster) < 10:
            # Buscar clusters cercanos
            neighbors = [user_cluster - 1, user_cluster + 1]
            neighbors = [c for c in neighbors if 0 <= c < data['movies']['cluster'].nunique()]
            for neighbor in neighbors:
                neighbor_movies = data['movies'][data['movies']['cluster'] == neighbor]
                movies_in_cluster = pd.concat([movies_in_cluster, neighbor_movies])
        
        # Calcular rating predicho para cada película
        movie_ratings = []
        for _, movie in movies_in_cluster.iterrows():
            movie_id = movie['movieId']
            try:
                pred_rating = models['svd'].predict(user_id, movie_id).est
                # Si la película ya fue vista, reducir su puntuación para fomentar diversidad
                if movie_id in recommendation_history[user_id]:
                    pred_rating *= 0.7  # Penalizar películas ya vistas
                movie_ratings.append((movie_id, pred_rating))
            except:
                continue
        
        # Ordenar por rating predicho (descendente)
        movie_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Seleccionar top 15 (para tener variedad)
        top_movies = [m[0] for m in movie_ratings[:15]]
        
        # Filtrar películas ya recomendadas recientemente (últimas 10)
        seen_movies = recommendation_history[user_id][-10:]
        available_movies = [m for m in top_movies if m not in seen_movies]
        
        # Si no hay suficientes películas nuevas, permitir algunas repeticiones
        if len(available_movies) < 5:
            # Mezclar: 3 nuevas + 2 vistas (pero con penalización)
            new_movies = [m for m in top_movies if m not in seen_movies]
            old_movies = [m for m in top_movies if m in seen_movies]
            selected_movies = new_movies[:3] + old_movies[:2]
        else:
            selected_movies = available_movies[:5]
        
        # Guardar en historial
        recommendation_history[user_id].extend(selected_movies)
        recommendation_history[user_id] = recommendation_history[user_id][-20:]
        
        # Obtener información de las películas
        recommendations = []
        for movie_id in selected_movies:
            movie = data['movies'][data['movies']['movieId'] == movie_id].iloc[0]
            pred_rating = models['svd'].predict(user_id, movie_id).est
            
            # Obtener título y géneros
            title = str(movie['title']) if 'title' in movie.index else f"Película {movie_id}"
            
            # Buscar la columna de géneros
            genres_col = None
            for col in ['genres', 'genre', 'Genres', 'Genre']:
                if col in movie.index:
                    genres_col = col
                    break
            
            if genres_col:
                genres = str(movie[genres_col])
            else:
                # Usar géneros one-hot
                genre_cols = [col for col in data['movies'].columns if col in [
                    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 
                    'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi', 
                    'Thriller', 'War', 'Western'
                ]]
                if genre_cols:
                    genres_list = [col for col in genre_cols if movie[col] == 1]
                    genres = '|'.join(genres_list) if genres_list else 'Sin género'
                else:
                    genres = 'Sin género'
            
            recommendations.append({
                "movie_id": int(movie_id),
                "title": title,
                "genres": genres,
                "predicted_rating": round(float(pred_rating), 2)
            })
        
        return {
            "status": "success",
            "user_cluster": user_cluster,
            "action_taken": str(action),
            "action_description": models['action_map'][action],
            "q_value": round(float(q_value), 3),  # Mostrar el valor Q
            "recommendations": recommendations,
            "message": f"Recomendaciones para usuario {user_id}"
        }
        
    except Exception as e:
        print(f" Error en recommend: {e}")
        print(traceback.format_exc())
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

# Historial de recomendaciones por usuario
recommendation_history = {}

@app.post("/api/feedback")
async def receive_feedback(feedback: FeedbackData):
    """Endpoint para recibir feedback y actualizar el agente en tiempo real"""
    try:
        # Obtener el estado (cluster) del usuario
        user_ratings = data['ratings'][data['ratings']['userId'] == feedback.user_id]
        merged = user_ratings.merge(data['movies'][['movieId', 'cluster']], on='movieId')
        
        if len(merged) == 0:
            return JSONResponse({
                "status": "error",
                "message": "Usuario no encontrado"
            })
        
        cluster_preference = merged.groupby('cluster')['rating'].mean()
        state = int(cluster_preference.idxmax())
        
        # ===== NUEVO: Recompensa más sensible =====
        if feedback.rating >= 4.0:
            reward = 1.0  # Me gusta mucho
        elif feedback.rating >= 3.0:
            reward = 0.2  # Me gusta un poco
        elif feedback.rating >= 2.0:
            reward = -0.3  # No me gusta un poco
        else:
            reward = -0.8  # No me gusta mucho
        
        # Actualizar el agente en tiempo real
        agent = models['agent']
        action = feedback.action_taken
        
        # ===== NUEVO: Actualizar con mayor tasa de aprendizaje =====
        agent.alpha = 0.3  # Aumentar tasa de aprendizaje para que sea más sensible
        agent.update(state, action, reward, state)
        agent.save_model(str(Path("src/models/q_table.json")))
        
        # ===== NUEVO: Limpiar historial para forzar nuevas recomendaciones =====
        if feedback.user_id in recommendation_history:
            recommendation_history[feedback.user_id] = []
        
        print(f" Feedback: Usuario {feedback.user_id}, Rating {feedback.rating}, Reward {reward:.2f}")
        print(f"   Q-Table actualizada para estado {state}, acción {action}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Feedback recibido. Recompensa: {reward:.2f}",
            "q_table_updated": True
        })
        
    except Exception as e:
        print(f" Error en feedback: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)