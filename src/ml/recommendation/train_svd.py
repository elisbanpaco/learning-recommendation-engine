"""
Entrenamiento dinámico de SVD para sistema de recomendación.
Detecta automáticamente la estructura del dataset.
"""

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
import pandas as pd
import joblib
import json
from pathlib import Path
import numpy as np

def train_svd():
    """Entrena SVD dinámicamente con optimización de hiperparámetros"""
    
    print(" Iniciando entrenamiento de SVD...")
    
    # 1. Cargar datos
    processed_dir = Path("data/processed")
    ratings = pd.read_csv(processed_dir / 'ratings_clean.csv')
    
    print(f" Cargadas {len(ratings):,} calificaciones")
    print(f"   Columnas: {list(ratings.columns)}")
    
    # 2. Configurar Surprise
    min_rating = ratings['rating'].min()
    max_rating = ratings['rating'].max()
    print(f"   Rango de ratings: {min_rating} - {max_rating}")
    
    reader = Reader(rating_scale=(min_rating, max_rating))
    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']], 
        reader
    )
    
    # 3. Optimización de hiperparámetros con GridSearch
    print("\n Optimizando hiperparámetros con GridSearch...")
    
    # GridSearch más rápido
    param_grid = {
        'n_factors': [20, 50, 100],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.05]
    }
    
    gs = GridSearchCV(
        SVD, 
        param_grid, 
        measures=['rmse', 'mae'],
        cv=3,
        n_jobs=-1
    )
    gs.fit(data)
    
    # 4. Mejores parámetros
    best_params = gs.best_params['rmse']
    print(f"\n Mejores hiperparámetros:")
    print(f"   n_factors: {best_params['n_factors']}")
    print(f"   n_epochs: {best_params['n_epochs']}")
    print(f"   lr_all: {best_params['lr_all']}")
    print(f"   reg_all: {best_params['reg_all']}")
    print(f"   Best RMSE: {gs.best_score['rmse']:.4f}")
    
    # 5. Entrenar modelo final
    print("\n Entrenando modelo final...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    model = SVD(
        n_factors=best_params['n_factors'],
        n_epochs=best_params['n_epochs'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all'],
        random_state=42
    )
    model.fit(trainset)
    
    # 6. Evaluar
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    print(f"\n Métricas en test:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # 7. Guardar modelo
    models_dir = Path("src/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, models_dir / 'svd_model.pkl')
    print(f"\nModelo guardado en {models_dir / 'svd_model.pkl'}")
    
    # 8. Guardar métricas - CORREGIDO FINAL
    # gs.best_index es un dict con {'rmse': idx, 'mae': idx}
    best_index_rmse = int(gs.best_index['rmse'])
    
    metrics = {
        'best_params': best_params,
        'rmse': float(rmse),
        'mae': float(mae),
        'n_users': trainset.n_users,
        'n_items': trainset.n_items,
        'n_ratings': trainset.n_ratings,
        'test_size': len(testset),
        'grid_search_results': {
            'mean_rmse': float(gs.best_score['rmse']),
            'std_rmse': float(gs.cv_results['std_test_rmse'][best_index_rmse])
        }
    }
    
    with open(models_dir / 'svd_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas en svd_metrics.json")
    
    return model, metrics

if __name__ == "__main__":
    train_svd()