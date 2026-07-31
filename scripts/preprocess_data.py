"""
Script de preprocesamiento dinámico para MovieLens Latest Small.
Detecta automaticamente la estructura del dataset sin hardcodear nombres.
Descarga automaticamente el dataset si no existe localmente.
"""

import pandas as pd
import os
import json
import zipfile
import urllib.request
from pathlib import Path
import shutil
import sys
import io

# Forzar UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def download_movielens():
    """Descarga el dataset MovieLens Latest Small desde la URL oficial"""
    
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "ml-latest-small.zip"
    extract_path = raw_dir / "ml-latest-small"
    
    # Si ya existe, no descargar
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"[OK] Dataset ya existe en {extract_path}")
        return True
    
    # URL oficial de MovieLens Latest Small
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    print(f"[INFO] Descargando dataset desde: {url}")
    print(f"   Esto puede tomar unos segundos...")
    
    try:
        # Descargar con progreso
        urllib.request.urlretrieve(url, zip_path)
        print(f"[OK] Descarga completada: {zip_path}")
        
        # Descomprimir
        print(f"[INFO] Descomprimiendo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print(f"[OK] Extraccion completada en {extract_path}")
        
        # Eliminar ZIP para ahorrar espacio (opcional)
        # zip_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error descargando dataset: {e}")
        print(f"   Por favor descarga manualmente desde:")
        print(f"   {url}")
        print(f"   Y descomprime en: {extract_path}")
        return False

def detect_and_preprocess():
    """Detecta automaticamente la estructura y preprocesa los datos"""
    
    print("[INFO] Iniciando preprocesamiento dinamico de MovieLens...")
    
    # 1. Descargar dataset si no existe
    if not download_movielens():
        print("[ERROR] No se pudo descargar el dataset. Abortando.")
        return None, None, None
    
    # 2. Definir rutas
    raw_dir = Path("data/raw/ml-latest-small")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Verificar que los archivos existen
    required_files = ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv']
    missing = [f for f in required_files if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Archivos faltantes: {missing}")
    
    print("[OK] Todos los archivos encontrados")
    
    # 4. Cargar ratings dinamicamente
    print("\n[INFO] Cargando ratings...")
    ratings = pd.read_csv(raw_dir / 'ratings.csv')
    print(f"   [OK] {len(ratings):,} calificaciones")
    print(f"   Columnas: {list(ratings.columns)}")
    print(f"   Rango de ratings: {ratings['rating'].min()} - {ratings['rating'].max()}")
    
    # 5. Cargar movies dinamicamente
    print("\n[INFO] Cargando peliculas...")
    movies = pd.read_csv(raw_dir / 'movies.csv')
    print(f"   [OK] {len(movies):,} peliculas")
    print(f"   Columnas: {list(movies.columns)}")
    
    # 6. Procesar generos (pipe-separated a one-hot)
    print("\n[INFO] Procesando generos...")
    genres_col = 'genres'
    
    # Obtener todos los generos unicos
    all_genres = set()
    for genres_str in movies[genres_col].dropna():
        all_genres.update(genres_str.split('|'))
    all_genres = sorted(all_genres)
    print(f"   Generos encontrados: {len(all_genres)}")
    print(f"   {all_genres}")
    
    # Crear columnas one-hot
    for genre in all_genres:
        movies[genre] = movies[genres_col].apply(
            lambda x: 1 if isinstance(x, str) and genre in x.split('|') else 0
        )
    
    # 7. Guardar datasets procesados
    print("\n[INFO] Guardando datos procesados...")
    
    movies.to_csv(processed_dir / 'movies_with_genres.csv', index=False)
    print(f"   [OK] movies_with_genres.csv ({len(movies):,} filas)")
    
    ratings.to_csv(processed_dir / 'ratings_clean.csv', index=False)
    print(f"   [OK] ratings_clean.csv ({len(ratings):,} filas)")
    
    # 8. Guardar metadata
    metadata = {
        'total_movies': len(movies),
        'total_users': ratings['userId'].nunique(),
        'total_ratings': len(ratings),
        'genres': all_genres,
        'rating_range': [float(ratings['rating'].min()), float(ratings['rating'].max())],
        'date_processed': pd.Timestamp.now().isoformat(),
        'dataset_source': 'https://grouplens.org/datasets/movielens/latest/',
        'dataset_version': 'ml-latest-small'
    }
    
    with open(processed_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   [OK] metadata.json guardado")
    
    # 9. Mostrar estadisticas
    print("\n[STATS] Estadisticas del Dataset:")
    print(f"   Usuarios: {metadata['total_users']:,}")
    print(f"   Peliculas: {metadata['total_movies']:,}")
    print(f"   Calificaciones: {metadata['total_ratings']:,}")
    print(f"   Ratings: {metadata['rating_range'][0]} - {metadata['rating_range'][1]}")
    print(f"   Generos: {len(metadata['genres'])}")
    
    print("\n[OK] Preprocesamiento completado!")
    return movies, ratings, metadata

if __name__ == "__main__":
    detect_and_preprocess()