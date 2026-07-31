#!/usr/bin/env python
"""
Script de inicio unico para la aplicacion Movie Recommender with RL.
Descarga dataset, entrena modelos y levanta el servidor automaticamente.
"""

import sys
import subprocess
from pathlib import Path
import time
import io

# Forzar UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def print_section(title):
    """Imprime una seccion con formato"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()

def run_command(cmd, description):
    """Ejecuta un comando y maneja errores"""
    print(f"[>]  {description}...")
    print(f"   Comando: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Error en {description}:")
            print(result.stderr)
            return False
        
        # Mostrar output relevante (ultimas lineas)
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 5:
                print("   ... (ultimas lineas) ...")
                for line in lines[-5:]:
                    print(f"   {line}")
            else:
                for line in lines:
                    print(f"   {line}")
        
        print(f"[OK] {description} completado")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error ejecutando {description}: {e}")
        return False

def check_dependencies():
    """Verifica que uv este instalado"""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except:
        print("[ERROR] uv no esta instalado. Instala con: pip install uv")
        return False

def check_dataset():
    """Verifica si el dataset existe, si no, lo descarga"""
    raw_dir = Path("data/raw/ml-latest-small")
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print("[OK] Dataset ya existe localmente")
        return True
    return False

def run_all():
    """Ejecuta todo el pipeline completo"""
    
    print("Movie Recommender with RL")
    print("=" * 50)
    print("Sistema de recomendacion con:")
    print("  * Unsupervised Learning (K-Means)")
    print("  * Sistema de Recomendacion (SVD)")
    print("  * Reinforcement Learning (Q-Learning)")
    print()
    
    # Verificar dependencias
    print("[INFO] Verificando dependencias...")
    if not check_dependencies():
        return
    
    # Verificar dataset
    print("[INFO] Verificando dataset...")
    if check_dataset():
        print("[OK] Dataset encontrado")
    else:
        print("[INFO] Dataset no encontrado. Se descargara automaticamente.")
    
    print()
    
    # 1. Preprocesar datos (incluye descarga automatica)
    print_section("PASO 1: Preprocesamiento de datos")
    if not run_command(
        ["uv", "run", "python", "scripts/preprocess_data.py"],
        "Preprocesamiento (incluye descarga automatica del dataset)"
    ):
        print("\n[ERROR] Fallo el preprocesamiento. Verifica los errores arriba.")
        return
    
    # 2. Entrenar Clustering
    print_section("PASO 2: Entrenando Clustering (K-Means)")
    if not run_command(
        ["uv", "run", "python", "src/ml/clustering/train_clustering.py"],
        "Clustering de peliculas"
    ):
        print("\n[ERROR] Fallo el entrenamiento de Clustering.")
        return
    
    # 3. Entrenar SVD
    print_section("PASO 3: Entrenando Sistema de Recomendacion (SVD)")
    if not run_command(
        ["uv", "run", "python", "src/ml/recommendation/train_svd.py"],
        "Factorizacion Matricial"
    ):
        print("\n[ERROR] Fallo el entrenamiento de SVD.")
        return
    
    # 4. Entrenar Agente RL
    print_section("PASO 4: Entrenando Agente de Refuerzo (Q-Learning)")
    if not run_command(
        ["uv", "run", "python", "src/ml/reinforcement/train_agent.py"],
        "Agente Q-Learning"
    ):
        print("\n[ERROR] Fallo el entrenamiento del Agente RL.")
        return
    
    # 5. Levantar servidor
    print_section("PASO 5: Levantando servidor FastAPI")
    print("[INFO] Servidor iniciando...")
    print("   Abre http://127.0.0.1:8000 en tu navegador")
    print("   Presiona CTRL+C para detener")
    print()
    
    time.sleep(2)
    
    try:
        subprocess.run(
            ["uv", "run", "uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        )
    except KeyboardInterrupt:
        print("\n\n[INFO] Servidor detenido por el usuario")

if __name__ == "__main__":
    try:
        run_all()
    except KeyboardInterrupt:
        print("\n\n[INFO] Proceso interrumpido")
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
        sys.exit(1)