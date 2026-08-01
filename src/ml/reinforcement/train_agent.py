"""
Entrenamiento del agente Q-Learning con validación y métricas.
"""

import numpy as np
import pandas as pd
import json
import os
import mlflow
from pathlib import Path
from environment import MovieRecommendationEnv
from agent import QLearningAgent

def train_agent():
    """Entrena el agente RL con validación"""
    
    print(" Iniciando entrenamiento del agente RL...")
    
    # 1. Inicializar entorno y agente
    env = MovieRecommendationEnv()
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0
    )
    
    # 2. Parámetros de entrenamiento
    episodes = 5000
    validation_episodes = 100
    
    # Obtener usuarios para entrenamiento
    users = env.ratings['userId'].unique()
    print(f"\n Entrenando con {len(users)} usuarios")
    print(f"   Episodios: {episodes}")
    print(f"   Estados: {env.num_states}")
    print(f"   Acciones: {env.num_actions}")
    
    # 3. Entrenamiento
    print("\n Entrenando...")
    episode_rewards = []
    
    for episode in range(episodes):
        # Seleccionar usuario aleatorio
        user_id = np.random.choice(users)
        
        # Obtener estado inicial (cluster preferido del usuario)
        state = env.get_state_for_user(user_id)
        
        # Elegir acción
        action = agent.choose_action(state)
        
        # Obtener recompensa
        reward = env.get_reward(state, action, user_id)
        episode_rewards.append(reward)
        
        # Actualizar Q-Table (next_state = mismo estado para simplificar)
        agent.update(state, action, reward, state)
        
        # Decaer epsilon
        if episode < episodes * 0.8:  # Decaer por 80% del entrenamiento
            agent.decay_epsilon(decay_rate=0.998, min_epsilon=0.01)
        
        # Mostrar progreso cada 1000 episodios
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"   Episodio {episode+1}/{episodes}: "
                  f"Avg Reward (últimos 100) = {avg_reward:.3f}, "
                  f"Epsilon = {agent.epsilon:.3f}")
    
    # 4. Validación final
    print("\n Validando agente...")
    validation_rewards = []
    
    for _ in range(validation_episodes):
        user_id = np.random.choice(users)
        state = env.get_state_for_user(user_id)
        
        # Usar política greedy (sin exploración)
        agent.epsilon = 0.0
        action = agent.choose_action(state)
        reward = env.get_reward(state, action, user_id)
        validation_rewards.append(reward)
    
    avg_validation = np.mean(validation_rewards)
    print(f"   Recompensa promedio en validación: {avg_validation:.3f}")
    
    # 5. Guardar modelo
    models_dir = Path("src/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'q_table.json'
    
    agent.save_model(str(model_path))
    print(f"\n Modelo guardado en {model_path}")
    
    # 6. Guardar métricas
    metrics = {
        'episodes': episodes,
        'validation_episodes': validation_episodes,
        'avg_reward_training': float(np.mean(episode_rewards[-100:])),
        'avg_reward_validation': float(avg_validation),
        'final_epsilon': float(agent.epsilon),
        'num_states': env.num_states,
        'num_actions': env.num_actions,
        'q_table': agent.q_table.tolist(),
        'best_actions': [int(np.argmax(agent.q_table[s, :])) for s in range(env.num_states)],
        'learning_rate': agent.alpha,
        'discount_factor': agent.gamma
    }
    
    with open(models_dir / 'rl_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f" Métricas guardadas en rl_metrics.json")
    
    # MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "local")
    if tracking_uri != "local":
        mlflow.set_tracking_uri(tracking_uri)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "movie-recommender")
    mlflow.set_experiment(exp_name)
    
    with mlflow.start_run(run_name="QLearning_Agent"):
        mlflow.log_params({
            'episodes': episodes,
            'learning_rate': agent.alpha,
            'discount_factor': agent.gamma
        })
        mlflow.log_metrics({
            'avg_reward_training': metrics['avg_reward_training'],
            'avg_reward_validation': metrics['avg_reward_validation']
        })
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(models_dir / 'rl_metrics.json'))
        
    # 7. Mostrar política aprendida
    print("\n Política aprendida:")
    action_names = {0: 'Explotar', 1: 'Explorar', 2: 'Mezclar'}
    for state in range(env.num_states):
        best_action = np.argmax(agent.q_table[state, :])
        print(f"   Cluster {state}: {action_names[best_action]} "
              f"(Q={agent.q_table[state, best_action]:.3f})")
    
    return agent, metrics

if __name__ == "__main__":
    train_agent()