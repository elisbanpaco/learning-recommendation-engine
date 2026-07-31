import numpy as np
import json
from pathlib import Path

class QLearningAgent:
    def __init__(self, num_states: int, num_actions: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        self.q_table = np.zeros((num_states, num_actions))
        self.rewards_history = []
        self.exploration_history = []
    
    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
            self.exploration_history.append(1)
        else:
            action = np.argmax(self.q_table[state, :])
            self.exploration_history.append(0)
        return action
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        self.rewards_history.append(reward)
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    # ===== NUEVO MÉTODO: Obtener valor Q =====
    def get_action_value(self, state: int, action: int) -> float:
        """Retorna el valor Q para un par estado-acción"""
        return float(self.q_table[state, action])
    
    # ===== NUEVO MÉTODO: Obtener mejor acción =====
    def get_best_action(self, state: int) -> int:
        """Retorna la mejor acción para un estado"""
        return int(np.argmax(self.q_table[state, :]))
    
    # ===== NUEVO MÉTODO: Mostrar tabla Q =====
    def get_q_table(self):
        """Retorna la tabla Q completa"""
        return self.q_table.tolist()
    
    def save_model(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'q_table': self.q_table.tolist(),
                'epsilon': self.epsilon,
                'num_states': self.num_states,
                'num_actions': self.num_actions,
                'rewards_history': self.rewards_history[-1000:]
            }, f, indent=2)
    
    def load_model(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.q_table = np.array(data['q_table'])
        self.epsilon = data.get('epsilon', 0.01)
        self.num_states = data.get('num_states', self.num_states)
        self.num_actions = data.get('num_actions', self.num_actions)
        print(f" Modelo cargado: {self.num_states} estados, {self.num_actions} acciones")
        return self.q_table