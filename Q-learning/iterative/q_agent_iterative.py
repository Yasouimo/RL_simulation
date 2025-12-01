import numpy as np
from collections import defaultdict


class QLearningAgentIterative:
    """
    Agent Q-Learning qui apprend itération par itération.
    Met à jour la Q-table immédiatement après chaque transition.
    """
    
    def __init__(self, num_actions=4, learning_rate=0.1, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialise l'agent Q-Learning itératif.
        
        Args:
            num_actions: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (alpha)
            gamma: Facteur d'actualisation
            epsilon: Probabilité d'exploration initiale
            epsilon_decay: Facteur de décroissance d'epsilon
            epsilon_min: Valeur minimale d'epsilon
        """
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionnaire avec clé = state_features, valeur = array d'actions
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        
        # Compteur de mises à jour
        self.update_count = 0
    
    def get_action(self, state, training=True):
        """
        Choisit une action selon la politique epsilon-greedy.
        
        Args:
            state: Features de l'état actuel
            training: Mode entraînement (avec exploration) ou non
            
        Returns:
            action: Action choisie
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(0, self.num_actions)
        else:
            # Exploitation: meilleure action selon Q-table
            q_values = self.q_table[state]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """
        Met à jour la Q-table immédiatement après une transition.
        Méthode itérative: mise à jour à chaque pas.
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Episode terminé
        """
        # Valeur Q actuelle
        current_q = self.q_table[state][action]
        
        # Calcul de la cible
        if done:
            target = reward
        else:
            # Q-learning: max_a' Q(s', a')
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q
        
        # Mise à jour Q-learning
        self.q_table[state][action] = current_q + self.lr * (target - current_q)
        
        self.update_count += 1
    
    def decay_epsilon(self):
        """
        Décroît epsilon (appelé à la fin de chaque épisode).
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, state):
        """
        Retourne les valeurs Q pour un état donné.
        
        Args:
            state: État (features)
            
        Returns:
            q_values: Array des valeurs Q pour chaque action
        """
        return self.q_table[state]
    
    def get_stats(self):
        """
        Retourne des statistiques sur l'agent.
        
        Returns:
            dict: Statistiques (epsilon, taille Q-table, etc.)
        """
        return {
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'update_count': self.update_count,
            'learning_rate': self.lr,
            'gamma': self.gamma
        }
