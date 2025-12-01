import numpy as np


class RandomAgent:
    """
    Agent qui choisit des actions aléatoires.
    """
    
    def __init__(self, num_actions=4):
        """
        Initialise l'agent aléatoire.
        
        Args:
            num_actions: Nombre d'actions possibles
        """
        self.num_actions = num_actions
    
    def choose_action(self, state=None):
        """
        Choisit une action aléatoire.
        
        Args:
            state: État actuel (non utilisé pour un agent aléatoire)
            
        Returns:
            action: Action aléatoire
        """
        return np.random.randint(0, self.num_actions)


class ValueIterationAgent:
    """
    Agent qui utilise l'algorithme de Value Iteration pour apprendre
    la politique optimale.
    """
    
    def __init__(self, gamma=0.9, theta=1e-6):
        """
        Initialise l'agent Value Iteration.
        
        Args:
            gamma: Facteur d'actualisation (discount factor)
            theta: Seuil de convergence
        """
        self.gamma = gamma
        self.theta = theta
        self.V = None  # Table des valeurs d'états
        self.env = None
    
    def train(self, env, max_iterations=1000):
        """
        Entraîne l'agent en utilisant l'algorithme de Value Iteration.
        
        Args:
            env: Environnement GridWorld
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            V: Table des valeurs d'états après convergence
        """
        self.env = env
        
        # Initialiser la table des valeurs à zéro
        self.V = np.zeros((env.rows, env.cols))
        
        print("Début de l'entraînement avec Value Iteration...")
        
        for iteration in range(max_iterations):
            delta = 0  # Pour vérifier la convergence
            V_old = self.V.copy()
            
            # Pour chaque état de la grille
            for i in range(env.rows):
                for j in range(env.cols):
                    state = (i, j)
                    
                    # Skip si c'est le goal
                    if state == env.goal_pos:
                        self.V[i, j] = 0  # Le goal a une valeur de 0 (état terminal)
                        continue
                    
                    # Skip si c'est un obstacle (on peut le laisser avec sa valeur)
                    if state in env.obstacles:
                        continue
                    
                    # Calculer la valeur pour toutes les actions possibles
                    action_values = []
                    
                    for action in range(env.num_actions):
                        # Sauvegarder la position actuelle
                        old_pos = env.agent_pos.copy()
                        
                        # Placer l'agent dans cet état
                        env.agent_pos = [i, j]
                        
                        # Simuler l'action
                        next_state, reward, done, _ = env.step(action)
                        
                        # Calculer la valeur selon l'équation de Bellman
                        # V(s) = max_a [R(s,a) + gamma * V(s')]
                        if done:
                            value = reward
                        else:
                            value = reward + self.gamma * V_old[next_state[0], next_state[1]]
                        
                        action_values.append(value)
                        
                        # Restaurer la position
                        env.agent_pos = old_pos
                    
                    # Prendre le maximum sur toutes les actions (Bellman optimality)
                    self.V[i, j] = max(action_values)
                    
                    # Mettre à jour delta pour la convergence
                    delta = max(delta, abs(self.V[i, j] - V_old[i, j]))
            
            # Afficher la progression tous les 100 itérations
            if (iteration + 1) % 100 == 0:
                print(f"Itération {iteration + 1}: Delta = {delta:.6f}")
            
            # Vérifier la convergence
            if delta < self.theta:
                print(f"Convergence atteinte après {iteration + 1} itérations!")
                break
        
        print("Entraînement terminé.")
        return self.V
    
    def choose_action(self, state):
        """
        Choisit la meilleure action basée sur la politique gloutonne (greedy)
        dérivée de la table des valeurs.
        
        Args:
            state: État actuel (row, col)
            
        Returns:
            best_action: Meilleure action à prendre
        """
        if self.V is None:
            raise ValueError("L'agent n'a pas été entraîné. Appelez d'abord train().")
        
        # Sauvegarder la position actuelle de l'agent dans l'environnement
        old_pos = self.env.agent_pos.copy()
        
        # Placer l'agent dans l'état donné
        self.env.agent_pos = list(state)
        
        best_value = float('-inf')
        best_action = 0
        
        # Évaluer chaque action
        for action in range(self.env.num_actions):
            # Sauvegarder la position avant de tester l'action
            temp_pos = self.env.agent_pos.copy()
            
            # Simuler l'action
            next_state, reward, done, _ = self.env.step(action)
            
            # Calculer la valeur de cette action
            if done:
                value = reward
            else:
                value = reward + self.gamma * self.V[next_state[0], next_state[1]]
            
            # Garder la meilleure action
            if value > best_value:
                best_value = value
                best_action = action
            
            # Restaurer la position pour tester l'action suivante
            self.env.agent_pos = temp_pos
        
        # Restaurer la position originale
        self.env.agent_pos = old_pos
        
        return best_action
    
    def get_value_table(self):
        """
        Retourne la table des valeurs d'états.
        
        Returns:
            V: Table des valeurs (numpy array)
        """
        return self.V
