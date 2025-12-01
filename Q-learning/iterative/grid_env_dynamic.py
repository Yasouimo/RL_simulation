import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class DynamicGridWorldEnv:
    """
    Environnement GridWorld avec goal dynamique pour Q-learning.
    Le goal change de position de manière aléatoire pendant l'entraînement.
    """
    
    # Actions possibles
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(self, grid_size=5, obstacles=[(2, 2)], step_cost=-0.01, 
                 goal_reward=10.0, max_steps_per_episode=100):
        """
        Initialise l'environnement GridWorld dynamique.
        
        Args:
            grid_size: Taille de la grille
            obstacles: Liste des positions des obstacles
            step_cost: Coût de chaque déplacement
            goal_reward: Récompense pour atteindre le goal
            max_steps_per_episode: Nombre maximum de pas par épisode
        """
        if isinstance(grid_size, int):
            self.rows = self.cols = grid_size
        else:
            self.rows, self.cols = grid_size
            
        self.obstacles = obstacles
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.max_steps_per_episode = max_steps_per_episode
        
        self.agent_pos = None
        self.goal_pos = None
        self.num_actions = 4
        self.current_steps = 0
        
        # Récompenses
        self.obstacle_reward = -1.0
        
    def _get_random_free_position(self):
        """
        Génère une position aléatoire qui n'est pas un obstacle.
        """
        while True:
            pos = (np.random.randint(0, self.rows), 
                   np.random.randint(0, self.cols))
            if pos not in self.obstacles:
                return pos
    
    def reset(self):
        """
        Remet l'agent et le goal à des positions aléatoires.
        
        Returns:
            state: État initial (features)
        """
        self.agent_pos = list(self._get_random_free_position())
        self.goal_pos = self._get_random_free_position()
        
        # S'assurer que l'agent et le goal ne sont pas au même endroit
        while tuple(self.agent_pos) == self.goal_pos:
            self.goal_pos = self._get_random_free_position()
        
        self.current_steps = 0
        return self._get_state_features()
    
    def _get_state_features(self):
        """
        Calcule les features de l'état actuel.
        Features: (agent_row, agent_col, distance_manhattan_au_goal)
        
        Returns:
            features: Tuple de features
        """
        agent_row, agent_col = self.agent_pos
        goal_row, goal_col = self.goal_pos
        
        # Distance de Manhattan
        distance = abs(agent_row - goal_row) + abs(agent_col - goal_col)
        
        return (agent_row, agent_col, distance)
    
    def step(self, action):
        """
        Effectue une action dans l'environnement.
        
        Args:
            action: Action à effectuer (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            next_state: Nouvelles features
            reward: Récompense obtenue
            done: Episode terminé ou non
            info: Informations supplémentaires
        """
        self.current_steps += 1
        
        # Calcul de la nouvelle position
        new_pos = self.agent_pos.copy()
        
        if action == self.UP:
            new_pos[0] -= 1
        elif action == self.DOWN:
            new_pos[0] += 1
        elif action == self.LEFT:
            new_pos[1] -= 1
        elif action == self.RIGHT:
            new_pos[1] += 1
        
        # Vérification des limites de la grille
        if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
            # Collision avec un mur, l'agent reste en place
            new_pos = self.agent_pos.copy()
            reward = self.step_cost
            done = False
        # Vérification des obstacles
        elif tuple(new_pos) in self.obstacles:
            # Collision avec un obstacle
            reward = self.obstacle_reward
            done = False
            self.agent_pos = new_pos
        # Vérification si le goal est atteint
        elif tuple(new_pos) == self.goal_pos:
            # Atteint le but
            reward = self.goal_reward
            done = True
            self.agent_pos = new_pos
        else:
            # Déplacement normal
            reward = self.step_cost
            done = False
            self.agent_pos = new_pos
        
        # Vérifier si le nombre max de pas est atteint
        if self.current_steps >= self.max_steps_per_episode:
            done = True
        
        return self._get_state_features(), reward, done, {}
    
    def render(self, fig=None, ax=None, q_table=None):
        """
        Affiche la grille avec Matplotlib.
        
        Args:
            fig: Figure matplotlib existante (optionnel)
            ax: Axes matplotlib existant (optionnel)
            q_table: Table Q optionnelle pour afficher les valeurs
        """
        # Utiliser la figure existante ou en créer une nouvelle
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            ax.clear()
        
        # Grille de base
        for i in range(self.rows):
            for j in range(self.cols):
                rect = Rectangle((j, self.rows - 1 - i), 1, 1, 
                               facecolor='white', edgecolor='black', linewidth=2)
                ax.add_patch(rect)
        
        # Dessiner les obstacles
        for obs in self.obstacles:
            rect = Rectangle((obs[1], self.rows - 1 - obs[0]), 1, 1, 
                           facecolor='gray', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(obs[1] + 0.5, self.rows - 1 - obs[0] + 0.5, 
                   'X', ha='center', va='center', fontsize=20, 
                   color='white', fontweight='bold')
        
        # Dessiner le goal (dynamique)
        circle = Circle((self.goal_pos[1] + 0.5, self.rows - 1 - self.goal_pos[0] + 0.5),
                       0.3, facecolor='gold', edgecolor='orange', linewidth=3)
        ax.add_patch(circle)
        ax.text(self.goal_pos[1] + 0.5, self.rows - 1 - self.goal_pos[0] + 0.5, 
               'G', ha='center', va='center', fontsize=16, 
               color='green', fontweight='bold')
        
        # Dessiner l'agent (boule)
        agent_circle = Circle((self.agent_pos[1] + 0.5, 
                              self.rows - 1 - self.agent_pos[0] + 0.5),
                             0.25, facecolor='red', edgecolor='darkred', linewidth=2)
        ax.add_patch(agent_circle)
        
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.grid(True)
        ax.set_title('Dynamic GridWorld - Q-Learning (Iterative)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.pause(0.01)
        
        return fig, ax
