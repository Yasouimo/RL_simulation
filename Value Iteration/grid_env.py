import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class GridWorldEnv:
    """
    Environnement GridWorld compatible avec l'interface Gymnasium.
    """
    
    # Actions possibles
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    def __init__(self, grid_size=5, start_pos=(0, 0), goal_pos=(4, 4), 
                 obstacles=[(2, 2), (3, 2)], step_cost=-0.01):
        """
        Initialise l'environnement GridWorld.
        
        Args:
            grid_size: Taille de la grille (int pour carré ou tuple pour rectangle)
            start_pos: Position de départ de l'agent (row, col)
            goal_pos: Position du but (row, col)
            obstacles: Liste des positions des obstacles
            step_cost: Coût de chaque déplacement
        """
        if isinstance(grid_size, int):
            self.rows = self.cols = grid_size
        else:
            self.rows, self.cols = grid_size
            
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles
        self.step_cost = step_cost
        
        self.agent_pos = None
        self.num_actions = 4
        
        # Récompenses
        self.goal_reward = 1.0
        self.obstacle_reward = -1.0
        
        self.reset()
    
    def reset(self):
        """
        Remet l'agent à sa position de départ.
        
        Returns:
            state: Position initiale de l'agent
        """
        self.agent_pos = list(self.start_pos)
        return tuple(self.agent_pos)
    
    def step(self, action):
        """
        Effectue une action dans l'environnement.
        
        Args:
            action: Action à effectuer (UP, DOWN, LEFT, RIGHT)
            
        Returns:
            next_state: Nouvelle position
            reward: Récompense obtenue
            done: Episode terminé ou non
            info: Informations supplémentaires
        """
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
        
        # Vérification des obstacles
        if tuple(new_pos) in self.obstacles:
            # Collision avec un obstacle
            reward = self.obstacle_reward
            done = False
            self.agent_pos = new_pos  # L'agent se déplace quand même sur l'obstacle
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
        
        return tuple(self.agent_pos), reward, done, {}
    
    def render(self, value_table=None, fig=None, ax=None):
        """
        Affiche la grille avec Matplotlib.
        
        Args:
            value_table: Matrice optionnelle des valeurs d'états pour afficher une heatmap
            fig: Figure matplotlib existante (optionnel)
            ax: Axes matplotlib existant (optionnel)
        """
        # Utiliser la figure existante ou en créer une nouvelle
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            ax.clear()
        
        # Si value_table est fourni, créer une heatmap
        if value_table is not None:
            # Normaliser les valeurs pour la couleur
            vmin = np.min(value_table)
            vmax = np.max(value_table)
            
            for i in range(self.rows):
                for j in range(self.cols):
                    # Couleur basée sur la valeur
                    if vmax > vmin:
                        normalized_value = (value_table[i, j] - vmin) / (vmax - vmin)
                    else:
                        normalized_value = 0.5
                    
                    color = plt.cm.viridis(normalized_value)
                    rect = Rectangle((j, self.rows - 1 - i), 1, 1, 
                                    facecolor=color, edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    
                    # Afficher la valeur numérique
                    ax.text(j + 0.5, self.rows - 1 - i + 0.5, 
                           f'{value_table[i, j]:.2f}',
                           ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            # Grille simple sans valeurs
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
                   'X', ha='center', va='center', fontsize=20, color='white', fontweight='bold')
        
        # Dessiner le but
        rect = Rectangle((self.goal_pos[1], self.rows - 1 - self.goal_pos[0]), 1, 1, 
                       facecolor='gold', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(self.goal_pos[1] + 0.5, self.rows - 1 - self.goal_pos[0] + 0.5, 
               'G', ha='center', va='center', fontsize=20, color='green', fontweight='bold')
        
        # Dessiner l'agent
        ax.plot(self.agent_pos[1] + 0.5, self.rows - 1 - self.agent_pos[0] + 0.5, 
               'ro', markersize=20, label='Agent')
        
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.grid(True)
        ax.set_title('GridWorld Environment', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.pause(0.1)  # Pause courte pour animation fluide
        
        return fig, ax
    
    def get_state_index(self, state):
        """
        Convertit une position (row, col) en index d'état unique.
        """
        return state[0] * self.cols + state[1]
    
    def get_state_from_index(self, index):
        """
        Convertit un index d'état en position (row, col).
        """
        row = index // self.cols
        col = index % self.cols
        return (row, col)
