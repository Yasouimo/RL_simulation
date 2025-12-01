import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class DynamicGridWorldEnv:
    """
    Environnement GridWorld avec goal dynamique.
    Identique à celui utilisé pour Q-Learning.
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
        """
        self.agent_pos = list(self._get_random_free_position())
        self.goal_pos = self._get_random_free_position()
        
        while tuple(self.agent_pos) == self.goal_pos:
            self.goal_pos = self._get_random_free_position()
        
        self.current_steps = 0
        return tuple(self.agent_pos)
    
    def step(self, action):
        """
        Effectue une action dans l'environnement.
        """
        self.current_steps += 1
        
        new_pos = self.agent_pos.copy()
        
        if action == self.UP:
            new_pos[0] -= 1
        elif action == self.DOWN:
            new_pos[0] += 1
        elif action == self.LEFT:
            new_pos[1] -= 1
        elif action == self.RIGHT:
            new_pos[1] += 1
        
        # Vérification des limites
        if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
            new_pos = self.agent_pos.copy()
            reward = self.step_cost
            done = False
        elif tuple(new_pos) in self.obstacles:
            reward = self.obstacle_reward
            done = False
            self.agent_pos = new_pos
        elif tuple(new_pos) == self.goal_pos:
            reward = self.goal_reward
            done = True
            self.agent_pos = new_pos
        else:
            reward = self.step_cost
            done = False
            self.agent_pos = new_pos
        
        if self.current_steps >= self.max_steps_per_episode:
            done = True
        
        return tuple(self.agent_pos), reward, done, {}
    
    def render(self, fig=None, ax=None, value_table=None):
        """
        Affiche la grille avec Matplotlib.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            ax.clear()
        
        # Grille de base avec value table si fournie
        if value_table is not None:
            vmin = np.min(value_table)
            vmax = np.max(value_table)
            
            for i in range(self.rows):
                for j in range(self.cols):
                    if vmax > vmin:
                        normalized_value = (value_table[i, j] - vmin) / (vmax - vmin)
                    else:
                        normalized_value = 0.5
                    
                    color = plt.cm.viridis(normalized_value)
                    rect = Rectangle((j, self.rows - 1 - i), 1, 1, 
                                    facecolor=color, edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
                    
                    ax.text(j + 0.5, self.rows - 1 - i + 0.5, 
                           f'{value_table[i, j]:.2f}',
                           ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    rect = Rectangle((j, self.rows - 1 - i), 1, 1, 
                                   facecolor='white', edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
        
        # Obstacles
        for obs in self.obstacles:
            rect = Rectangle((obs[1], self.rows - 1 - obs[0]), 1, 1, 
                           facecolor='gray', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(obs[1] + 0.5, self.rows - 1 - obs[0] + 0.5, 
                   'X', ha='center', va='center', fontsize=20, 
                   color='white', fontweight='bold')
        
        # Goal
        circle = Circle((self.goal_pos[1] + 0.5, self.rows - 1 - self.goal_pos[0] + 0.5),
                       0.3, facecolor='gold', edgecolor='orange', linewidth=3)
        ax.add_patch(circle)
        ax.text(self.goal_pos[1] + 0.5, self.rows - 1 - self.goal_pos[0] + 0.5, 
               'G', ha='center', va='center', fontsize=16, 
               color='green', fontweight='bold')
        
        # Agent
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
        ax.set_title('Dynamic GridWorld - Random Agent', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.pause(0.01)
        
        return fig, ax
