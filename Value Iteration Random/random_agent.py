import numpy as np


class RandomAgent:
    """
    Agent qui choisit des actions complètement aléatoires.
    Ne fait AUCUN apprentissage - utilisé comme baseline de comparaison.
    """
    
    def __init__(self, num_actions=4):
        """
        Initialise l'agent aléatoire.
        
        Args:
            num_actions: Nombre d'actions possibles
        """
        self.num_actions = num_actions
        self.actions_taken = 0
    
    def get_action(self, state=None):
        """
        Choisit une action complètement aléatoire.
        
        Args:
            state: État actuel (ignoré pour un agent aléatoire)
            
        Returns:
            action: Action aléatoire
        """
        self.actions_taken += 1
        return np.random.randint(0, self.num_actions)
    
    def get_stats(self):
        """
        Retourne des statistiques sur l'agent.
        
        Returns:
            dict: Statistiques
        """
        return {
            'type': 'Random',
            'actions_taken': self.actions_taken,
            'learning': False
        }
