import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from grid_env_dynamic import DynamicGridWorldEnv
from q_agent_iterative import QLearningAgentIterative
import json
import os
from datetime import datetime
import time


def visualize_qtable(agent, env, ax, episode):
    """
    Visualise la Q-table sous forme de heatmap.
    Affiche la meilleure valeur Q pour chaque position de la grille.
    """
    ax.clear()
    
    # Créer une matrice pour stocker les meilleures valeurs Q
    q_values_grid = np.zeros((env.rows, env.cols))
    
    # Pour chaque position de la grille
    for i in range(env.rows):
        for j in range(env.cols):
            if (i, j) in env.obstacles:
                q_values_grid[i, j] = np.nan  # NaN pour les obstacles
            else:
                # Calculer la distance moyenne au goal pour cette position
                # On utilise la position du goal actuel comme référence
                distance = abs(i - env.goal_pos[0]) + abs(j - env.goal_pos[1])
                state = (i, j, distance)
                
                # Obtenir les valeurs Q pour cet état
                q_vals = agent.get_q_values(state)
                if len(q_vals) > 0:
                    q_values_grid[i, j] = np.max(q_vals)  # Meilleure action
                else:
                    q_values_grid[i, j] = 0
    
    # Créer la heatmap
    im = ax.imshow(q_values_grid, cmap='RdYlGn', aspect='auto')
    
    # Ajouter les valeurs dans les cases
    for i in range(env.rows):
        for j in range(env.cols):
            if not np.isnan(q_values_grid[i, j]):
                value = q_values_grid[i, j]
                color = 'white' if value < 0 else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                       color=color, fontsize=8, fontweight='bold')
            else:
                ax.text(j, i, 'X', ha='center', va='center',
                       color='white', fontsize=16, fontweight='bold')
    
    # Marquer le goal actuel
    goal_i, goal_j = env.goal_pos
    rect = Rectangle((goal_j - 0.5, goal_i - 0.5), 1, 1,
                     fill=False, edgecolor='gold', linewidth=4)
    ax.add_patch(rect)
    
    ax.set_title(f'Q-Table Heatmap (Épisode {episode})', fontweight='bold')
    ax.set_xlabel('Colonne')
    ax.set_ylabel('Ligne')
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    
    # Ajouter la colorbar
    plt.colorbar(im, ax=ax, label='Max Q-Value')


def display_info(ax, episode, stats, avg_reward, avg_length, 
                episode_rewards, episode_lengths):
    """
    Affiche les informations textuelles sur l'entraînement.
    """
    ax.clear()
    ax.axis('off')
    
    info_text = f"""
    📊 STATISTIQUES D'ENTRAÎNEMENT (ITÉRATIVE)
    
    Épisode: {episode}
    
    Performance (10 derniers):
      • Récompense moyenne: {avg_reward:.2f}
      • Longueur moyenne: {avg_length:.1f} steps
    
    Agent:
      • Epsilon: {stats['epsilon']:.3f}
      • Taille Q-table: {stats['q_table_size']}
      • Mises à jour: {stats['update_count']}
      • Learning rate: {stats['learning_rate']:.3f}
      • Gamma: {stats['gamma']:.2f}
    
    Progression globale:
      • Récompense max: {max(episode_rewards):.2f}
      • Récompense min: {min(episode_rewards):.2f}
      • Moyenne totale: {np.mean(episode_rewards):.2f}
    
    Succès récents (100 derniers):
    """
    
    if len(episode_rewards) >= 100:
        recent_success = sum(1 for r in episode_rewards[-100:] if r > 5)
        info_text += f"      • {recent_success}% d'épisodes réussis"
    else:
        recent_success = sum(1 for r in episode_rewards if r > 5)
        total = len(episode_rewards)
        info_text += f"      • {recent_success}/{total} épisodes réussis"
    
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))


def train_iterative(num_episodes=500, grid_size=5, render_frequency=50):
    """
    Entraîne un agent Q-Learning de manière itérative.
    Met à jour la Q-table après chaque transition (step).
    
    Args:
        num_episodes: Nombre d'épisodes d'entraînement
        grid_size: Taille de la grille
        render_frequency: Fréquence d'affichage (tous les N épisodes)
    """
    print("="*60)
    print("Q-LEARNING - MÉTHODE ITÉRATIVE")
    print("="*60)
    print(f"Nombre d'épisodes: {num_episodes}")
    print(f"Taille de la grille: {grid_size}x{grid_size}")
    print()
    
    # Créer l'environnement
    env = DynamicGridWorldEnv(
        grid_size=grid_size,
        obstacles=[(2, 2)],
        step_cost=-0.01,
        goal_reward=10.0,
        max_steps_per_episode=100
    )
    
    # Créer l'agent
    agent = QLearningAgentIterative(
        num_actions=4,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Statistiques d'entraînement
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    
    # Configuration de la visualisation
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_env = axes[0, 0]
    ax_stats = axes[0, 1]
    ax_qtable = axes[1, 0]
    ax_info = axes[1, 1]
    
    print("Début de l'entraînement...")
    print()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Exécuter l'épisode avec mise à jour itérative
        while not done:
            # Choisir une action
            action = agent.get_action(state, training=True)
            
            # Effectuer l'action
            next_state, reward, done, _ = env.step(action)
            
            # MISE À JOUR ITÉRATIVE: immédiatement après chaque transition
            agent.update(state, action, reward, next_state, done)
            
            # Accumuler les statistiques
            episode_reward += reward
            episode_length += 1
            
            state = next_state
        
        # Décroissance d'epsilon à la fin de l'épisode
        agent.decay_epsilon()
        
        # Enregistrer les statistiques
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculer le taux de succès (moyenne des 100 derniers épisodes)
        if len(episode_rewards) >= 100:
            recent_rewards = episode_rewards[-100:]
            success = sum(1 for r in recent_rewards if r > 5) / 100
            success_rate.append(success)
        
        # Affichage périodique
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            stats = agent.get_stats()
            print(f"Épisode {episode + 1}/{num_episodes}")
            print(f"  Récompense moyenne (10 derniers): {avg_reward:.2f}")
            print(f"  Longueur moyenne (10 derniers): {avg_length:.1f}")
            print(f"  Epsilon: {stats['epsilon']:.3f}")
            print(f"  Taille Q-table: {stats['q_table_size']}")
            print(f"  Nombre de mises à jour: {stats['update_count']}")
            print()
        
        # Visualisation périodique
        if (episode + 1) % render_frequency == 0:
            # Afficher l'environnement
            env.render(fig=fig, ax=ax_env)
            
            # Visualiser la Q-table
            visualize_qtable(agent, env, ax_qtable, episode + 1)
            
            # Afficher les informations textuelles
            display_info(ax_info, episode + 1, stats, avg_reward, avg_length, 
                        episode_rewards, episode_lengths)
            
            # Afficher les courbes de statistiques
            ax_stats.clear()
            
            # Sous-graphique pour les récompenses
            ax_stats.plot(episode_rewards, alpha=0.3, color='green', label='Récompense')
            if len(episode_rewards) >= 10:
                moving_avg = np.convolve(episode_rewards, 
                                        np.ones(10)/10, mode='valid')
                ax_stats.plot(range(9, len(episode_rewards)), moving_avg, 
                            color='green', linewidth=2, label='Moyenne mobile (10)')
            
            ax_stats.set_xlabel('Épisode')
            ax_stats.set_ylabel('Récompense')
            ax_stats.set_title('Progression de l\'entraînement (Itératif)')
            ax_stats.legend()
            ax_stats.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.5)  # Pause plus longue pour mieux voir
            time.sleep(0.3)  # Délai supplémentaire
    
    print("="*60)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    
    # Statistiques finales
    print(f"\nStatistiques finales:")
    print(f"  Récompense moyenne (100 derniers): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Longueur moyenne (100 derniers): {np.mean(episode_lengths[-100:]):.1f}")
    if success_rate:
        print(f"  Taux de succès (100 derniers): {success_rate[-1]*100:.1f}%")
    stats = agent.get_stats()
    print(f"  Taille finale Q-table: {stats['q_table_size']}")
    print(f"  Epsilon final: {stats['epsilon']:.3f}")
    print(f"  Nombre total de mises à jour: {stats['update_count']}")
    
    # Sauvegarder les résultats
    output_folder = "results_iterative"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder les courbes
    plt.savefig(os.path.join(output_folder, f"training_iterative_{timestamp}.png"), 
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Courbes sauvegardées dans {output_folder}/")
    
    # Sauvegarder les statistiques
    stats_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'final_stats': stats,
        'config': {
            'num_episodes': num_episodes,
            'grid_size': grid_size,
            'learning_rate': agent.lr,
            'gamma': agent.gamma
        }
    }
    
    with open(os.path.join(output_folder, f"stats_iterative_{timestamp}.json"), 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"✓ Statistiques sauvegardées dans {output_folder}/")
    
    plt.ioff()
    plt.show()
    
    return agent, env


if __name__ == "__main__":
    agent, env = train_iterative(num_episodes=500, grid_size=5, render_frequency=50)
