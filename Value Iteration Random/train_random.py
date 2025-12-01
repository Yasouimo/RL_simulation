import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from grid_env_dynamic import DynamicGridWorldEnv
from random_agent import RandomAgent
import json
import os
from datetime import datetime
import time


def display_info(ax, episode, stats, avg_reward, avg_length, 
                episode_rewards, episode_lengths):
    """
    Affiche les informations textuelles sur l'entraînement.
    """
    ax.clear()
    ax.axis('off')
    
    info_text = f"""
    🎲 AGENT ALÉATOIRE (BASELINE)
    
    Épisode: {episode}
    
    Performance (10 derniers):
      • Récompense moyenne: {avg_reward:.2f}
      • Longueur moyenne: {avg_length:.1f} steps
    
    Agent:
      • Type: Random (pas d'apprentissage)
      • Actions prises: {stats['actions_taken']}
      • Exploration: 100% (toujours aléatoire)
    
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
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))


def train_random(num_episodes=500, grid_size=5, render_frequency=50):
    """
    Fait jouer un agent aléatoire (baseline sans apprentissage).
    
    Args:
        num_episodes: Nombre d'épisodes
        grid_size: Taille de la grille
        render_frequency: Fréquence d'affichage
    """
    print("="*60)
    print("AGENT ALÉATOIRE - BASELINE (PAS D'APPRENTISSAGE)")
    print("="*60)
    print(f"Nombre d'épisodes: {num_episodes}")
    print(f"Taille de la grille: {grid_size}x{grid_size}")
    print()
    print("⚠️  Cet agent choisit des actions ALÉATOIRES")
    print("⚠️  Il ne fait AUCUN apprentissage")
    print("⚠️  Il sert de BASELINE pour comparer avec les méthodes intelligentes")
    print()
    
    # Créer l'environnement
    env = DynamicGridWorldEnv(
        grid_size=grid_size,
        obstacles=[(2, 2)],
        step_cost=-0.01,
        goal_reward=10.0,
        max_steps_per_episode=100
    )
    
    # Créer l'agent aléatoire
    agent = RandomAgent(num_actions=4)
    
    # Statistiques
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    
    # Configuration de la visualisation
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_env = axes[0, 0]
    ax_stats = axes[0, 1]
    ax_comparison = axes[1, 0]
    ax_info = axes[1, 1]
    
    print("Début de l'exécution...")
    print()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Exécuter l'épisode avec actions aléatoires
        while not done:
            # Action COMPLÈTEMENT ALÉATOIRE
            action = agent.get_action(state)
            
            # Effectuer l'action
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
        
        # Enregistrer les statistiques
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Calculer le taux de succès
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
            print(f"  Actions aléatoires prises: {stats['actions_taken']}")
            print()
        
        # Visualisation périodique
        if (episode + 1) % render_frequency == 0:
            # Afficher l'environnement
            env.render(fig=fig, ax=ax_env)
            
            # Afficher les informations
            stats = agent.get_stats()
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            display_info(ax_info, episode + 1, stats, avg_reward, avg_length,
                        episode_rewards, episode_lengths)
            
            # Courbe de performance
            ax_stats.clear()
            ax_stats.plot(episode_rewards, alpha=0.3, color='red', label='Récompense')
            if len(episode_rewards) >= 10:
                moving_avg = np.convolve(episode_rewards, 
                                        np.ones(10)/10, mode='valid')
                ax_stats.plot(range(9, len(episode_rewards)), moving_avg, 
                            color='red', linewidth=2, label='Moyenne mobile (10)')
            ax_stats.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax_stats.set_xlabel('Épisode')
            ax_stats.set_ylabel('Récompense')
            ax_stats.set_title('Performance de l\'Agent Aléatoire')
            ax_stats.legend()
            ax_stats.grid(True, alpha=0.3)
            
            # Comparaison avec une baseline théorique
            ax_comparison.clear()
            ax_comparison.hist(episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards,
                             bins=20, alpha=0.7, color='red', edgecolor='black')
            ax_comparison.axvline(x=0, color='gray', linestyle='--', linewidth=2, label='Seuil neutre')
            ax_comparison.axvline(x=5, color='green', linestyle='--', linewidth=2, label='Seuil succès')
            ax_comparison.set_xlabel('Récompense')
            ax_comparison.set_ylabel('Fréquence')
            ax_comparison.set_title('Distribution des Récompenses')
            ax_comparison.legend()
            ax_comparison.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.pause(0.5)
            time.sleep(0.3)
    
    print("="*60)
    print("EXÉCUTION TERMINÉE")
    print("="*60)
    
    # Statistiques finales
    print(f"\nStatistiques finales de l'Agent Aléatoire:")
    print(f"  Récompense moyenne (100 derniers): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Longueur moyenne (100 derniers): {np.mean(episode_lengths[-100:]):.1f}")
    if success_rate:
        print(f"  Taux de succès (100 derniers): {success_rate[-1]*100:.1f}%")
    else:
        recent_success = sum(1 for r in episode_rewards if r > 5)
        print(f"  Taux de succès: {recent_success/len(episode_rewards)*100:.1f}%")
    print(f"  Actions aléatoires totales: {agent.get_stats()['actions_taken']}")
    
    # Sauvegarder les résultats
    output_folder = "results_random"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sauvegarder les courbes
    plt.savefig(os.path.join(output_folder, f"random_baseline_{timestamp}.png"), 
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Courbes sauvegardées dans {output_folder}/")
    
    # Sauvegarder les statistiques
    stats_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'final_stats': agent.get_stats(),
        'config': {
            'num_episodes': num_episodes,
            'grid_size': grid_size,
            'agent_type': 'Random'
        }
    }
    
    with open(os.path.join(output_folder, f"stats_random_{timestamp}.json"), 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"✓ Statistiques sauvegardées dans {output_folder}/")
    
    print()
    print("="*60)
    print("📊 COMPARAISON ATTENDUE AVEC LES MÉTHODES INTELLIGENTES:")
    print("="*60)
    print("L'agent aléatoire devrait avoir:")
    print("  ❌ Très faible taux de succès (~0-10%)")
    print("  ❌ Récompenses majoritairement négatives")
    print("  ❌ Pas d'amélioration au fil du temps")
    print("  ❌ Longueur maximale des épisodes (timeout)")
    print()
    print("Comparé à Q-Learning qui devrait avoir:")
    print("  ✓ Taux de succès ~50-70%")
    print("  ✓ Récompenses positives")
    print("  ✓ Amélioration visible")
    print("  ✓ Épisodes courts et efficaces")
    print("="*60)
    
    plt.ioff()
    plt.show()
    
    return agent, env


if __name__ == "__main__":
    agent, env = train_random(num_episodes=500, grid_size=5, render_frequency=50)
