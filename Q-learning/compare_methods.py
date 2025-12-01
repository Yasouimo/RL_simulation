import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class MethodComparator:
    """
    Compare les performances des méthodes épisodique et itérative.
    """
    
    def __init__(self, episodic_stats_file, iterative_stats_file):
        """
        Initialise le comparateur avec les fichiers de statistiques.
        
        Args:
            episodic_stats_file: Chemin vers les stats épisodiques
            iterative_stats_file: Chemin vers les stats itératives
        """
        with open(episodic_stats_file, 'r') as f:
            self.episodic_data = json.load(f)
        
        with open(iterative_stats_file, 'r') as f:
            self.iterative_data = json.load(f)
    
    def calculate_metrics(self):
        """
        Calcule des métriques de comparaison détaillées.
        
        Returns:
            dict: Métriques pour les deux méthodes
        """
        metrics = {
            'episodic': self._compute_method_metrics(self.episodic_data),
            'iterative': self._compute_method_metrics(self.iterative_data)
        }
        
        return metrics
    
    def _compute_method_metrics(self, data):
        """
        Calcule les métriques pour une méthode donnée.
        """
        rewards = np.array(data['episode_rewards'])
        lengths = np.array(data['episode_lengths'])
        
        # Définir le seuil de succès (récompense > 5 signifie qu'on a atteint le goal)
        success_threshold = 5
        successes = rewards > success_threshold
        
        # Métriques globales
        metrics = {
            # Performance générale
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'median_reward': np.median(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            
            # Performance finale (100 derniers épisodes)
            'final_mean_reward': np.mean(rewards[-100:]),
            'final_std_reward': np.std(rewards[-100:]),
            'final_median_reward': np.median(rewards[-100:]),
            
            # Longueur des épisodes
            'mean_length': np.mean(lengths),
            'final_mean_length': np.mean(lengths[-100:]),
            
            # Taux de succès
            'overall_success_rate': np.mean(successes) * 100,
            'final_success_rate': np.mean(successes[-100:]) * 100,
            
            # Convergence
            'episodes_to_50_percent_success': self._episodes_to_threshold(successes, 0.5),
            'episodes_to_70_percent_success': self._episodes_to_threshold(successes, 0.7),
            
            # Stabilité (écart-type sur fenêtres glissantes)
            'early_stability': np.std(rewards[:100]) if len(rewards) >= 100 else np.std(rewards),
            'late_stability': np.std(rewards[-100:]),
            
            # Efficacité d'apprentissage
            'learning_speed': self._compute_learning_speed(rewards),
            'sample_efficiency': self._compute_sample_efficiency(rewards, successes),
            
            # Q-table
            'q_table_size': data['final_stats']['q_table_size'],
            'final_epsilon': data['final_stats']['epsilon'],
        }
        
        # Ajouter le nombre de mises à jour si disponible (itératif seulement)
        if 'update_count' in data['final_stats']:
            metrics['update_count'] = data['final_stats']['update_count']
        
        return metrics
    
    def _episodes_to_threshold(self, successes, threshold):
        """
        Calcule le nombre d'épisodes nécessaires pour atteindre un seuil de succès.
        """
        window_size = 100
        for i in range(window_size, len(successes)):
            if np.mean(successes[i-window_size:i]) >= threshold:
                return i
        return len(successes)  # Pas atteint
    
    def _compute_learning_speed(self, rewards):
        """
        Calcule la vitesse d'apprentissage (pente de la courbe de récompense).
        """
        if len(rewards) < 50:
            return 0
        
        # Calculer la pente sur les 200 premiers épisodes
        x = np.arange(min(200, len(rewards)))
        y = rewards[:len(x)]
        
        # Régression linéaire simple
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _compute_sample_efficiency(self, rewards, successes):
        """
        Efficacité d'échantillonnage : récompense moyenne / épisodes nécessaires.
        """
        episodes_to_converge = self._episodes_to_threshold(successes, 0.5)
        if episodes_to_converge == 0:
            return 0
        return np.mean(rewards[:episodes_to_converge]) / episodes_to_converge
    
    def print_comparison(self, metrics):
        """
        Affiche une comparaison détaillée des métriques.
        """
        print("="*80)
        print("COMPARAISON DES MÉTHODES Q-LEARNING")
        print("="*80)
        print()
        
        # Tableau de comparaison
        print(f"{'MÉTRIQUE':<45} {'ÉPISODIQUE':<15} {'ITÉRATIVE':<15} {'GAGNANT':<10}")
        print("-" * 80)
        
        comparisons = [
            ("Performance Globale", "", "", ""),
            ("  Récompense moyenne", 'mean_reward', 'mean_reward', 'higher'),
            ("  Récompense médiane", 'median_reward', 'median_reward', 'higher'),
            ("  Récompense max", 'max_reward', 'max_reward', 'higher'),
            ("", "", "", ""),
            
            ("Performance Finale (100 derniers)", "", "", ""),
            ("  Récompense moyenne", 'final_mean_reward', 'final_mean_reward', 'higher'),
            ("  Écart-type", 'final_std_reward', 'final_std_reward', 'lower'),
            ("  Longueur moyenne", 'final_mean_length', 'final_mean_length', 'lower'),
            ("", "", "", ""),
            
            ("Taux de Succès (%)", "", "", ""),
            ("  Global", 'overall_success_rate', 'overall_success_rate', 'higher'),
            ("  Final (100 derniers)", 'final_success_rate', 'final_success_rate', 'higher'),
            ("", "", "", ""),
            
            ("Convergence", "", "", ""),
            ("  Épisodes pour 50% succès", 'episodes_to_50_percent_success', 
             'episodes_to_50_percent_success', 'lower'),
            ("  Épisodes pour 70% succès", 'episodes_to_70_percent_success', 
             'episodes_to_70_percent_success', 'lower'),
            ("", "", "", ""),
            
            ("Stabilité", "", "", ""),
            ("  Début (std premiers 100)", 'early_stability', 'early_stability', 'lower'),
            ("  Fin (std derniers 100)", 'late_stability', 'late_stability', 'lower'),
            ("", "", "", ""),
            
            ("Efficacité", "", "", ""),
            ("  Vitesse d'apprentissage", 'learning_speed', 'learning_speed', 'higher'),
            ("  Efficacité d'échantillonnage", 'sample_efficiency', 
             'sample_efficiency', 'higher'),
            ("", "", "", ""),
            
            ("Autres", "", "", ""),
            ("  Taille Q-table", 'q_table_size', 'q_table_size', 'equal'),
            ("  Epsilon final", 'final_epsilon', 'final_epsilon', 'equal'),
        ]
        
        score_episodic = 0
        score_iterative = 0
        
        for item in comparisons:
            if len(item[1]) == 0:  # Ligne de titre ou vide
                print(f"{item[0]:<45}")
                continue
            
            metric_name = item[0]
            episodic_key = item[1]
            iterative_key = item[2]
            comparison_type = item[3]
            
            episodic_val = metrics['episodic'].get(episodic_key, 0)
            iterative_val = metrics['iterative'].get(iterative_key, 0)
            
            # Déterminer le gagnant
            winner = ""
            if comparison_type == 'higher':
                if episodic_val > iterative_val:
                    winner = "📗 Épisodique"
                    score_episodic += 1
                elif iterative_val > episodic_val:
                    winner = "📘 Itérative"
                    score_iterative += 1
                else:
                    winner = "⚖️ Égalité"
            elif comparison_type == 'lower':
                if episodic_val < iterative_val:
                    winner = "📗 Épisodique"
                    score_episodic += 1
                elif iterative_val < episodic_val:
                    winner = "📘 Itérative"
                    score_iterative += 1
                else:
                    winner = "⚖️ Égalité"
            else:  # equal
                winner = "⚖️ Égalité"
            
            print(f"{metric_name:<45} {episodic_val:<15.3f} {iterative_val:<15.3f} {winner:<10}")
        
        print("-" * 80)
        print()
        
        # Résumé
        print("="*80)
        print("RÉSUMÉ")
        print("="*80)
        print(f"Score Épisodique: {score_episodic} points")
        print(f"Score Itérative: {score_iterative} points")
        print()
        
        if score_iterative > score_episodic:
            print("🏆 GAGNANT: MÉTHODE ITÉRATIVE")
            advantage = score_iterative - score_episodic
            print(f"   Avantage de {advantage} points")
        elif score_episodic > score_iterative:
            print("🏆 GAGNANT: MÉTHODE ÉPISODIQUE")
            advantage = score_episodic - score_iterative
            print(f"   Avantage de {advantage} points")
        else:
            print("⚖️ ÉGALITÉ PARFAITE")
        
        print()
        
        # Analyse détaillée
        print("="*80)
        print("ANALYSE DÉTAILLÉE")
        print("="*80)
        print()
        
        print("📊 Points Forts de la Méthode ITÉRATIVE:")
        if metrics['iterative']['final_success_rate'] > metrics['episodic']['final_success_rate']:
            diff = metrics['iterative']['final_success_rate'] - metrics['episodic']['final_success_rate']
            print(f"  ✓ Meilleur taux de succès final: +{diff:.1f}%")
        
        if metrics['iterative']['final_mean_reward'] > metrics['episodic']['final_mean_reward']:
            diff = metrics['iterative']['final_mean_reward'] - metrics['episodic']['final_mean_reward']
            print(f"  ✓ Meilleure récompense finale: +{diff:.2f}")
        
        if metrics['iterative']['learning_speed'] > metrics['episodic']['learning_speed']:
            print(f"  ✓ Apprentissage plus rapide")
        
        if 'update_count' in metrics['iterative']:
            print(f"  ✓ Nombre de mises à jour: {metrics['iterative']['update_count']}")
        
        print()
        print("📊 Points Forts de la Méthode ÉPISODIQUE:")
        if metrics['episodic']['late_stability'] < metrics['iterative']['late_stability']:
            print(f"  ✓ Plus stable en fin d'entraînement")
        
        if metrics['episodic']['final_success_rate'] > metrics['iterative']['final_success_rate']:
            diff = metrics['episodic']['final_success_rate'] - metrics['iterative']['final_success_rate']
            print(f"  ✓ Meilleur taux de succès final: +{diff:.1f}%")
        
        print()
        
        # Recommandation
        print("="*80)
        print("RECOMMANDATION")
        print("="*80)
        print()
        
        if score_iterative > score_episodic + 2:
            print("💡 Pour ce problème, la méthode ITÉRATIVE est recommandée:")
            print("   - Apprentissage plus efficace")
            print("   - Meilleures performances finales")
            print("   - Convergence plus rapide")
        elif score_episodic > score_iterative + 2:
            print("💡 Pour ce problème, la méthode ÉPISODIQUE est recommandée:")
            print("   - Plus stable")
            print("   - Meilleure généralisation")
        else:
            print("💡 Les deux méthodes sont équivalentes pour ce problème.")
            print("   Choisir selon les préférences:")
            print("   - Itérative: plus standard et rapide")
            print("   - Épisodique: plus stable et théorique")
        
        print()
    
    def plot_comparison(self, save_path='comparison_plots.png'):
        """
        Crée des graphiques de comparaison.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comparaison des Méthodes Q-Learning', fontsize=16, fontweight='bold')
        
        episodic_rewards = np.array(self.episodic_data['episode_rewards'])
        iterative_rewards = np.array(self.iterative_data['episode_rewards'])
        
        episodic_lengths = np.array(self.episodic_data['episode_lengths'])
        iterative_lengths = np.array(self.iterative_data['episode_lengths'])
        
        # 1. Récompenses brutes
        ax = axes[0, 0]
        ax.plot(episodic_rewards, alpha=0.3, color='blue', label='Épisodique')
        ax.plot(iterative_rewards, alpha=0.3, color='green', label='Itérative')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense')
        ax.set_title('Récompenses par Épisode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Moyennes mobiles
        ax = axes[0, 1]
        window = 50
        if len(episodic_rewards) >= window:
            ep_smooth = np.convolve(episodic_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episodic_rewards)), ep_smooth, 
                   color='blue', linewidth=2, label='Épisodique (MA-50)')
        
        if len(iterative_rewards) >= window:
            it_smooth = np.convolve(iterative_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(iterative_rewards)), it_smooth, 
                   color='green', linewidth=2, label='Itérative (MA-50)')
        
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense (Moyenne Mobile)')
        ax.set_title('Récompenses Lissées')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Taux de succès cumulatif
        ax = axes[0, 2]
        window = 100
        ep_success_rate = []
        it_success_rate = []
        
        for i in range(window, len(episodic_rewards)):
            ep_success_rate.append(np.mean(episodic_rewards[i-window:i] > 5) * 100)
        
        for i in range(window, len(iterative_rewards)):
            it_success_rate.append(np.mean(iterative_rewards[i-window:i] > 5) * 100)
        
        ax.plot(range(window, len(episodic_rewards)), ep_success_rate, 
               color='blue', linewidth=2, label='Épisodique')
        ax.plot(range(window, len(iterative_rewards)), it_success_rate, 
               color='green', linewidth=2, label='Itérative')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Taux de Succès (%)')
        ax.set_title(f'Taux de Succès (Fenêtre {window})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        # 4. Longueur des épisodes
        ax = axes[1, 0]
        ax.plot(episodic_lengths, alpha=0.3, color='blue', label='Épisodique')
        ax.plot(iterative_lengths, alpha=0.3, color='green', label='Itérative')
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Longueur (steps)')
        ax.set_title('Longueur des Épisodes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Distribution des récompenses finales
        ax = axes[1, 1]
        ax.hist(episodic_rewards[-100:], bins=20, alpha=0.5, color='blue', 
               label='Épisodique', edgecolor='black')
        ax.hist(iterative_rewards[-100:], bins=20, alpha=0.5, color='green', 
               label='Itérative', edgecolor='black')
        ax.set_xlabel('Récompense')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution (100 derniers épisodes)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Box plot comparatif
        ax = axes[1, 2]
        data_to_plot = [episodic_rewards[-100:], iterative_rewards[-100:]]
        bp = ax.boxplot(data_to_plot, labels=['Épisodique', 'Itérative'],
                       patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        ax.set_ylabel('Récompense')
        ax.set_title('Comparaison (100 derniers épisodes)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphiques sauvegardés: {save_path}")
        
        return fig


def compare_latest_results():
    """
    Compare les résultats les plus récents des deux méthodes.
    """
    # Trouver les fichiers les plus récents
    episodic_folder = "episodic/results_episodic"
    iterative_folder = "iterative/results_iterative"
    
    episodic_files = [f for f in os.listdir(episodic_folder) if f.endswith('.json')]
    iterative_files = [f for f in os.listdir(iterative_folder) if f.endswith('.json')]
    
    if not episodic_files or not iterative_files:
        print("Erreur: Fichiers de statistiques non trouvés!")
        return
    
    episodic_latest = sorted(episodic_files)[-1]
    iterative_latest = sorted(iterative_files)[-1]
    
    episodic_path = os.path.join(episodic_folder, episodic_latest)
    iterative_path = os.path.join(iterative_folder, iterative_latest)
    
    print(f"Comparaison des fichiers:")
    print(f"  Épisodique: {episodic_latest}")
    print(f"  Itérative: {iterative_latest}")
    print()
    
    # Créer le comparateur
    comparator = MethodComparator(episodic_path, iterative_path)
    
    # Calculer les métriques
    metrics = comparator.calculate_metrics()
    
    # Afficher la comparaison
    comparator.print_comparison(metrics)
    
    # Créer les graphiques
    output_folder = "comparison_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_folder, f"comparison_{timestamp}.png")
    
    comparator.plot_comparison(save_path=plot_path)
    
    plt.show()


if __name__ == "__main__":
    compare_latest_results()
