import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv
from agents import RandomAgent, ValueIterationAgent
import time
import json
import os
from datetime import datetime


def load_config_from_file(config_file='config.json'):
    """
    Charge la configuration depuis un fichier JSON.
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Convertir les listes en tuples pour les positions
        config['start_pos'] = tuple(config['start_pos'])
        config['goal_pos'] = tuple(config['goal_pos'])
        config['obstacles'] = [tuple(obs) for obs in config['obstacles']]
        
        # Mapper la vitesse
        speed_map = {1: 1.0, 2: 0.5, 3: 0.3, 4: 0.1}
        delay = speed_map.get(config.get('animation_speed', 3), 0.3)
        
        return (config['grid_size'], config['start_pos'], config['goal_pos'], 
                config['obstacles'], delay, config.get('save_figures', True),
                config.get('output_folder', 'results'))
    except FileNotFoundError:
        print(f"Fichier {config_file} non trouvé. Utilisation du mode interactif.")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de configuration: {e}")
        return None


def get_user_config():
    """
    Demande à l'utilisateur de configurer l'environnement.
    """
    print("="*60)
    print("CONFIGURATION DU GRIDWORLD")
    print("="*60)
    
    # Taille de la grille
    while True:
        try:
            grid_size = int(input("\nTaille de la grille (ex: 5 pour 5x5): "))
            if grid_size < 3:
                print("La grille doit être au moins 3x3")
                continue
            break
        except ValueError:
            print("Veuillez entrer un nombre valide")
    
    # Position de départ
    print(f"\nPosition de départ (entre 0 et {grid_size-1})")
    while True:
        try:
            start_row = int(input(f"  Ligne de départ (0-{grid_size-1}): "))
            start_col = int(input(f"  Colonne de départ (0-{grid_size-1}): "))
            if 0 <= start_row < grid_size and 0 <= start_col < grid_size:
                start_pos = (start_row, start_col)
                break
            else:
                print(f"Les valeurs doivent être entre 0 et {grid_size-1}")
        except ValueError:
            print("Veuillez entrer des nombres valides")
    
    # Position du but
    print(f"\nPosition du but (entre 0 et {grid_size-1})")
    while True:
        try:
            goal_row = int(input(f"  Ligne du but (0-{grid_size-1}): "))
            goal_col = int(input(f"  Colonne du but (0-{grid_size-1}): "))
            if 0 <= goal_row < grid_size and 0 <= goal_col < grid_size:
                if (goal_row, goal_col) == start_pos:
                    print("Le but ne peut pas être à la même position que le départ!")
                    continue
                goal_pos = (goal_row, goal_col)
                break
            else:
                print(f"Les valeurs doivent être entre 0 et {grid_size-1}")
        except ValueError:
            print("Veuillez entrer des nombres valides")
    
    # Obstacles
    print(f"\nCombien d'obstacles voulez-vous? (0-{grid_size*grid_size-2})")
    while True:
        try:
            num_obstacles = int(input("  Nombre d'obstacles: "))
            if 0 <= num_obstacles < grid_size * grid_size - 1:
                break
            else:
                print("Nombre d'obstacles invalide")
        except ValueError:
            print("Veuillez entrer un nombre valide")
    
    obstacles = []
    for i in range(num_obstacles):
        while True:
            try:
                print(f"\nObstacle {i+1}/{num_obstacles}")
                obs_row = int(input(f"  Ligne (0-{grid_size-1}): "))
                obs_col = int(input(f"  Colonne (0-{grid_size-1}): "))
                obs_pos = (obs_row, obs_col)
                
                if not (0 <= obs_row < grid_size and 0 <= obs_col < grid_size):
                    print(f"Les valeurs doivent être entre 0 et {grid_size-1}")
                    continue
                    
                if obs_pos == start_pos:
                    print("L'obstacle ne peut pas être sur la position de départ!")
                    continue
                    
                if obs_pos == goal_pos:
                    print("L'obstacle ne peut pas être sur la position du but!")
                    continue
                    
                if obs_pos in obstacles:
                    print("Il y a déjà un obstacle à cette position!")
                    continue
                
                obstacles.append(obs_pos)
                break
            except ValueError:
                print("Veuillez entrer des nombres valides")
    
    # Vitesse d'animation
    print("\nVitesse d'animation")
    print("  1 - Très lent (1.0 sec/étape)")
    print("  2 - Lent (0.5 sec/étape)")
    print("  3 - Normal (0.3 sec/étape)")
    print("  4 - Rapide (0.1 sec/étape)")
    while True:
        try:
            speed = int(input("Choisissez la vitesse (1-4): "))
            if speed == 1:
                delay = 1.0
            elif speed == 2:
                delay = 0.5
            elif speed == 3:
                delay = 0.3
            elif speed == 4:
                delay = 0.1
            else:
                print("Choisissez entre 1 et 4")
                continue
            break
        except ValueError:
            print("Veuillez entrer un nombre valide")
    
    # Sauvegarder les figures
    print("\nSauvegarder les figures?")
    save_figures = input("Oui (O) / Non (N): ").strip().upper() == 'O'
    
    output_folder = "results"
    if save_figures:
        output_folder = input("Dossier de sortie (par défaut: results): ").strip() or "results"
    
    return grid_size, start_pos, goal_pos, obstacles, delay, save_figures, output_folder


def main():
    """
    Script principal pour démontrer le fonctionnement du GridWorld
    avec un agent aléatoire et un agent Value Iteration.
    """
    
    # ===========================
    # CONFIGURATION
    # ===========================
    print("\nChoisissez le mode de configuration:")
    print("  1 - Charger depuis config.json")
    print("  2 - Configuration interactive")
    
    while True:
        try:
            choice = int(input("Votre choix (1 ou 2): "))
            if choice in [1, 2]:
                break
            print("Veuillez choisir 1 ou 2")
        except ValueError:
            print("Veuillez entrer un nombre valide")
    
    if choice == 1:
        config = load_config_from_file()
        if config is None:
            print("\nPassage au mode interactif...")
            config = get_user_config()
    else:
        config = get_user_config()
    
    grid_size, start_pos, goal_pos, obstacles, delay, save_figures, output_folder = config
    
    # Créer le dossier de sortie si nécessaire
    if save_figures and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nDossier '{output_folder}' créé pour sauvegarder les figures.")
    
    # ===========================
    # CRÉATION DE L'ENVIRONNEMENT
    # ===========================
    print("\n" + "="*60)
    print(f"CRÉATION DE L'ENVIRONNEMENT GRIDWORLD ({grid_size}x{grid_size})")
    print("="*60)
    
    env = GridWorldEnv(
        grid_size=grid_size,
        start_pos=start_pos,
        goal_pos=goal_pos,
        obstacles=obstacles,
        step_cost=-0.01
    )
    
    print(f"Grille: {env.rows}x{env.cols}")
    print(f"Position de départ: {env.start_pos}")
    print(f"Position du but: {env.goal_pos}")
    print(f"Obstacles: {env.obstacles}")
    print()
    
    # Créer UNE SEULE fenêtre pour toute la simulation
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # ===========================
    # PARTIE 1: AGENT ALÉATOIRE
    # ===========================
    print("="*60)
    print("PARTIE 1: AGENT ALÉATOIRE (10 ÉTAPES)")
    print("="*60)
    
    random_agent = RandomAgent(num_actions=4)
    state = env.reset()
    
    for step in range(10):
        print(f"\nÉtape {step + 1}:")
        print(f"  Position actuelle: {state}")
        
        # Choisir une action aléatoire
        action = random_agent.choose_action(state)
        action_names = ['HAUT', 'BAS', 'GAUCHE', 'DROITE']
        print(f"  Action choisie: {action_names[action]}")
        
        # Effectuer l'action
        next_state, reward, done, _ = env.step(action)
        print(f"  Nouvelle position: {next_state}")
        print(f"  Récompense: {reward}")
        
        # Afficher la grille dans la MÊME fenêtre
        fig, ax = env.render(fig=fig, ax=ax)
        time.sleep(delay)  # Pause configurable
        
        state = next_state
        
        if done:
            print("\n*** But atteint! ***")
            break
    
    # ===========================
    # PARTIE 2: VALUE ITERATION
    # ===========================
    print("\n" + "="*60)
    print("PARTIE 2: ENTRAÎNEMENT VALUE ITERATION")
    print("="*60)
    
    # Réinitialiser l'environnement
    env.reset()
    
    # Créer et entraîner l'agent Value Iteration
    vi_agent = ValueIterationAgent(gamma=0.9, theta=1e-6)
    value_table = vi_agent.train(env, max_iterations=1000)
    
    print("\nTable des valeurs après convergence:")
    print(value_table)
    
    # ===========================
    # PARTIE 3: VISUALISATION
    # ===========================
    print("\n" + "="*60)
    print("PARTIE 3: VISUALISATION DES STATE VALUES")
    print("="*60)
    
    # Réinitialiser l'environnement
    env.reset()
    
    # Afficher la grille avec les valeurs d'états dans la MÊME fenêtre
    print("\nAffichage de la grille avec les valeurs d'états...")
    print("Les valeurs augmentent en s'approchant du but (case dorée 'G').")
    print("Les cases plus claires ont des valeurs plus élevées.")
    fig, ax = env.render(value_table=value_table, fig=fig, ax=ax)
    ax.set_title('GridWorld - State Values après Value Iteration', 
                fontsize=16, fontweight='bold')
    
    # Sauvegarder la figure
    if save_figures:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"value_table_{grid_size}x{grid_size}_{timestamp}.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure sauvegardée: {filepath}")
        
        # Sauvegarder aussi les valeurs dans un fichier texte
        txt_filename = f"value_table_{grid_size}x{grid_size}_{timestamp}.txt"
        txt_filepath = os.path.join(output_folder, txt_filename)
        np.savetxt(txt_filepath, value_table, fmt='%.6f')
        print(f"✓ Valeurs sauvegardées: {txt_filepath}")
    
    plt.pause(5)  # Pause pour voir les valeurs
    
    # ===========================
    # PARTIE 4: AGENT ENTRAÎNÉ
    # ===========================
    print("\n" + "="*60)
    print("PARTIE 4: L'AGENT ENTRAÎNÉ ATTEINT LE BUT")
    print("="*60)
    
    # Réinitialiser l'environnement
    state = env.reset()
    
    print("\nL'agent utilise maintenant la politique optimale apprise...")
    max_steps = 50
    
    for step in range(max_steps):
        print(f"\nÉtape {step + 1}:")
        print(f"  Position: {state}")
        
        # Choisir la meilleure action selon la politique apprise
        action = vi_agent.choose_action(state)
        action_names = ['HAUT', 'BAS', 'GAUCHE', 'DROITE']
        print(f"  Meilleure action: {action_names[action]}")
        
        # Effectuer l'action
        next_state, reward, done, _ = env.step(action)
        print(f"  Nouvelle position: {next_state}")
        print(f"  Récompense: {reward}")
        
        # Afficher la grille dans la MÊME fenêtre
        fig, ax = env.render(fig=fig, ax=ax)
        time.sleep(delay)  # Pause configurable
        
        state = next_state
        
        if done:
            print("\n" + "="*60)
            print("*** BUT ATTEINT AVEC SUCCÈS! ***")
            print(f"L'agent a atteint le but en {step + 1} étapes!")
            print("="*60)
            break
    
    plt.ioff()
    
    # Garder la fenêtre ouverte
    print("\nFermez la fenêtre pour terminer le programme.")
    plt.show()


if __name__ == "__main__":
    main()
