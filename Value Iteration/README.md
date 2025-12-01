# GridWorld - Reinforcement Learning

Projet d'Apprentissage par Renforcement avec environnement GridWorld.

## 📋 Structure du Projet

```
RL_exo/
├── grid_env.py          # Environnement GridWorld
├── agents.py            # Agents (Random et Value Iteration)
├── main.py              # Script principal
├── config.json          # Fichier de configuration
├── requirements.txt     # Dépendances Python
└── results/             # Dossier des résultats (créé automatiquement)
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Méthode 1 : Fichier de configuration (Recommandée)

Éditez le fichier `config.json` :

```json
{
  "grid_size": 5,
  "start_pos": [0, 0],
  "goal_pos": [4, 4],
  "obstacles": [
    [2, 2],
    [3, 2]
  ],
  "step_cost": -0.01,
  "animation_speed": 3,
  "save_figures": true,
  "output_folder": "results"
}
```

**Paramètres :**
- `grid_size` : Taille de la grille (ex: 5 pour 5x5)
- `start_pos` : Position de départ [ligne, colonne]
- `goal_pos` : Position du but [ligne, colonne]
- `obstacles` : Liste des obstacles [[ligne1, col1], [ligne2, col2], ...]
- `step_cost` : Pénalité par déplacement (ex: -0.01)
- `animation_speed` : Vitesse (1=très lent, 2=lent, 3=normal, 4=rapide)
- `save_figures` : Sauvegarder les figures (true/false)
- `output_folder` : Dossier de sauvegarde des résultats

### Méthode 2 : Mode interactif

Le programme vous demandera tous les paramètres au démarrage.

## ▶️ Exécution

```bash
python main.py
```

Au lancement, choisissez :
- **Option 1** : Charger la configuration depuis `config.json`
- **Option 2** : Mode interactif (saisie manuelle)

## 📊 Résultats

Les figures et valeurs sont sauvegardées dans le dossier `results/` :
- **PNG** : Visualisation de la grille avec heatmap des valeurs
- **TXT** : Matrice des valeurs d'états

Nom des fichiers : `value_table_5x5_YYYYMMDD_HHMMSS.png`

## 🎮 Fonctionnalités

1. **Agent Aléatoire** : Déplacement aléatoire (10 étapes)
2. **Value Iteration** : Apprentissage de la politique optimale
3. **Visualisation** : Heatmap des valeurs d'états
4. **Agent Entraîné** : Démonstration du chemin optimal

## 📝 Exemple de Configuration

### Grille simple (5x5)
```json
{
  "grid_size": 5,
  "start_pos": [0, 0],
  "goal_pos": [4, 4],
  "obstacles": [[2, 2], [3, 2]],
  "animation_speed": 3,
  "save_figures": true
}
```

### Grille complexe (10x10)
```json
{
  "grid_size": 10,
  "start_pos": [0, 0],
  "goal_pos": [9, 9],
  "obstacles": [
    [3, 3], [3, 4], [3, 5],
    [6, 2], [6, 3], [6, 4],
    [7, 7], [8, 7]
  ],
  "animation_speed": 4,
  "save_figures": true
}
```

## 🔧 Dépendances

- `numpy` : Calculs matriciels
- `matplotlib` : Visualisation

## 📚 Algorithme

**Value Iteration** utilise l'équation de Bellman :

```
V(s) = max_a [R(s,a) + γ * V(s')]
```

- **V(s)** : Valeur de l'état s
- **R(s,a)** : Récompense immédiate
- **γ** : Facteur d'actualisation (gamma = 0.9)
- **s'** : État suivant

## 🎯 Objectif

L'agent apprend à atteindre le but (case dorée 'G') en évitant les obstacles (cases grises 'X') tout en minimisant le nombre de déplacements.
