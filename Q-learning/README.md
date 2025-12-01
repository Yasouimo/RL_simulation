# Q-Learning avec Goal Dynamique

Implémentation de Q-Learning pour un environnement GridWorld avec goal dynamique.

## 🎯 Caractéristiques

### Environnement Dynamique
- **Goal mobile** : Change de position aléatoirement à chaque épisode
- **Agent (boule rouge)** : Se déplace sur la grille
- **Features** : Position (row, col) + Distance Manhattan au goal

### Deux Méthodes d'Apprentissage

## 📁 Structure

```
Q-learning/
├── episodic/                    # Méthode épisodique
│   ├── grid_env_dynamic.py     # Environnement
│   ├── q_agent_episodic.py     # Agent Q-Learning épisodique
│   ├── train_episodic.py       # Script d'entraînement
│   └── results_episodic/       # Résultats (créé automatiquement)
│
├── iterative/                   # Méthode itérative
│   ├── grid_env_dynamic.py     # Environnement
│   ├── q_agent_iterative.py    # Agent Q-Learning itératif
│   ├── train_iterative.py      # Script d'entraînement
│   └── results_iterative/      # Résultats (créé automatiquement)
│
└── README.md
```

## 🔄 Méthode 1 : Épisodique

**Principe** : Collecte toutes les transitions d'un épisode, puis met à jour la Q-table à la fin.

### Avantages
- Stabilité de l'apprentissage
- Peut utiliser des techniques de replay
- Bon pour des environnements déterministes

### Algorithme
```python
for episode in episodes:
    transitions = []
    while not done:
        action = choose_action(state)
        next_state, reward = env.step(action)
        transitions.append((state, action, reward, next_state))
    
    # Mise à jour après l'épisode
    for transition in transitions:
        update_q_table(transition)
```

### Exécution
```bash
cd episodic
python train_episodic.py
```

## ⚡ Méthode 2 : Itérative

**Principe** : Met à jour la Q-table immédiatement après chaque transition (step).

### Avantages
- Apprentissage plus rapide
- Réagit immédiatement aux nouvelles informations
- Standard pour Q-Learning

### Algorithme
```python
for episode in episodes:
    while not done:
        action = choose_action(state)
        next_state, reward = env.step(action)
        
        # Mise à jour immédiate
        update_q_table(state, action, reward, next_state)
```

### Exécution
```bash
cd iterative
python train_iterative.py
```

## 📊 Q-Learning Update Rule

Les deux méthodes utilisent la même règle de mise à jour :

```
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
```

Où :
- **α** (learning_rate) = 0.1 : Taux d'apprentissage
- **γ** (gamma) = 0.99 : Facteur d'actualisation
- **r** : Récompense
- **s'** : État suivant
- **max_a' Q(s', a')** : Meilleure valeur Q de l'état suivant

## 🎮 Features de l'État

Chaque état est représenté par :
1. **Position de l'agent** : (row, col)
2. **Distance au goal** : Distance de Manhattan

Exemple : `(2, 3, 4)` = Agent en (2,3), distance 4 du goal

## 🏆 Récompenses

- **Goal atteint** : +10.0
- **Déplacement normal** : -0.01
- **Obstacle** : -1.0
- **Mur** : -0.01

## 📈 Résultats

Les scripts sauvegardent automatiquement :
- **Courbes de progression** (PNG)
- **Statistiques détaillées** (JSON)
  - Récompenses par épisode
  - Longueur des épisodes
  - Taux de succès
  - Taille de la Q-table

## ⚙️ Paramètres

### Environnement
```python
grid_size = 5                    # Grille 5x5
obstacles = [(2, 2)]            # Un obstacle
max_steps_per_episode = 100     # Limite de pas
```

### Agent
```python
learning_rate = 0.1             # Alpha
gamma = 0.99                    # Facteur d'actualisation
epsilon = 1.0                   # Exploration initiale
epsilon_decay = 0.995           # Décroissance
epsilon_min = 0.01              # Exploration minimale
```

### Entraînement
```python
num_episodes = 500              # Nombre d'épisodes
render_frequency = 50           # Affichage tous les 50 épisodes
```

## 🔍 Comparaison des Méthodes

| Aspect | Épisodique | Itérative |
|--------|-----------|-----------|
| Mise à jour | Fin d'épisode | Chaque step |
| Vitesse | Plus lent | Plus rapide |
| Stabilité | Plus stable | Peut osciller |
| Mémoire | Buffer requis | Pas de buffer |
| Standard | Monte Carlo | Q-Learning classique |

## 🚀 Pour commencer

1. **Méthode épisodique** :
```bash
cd episodic
python train_episodic.py
```

2. **Méthode itérative** :
```bash
cd iterative
python train_iterative.py
```

Les deux afficheront :
- La grille en temps réel
- Les courbes de progression
- Les statistiques d'entraînement

## 📝 Notes

- Le goal change à chaque nouvel épisode
- L'agent apprend à se diriger vers le goal quelle que soit sa position
- Les features incluent la distance, permettant une généralisation
- Epsilon-greedy pour équilibrer exploration/exploitation
