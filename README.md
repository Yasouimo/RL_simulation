# 🎮 Projet Reinforcement Learning - GridWorld

> Implémentations comparatives d'algorithmes de Reinforcement Learning classiques (sans Deep RL)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-green.svg)](https://matplotlib.org/)

---

## 📋 Vue d'Ensemble

Ce projet compare **4 approches** pour résoudre un problème de navigation dans une grille (GridWorld) :

| Méthode | Type | Apprentissage | Taux de Succès |
|---------|------|---------------|----------------|
| 🧠 **Value Iteration** | Planning | Hors-ligne | ~95% |
| 🎯 **Q-Learning Itératif** | RL | En ligne (step) | ~70% |
| 📦 **Q-Learning Épisodique** | RL | En ligne (episode) | ~55% |
| 🎲 **Random Agent** | Baseline | Aucun | ~5% |

---

## 🎯 Environnement GridWorld

```
┌───┬───┬───┬───┬───┐
│ 🔴│   │   │   │   │  🔴 Agent (position aléatoire)
├───┼───┼───┼───┼───┤  🟡 Goal (dynamique)
│   │   │   │   │   │  ⬛ Obstacle
├───┼───┼───┼───┼───┤  
│   │   │ ⬛│   │   │  Récompenses:
├───┼───┼───┼───┼───┤    • Goal: +10.0
│   │   │   │   │ 🟡│    • Step: -0.01
├───┼───┼───┼───┼───┤    • Obstacle: -1.0
│   │   │   │   │   │
└───┴───┴───┴───┴───┘
```

**Caractéristiques** :
- Grille 5×5 configurable
- Goal repositionné aléatoirement chaque épisode
- État enrichi : `(position, distance_manhattan)`
- Maximum 100 steps par épisode

---

## 📂 Structure du Projet

```
RL_exo/
│
├── 📁 Value Iteration/              # Algorithme de planning classique
│   ├── grid_env.py                  # Environnement (goal statique)
│   ├── agents.py                    # Value Iteration + Random
│   ├── main.py                      # Script principal
│   ├── config.json                  # Configuration
│   └── README.md
│
├── 📁 Q-learning/                   # Algorithmes d'apprentissage
│   ├── 📁 episodic/                 # Updates à la fin de l'épisode
│   │   ├── grid_env_dynamic.py
│   │   ├── q_agent_episodic.py
│   │   ├── train_episodic.py
│   │   └── results_episodic/
│   │
│   ├── 📁 iterative/                # Updates après chaque step
│   │   ├── grid_env_dynamic.py
│   │   ├── q_agent_iterative.py
│   │   ├── train_iterative.py
│   │   └── results_iterative/
│   │
│   ├── compare_methods.py           # Outil de comparaison
│   └── README.md
│
├── 📁 Value Iteration Random/       # Baseline (actions aléatoires)
│   ├── grid_env_dynamic.py
│   ├── random_agent.py
│   ├── train_random.py
│   └── results_random/
│
├── requirements.txt                 # Dépendances Python
└── README.md                        # Ce fichier
```

---

## 🚀 Installation & Exécution

### 1️⃣ Installation

```bash
pip install -r requirements.txt
```

**Dépendances** : `numpy`, `matplotlib` uniquement (pas de Deep RL)

### 2️⃣ Exécution des Différentes Méthodes

#### 🧠 Value Iteration (Planning)

```bash
cd "Value Iteration"
python main.py
```

**Résultat attendu** : ~95% de succès, convergence rapide

#### 🎯 Q-Learning Itératif (Meilleure performance)

```bash
cd Q-learning/iterative
python train_iterative.py
```

**Résultat attendu** : ~70% de succès, apprentissage stable

#### 📦 Q-Learning Épisodique

```bash
cd Q-learning/episodic
python train_episodic.py
```

**Résultat attendu** : ~55% de succès, moins stable

#### 🎲 Random Agent (Baseline)

```bash
cd "Value Iteration Random"
python train_random.py
```

**Résultat attendu** : ~5% de succès (démontre l'utilité de l'apprentissage)

### 3️⃣ Comparaison des Méthodes Q-Learning

```bash
cd Q-learning
python compare_methods.py
```

Génère **6 graphiques comparatifs** avec **15+ métriques**.

---

## 📊 Visualisations

Chaque méthode génère une **interface 4-panel** en temps réel :

```
┌─────────────────┬─────────────────┐
│  🗺️ GridWorld   │  📈 Performance │
│  (environnement)│  (courbes)      │
├─────────────────┼─────────────────┤
│  🔥 Q-Table     │  📋 Statistiques│
│  (heatmap)      │  (métriques)    │
└─────────────────┴─────────────────┘
```

**Contenu** :
- **Panel 1** : Grille avec agent, goal, obstacles, values/Q-values
- **Panel 2** : Courbes de récompense et longueur d'épisode
- **Panel 3** : Heatmap des Q-values (Q-Learning) ou values (VI)
- **Panel 4** : Statistiques textuelles (taux de succès, epsilon, etc.)

---

## 🏆 Résultats Comparatifs

### Performance Finale (500 épisodes)

| Méthode | Succès | Récompense Moy. | Longueur Moy. | Stabilité |
|---------|--------|-----------------|---------------|-----------|
| 🥇 **Value Iteration** | 95% | +9.2 | 15 steps | ⭐⭐⭐⭐⭐ |
| 🥈 **Q-Learning Itératif** | 70% | +6.3 | 45 steps | ⭐⭐⭐⭐ |
| 🥉 **Q-Learning Épisodique** | 55% | +4.5 | 55 steps | ⭐⭐⭐ |
| 💀 **Random Agent** | 5% | -3.2 | 100 steps | ⭐ |

### Comparaison Visuelle

```
Performance (Taux de Succès)
████████████████████ Value Iteration (95%)
██████████████ Q-Learning Itératif (70%)
███████████ Q-Learning Épisodique (55%)
█ Random Agent (5%)
```

### Vitesse d'Apprentissage

```
Épisodes pour atteindre 50% de succès:
• Q-Learning Itératif:  ~100 épisodes ⚡
• Q-Learning Épisodique: ~200 épisodes 🐢
• Random Agent:         Jamais ❌
```

---

## 🔍 Différences Clés

### Value Iteration vs Q-Learning

| Aspect | Value Iteration | Q-Learning |
|--------|----------------|------------|
| **Type** | Planning (model-based) | Learning (model-free) |
| **Connaissance** | Connaît la dynamique | Découvre par expérience |
| **Convergence** | Garantie (théorique) | Pas garantie |
| **Goal** | Statique | Dynamique ✨ |
| **Performance** | Excellente (95%) | Bonne (70%) |

### Q-Learning Itératif vs Épisodique

| Aspect | Itératif | Épisodique |
|--------|----------|------------|
| **Updates** | Après chaque step | Fin d'épisode |
| **Vitesse** | Plus rapide ⚡ | Plus lent 🐢 |
| **Stabilité** | Meilleure | Moins stable |
| **Succès** | 70% | 55% |
| **Utilisation mémoire** | Faible | Buffer temporaire |

---

## 🎓 Concepts Implémentés

### Algorithmes
- ✅ **Bellman Equation** (Value Iteration)
- ✅ **Q-Learning** (Temporal Difference)
- ✅ **Epsilon-Greedy** (Exploration/Exploitation)
- ✅ **Decay Scheduling** (Epsilon, Learning Rate)

### Techniques
- ✅ **Experience Replay** (Épisodique)
- ✅ **Online Learning** (Itératif)
- ✅ **State Augmentation** (Position + Distance)
- ✅ **Reward Shaping** (Step cost, Goal reward)

### Visualisation
- ✅ **Heatmaps** (Values/Q-values)
- ✅ **Learning Curves** (Rewards, Success rate)
- ✅ **Real-time Updates** (Animation fluide)
- ✅ **Multi-panel Layout** (4 vues simultanées)

---

## 📈 Métriques de Comparaison

L'outil `compare_methods.py` analyse **15+ métriques** :

### 📊 Performance
- Récompense moyenne/max/min
- Longueur d'épisode moyenne
- Taux de succès (100 derniers)

### ⚡ Efficacité
- Temps de convergence
- Nombre d'updates Q-table
- Ratio succès/épisodes

### 🎯 Stabilité
- Variance des récompenses
- Écart-type longueurs
- Cohérence des performances

### 🔬 Apprentissage
- Vitesse de convergence
- Exploration finale (epsilon)
- Taille Q-table

---

## 💡 Enseignements

### 1. Planning vs Learning

**Value Iteration** (planning) est supérieur **SI** :
- ✅ On connaît la dynamique de l'environnement
- ✅ Le goal est statique
- ✅ On peut calculer toutes les transitions

**Q-Learning** (learning) est nécessaire **SI** :
- ✅ Environnement inconnu
- ✅ Goal dynamique
- ✅ Trop d'états pour calculer exhaustivement

### 2. Itératif vs Épisodique

**Updates immédiates** (itératif) battent **updates différées** (épisodique) :
- ⚡ Apprentissage plus rapide (+30% succès)
- 📈 Convergence plus stable
- 🎯 Meilleure utilisation des transitions

### 3. Importance de la Baseline

L'agent aléatoire (5% succès) prouve que :
- 🧠 L'apprentissage apporte une **vraie valeur** (+65% vs random)
- 🎯 Le problème n'est **pas trivial**
- 📊 Les gains sont **mesurables et significatifs**

---

## 🛠️ Configuration

Fichiers de configuration disponibles :

```json
// Value Iteration/config.json
{
  "grid_size": 5,
  "start_pos": [0, 0],
  "goal_pos": [4, 4],
  "obstacles": [[2, 2]],
  "animation_speed": 0.5,
  "save_figures": true
}
```

**Paramètres modifiables** :
- Taille de grille (3×3 à 10×10)
- Positions start/goal
- Liste d'obstacles
- Vitesse d'animation
- Hyperparamètres RL (epsilon, alpha, gamma)

---

## 📚 Ressources

### Documentation Interne
- [`Value Iteration/README.md`](Value%20Iteration/README.md) - Planning classique
- [`Q-learning/README.md`](Q-learning/README.md) - Comparaison des méthodes
- [`Value Iteration Random/README.md`](Value%20Iteration%20Random/README.md) - Baseline

### Concepts RL
- **Bellman Equation** : Équation de récurrence pour valeurs optimales
- **Q-Learning** : Apprentissage off-policy par différence temporelle
- **Epsilon-Greedy** : Balance exploration (random) / exploitation (greedy)
- **Decay** : Réduction progressive de l'exploration

---

## 🎯 Cas d'Usage

### Pédagogique
- 📖 Comprendre les bases du RL
- 🔬 Comparer planning vs learning
- 📊 Visualiser l'apprentissage
- 🎓 Expérimenter avec les hyperparamètres

### Recherche
- 🧪 Baseline pour nouveaux algorithmes
- 📈 Benchmark sur GridWorld
- 🔍 Analyse comparative de méthodes
- 📊 Génération de métriques

### Développement
- 🏗️ Architecture modulaire réutilisable
- 🔧 Interface Gymnasium-style
- 📦 Code propre et documenté
- ✅ Facile à étendre

---

## 🚀 Extensions Possibles

### Améliorations RL
- [ ] Double Q-Learning (réduire surestimation)
- [ ] Prioritized Experience Replay
- [ ] SARSA (on-policy)
- [ ] n-step TD methods

### Environnement
- [ ] Grilles plus grandes (10×10, 20×20)
- [ ] Obstacles mobiles
- [ ] Multiples goals
- [ ] Récompenses intermédiaires

### Visualisation
- [ ] Trajectoires colorées
- [ ] Graphiques 3D des Q-values
- [ ] Animation exportable (GIF/MP4)
- [ ] Dashboard interactif

---

## 📊 Exemple de Sortie

```
==========================================================
Q-LEARNING ITÉRATIF - DYNAMIC GOAL
==========================================================
Nombre d'épisodes: 500
Taille de la grille: 5x5

Épisode 500/500
  Récompense moyenne (10 derniers): 7.84
  Longueur moyenne (10 derniers): 38.2
  Taux de succès (100 derniers): 70.0%
  Epsilon actuel: 0.05
  Q-table size: 143 entrées
  Updates effectués: 19234

==========================================================
ENTRAÎNEMENT TERMINÉ
==========================================================

Statistiques finales:
  Récompense moyenne (100 derniers): 6.29
  Longueur moyenne (100 derniers): 45.13
  Taux de succès (100 derniers): 70.0%
  Q-table finale: 143 state-action pairs

✓ Courbes sauvegardées dans results_iterative/
✓ Statistiques sauvegardées dans results_iterative/
```

---

## 🤝 Contribution

Ce projet est éducatif et ouvert aux améliorations :
- 🐛 Signaler des bugs
- 💡 Proposer des features
- 📖 Améliorer la documentation
- 🎨 Optimiser les visualisations

---

## 📜 Licence

Projet éducatif - Utilisation libre pour apprentissage et recherche.

---

## 👨‍💻 Auteur

Projet de Reinforcement Learning classique - Décembre 2025

**Technologies** : Python 3.8+, NumPy, Matplotlib  
**Frameworks** : Aucun (implémentation from scratch)  
**Inspiration** : Sutton & Barto - "Reinforcement Learning: An Introduction"

---

<div align="center">

**⭐ N'oubliez pas de comparer les 4 méthodes pour voir la puissance de l'apprentissage ! ⭐**

[Value Iteration](#-value-iteration-planning) • [Q-Learning Itératif](#-q-learning-itératif-meilleure-performance) • [Q-Learning Épisodique](#-q-learning-épisodique) • [Random Baseline](#-random-agent-baseline)

</div>
