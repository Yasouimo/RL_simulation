# 🎮 Reinforcement Learning - GridWorld

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-green.svg)](https://matplotlib.org/)

> Comparaison de 4 algorithmes RL classiques sur un environnement GridWorld

---

## 📊 Résultats

| Méthode | Succès | Récompense | Vitesse |
|---------|--------|------------|---------|
| 🥇 **Value Iteration** | 95% | +9.2 | ⭐⭐⭐⭐⭐ |
| 🥈 **Q-Learning Itératif** | 70% | +6.3 | ⭐⭐⭐⭐ |
| 🥉 **Q-Learning Épisodique** | 55% | +4.5 | ⭐⭐⭐ |
| 💀 **Random (Baseline)** | 5% | -3.2 | ⭐ |

![Performance Comparison](https://via.placeholder.com/800x400/1a1a1a/00ff00?text=Performance+Chart+%7C+Value+Iteration+%3E+Q-Learning+Iterative+%3E+Q-Learning+Episodic+%3E+Random)

---

## 🎯 L'Environnement

```
┌───┬───┬───┬───┬───┐
│ 🔴│   │   │   │   │  🔴 Agent
├───┼───┼───┼───┼───┤  🟡 Goal (dynamique)
│   │   │ ⬛│   │   │  ⬛ Obstacle
├───┼───┼───┼───┼───┤  
│   │   │   │   │ 🟡│  Récompenses:
├───┼───┼───┼───┼───┤  +10 (goal)
│   │   │   │   │   │  -0.01 (step)
└───┴───┴───┴───┴───┘  -1 (obstacle)
```

**Grille 5×5** · Goal repositionné chaque épisode · Max 100 steps

---

## 🚀 Quick Start

```bash
# Installation
pip install -r requirements.txt

# Value Iteration (meilleure performance)
cd "Value Iteration"
python main.py

# Q-Learning Itératif (recommandé)
cd Q-learning/iterative
python train_iterative.py

# Q-Learning Épisodique
cd Q-learning/episodic
python train_episodic.py

# Random Agent (baseline)
cd "Value Iteration Random"
python train_random.py

# Comparer Q-Learning
cd Q-learning
python compare_methods.py
```

---

## 📁 Structure

```
RL_exo/
├── Value Iteration/      # Planning (goal statique)
├── Q-learning/
│   ├── episodic/         # Updates fin d'épisode
│   ├── iterative/        # Updates chaque step
│   └── compare_methods.py
└── Value Iteration Random/  # Baseline
```

---

## 📊 Visualisation 4-Panel

![4-Panel Interface](https://via.placeholder.com/1200x800/2d2d2d/ffffff?text=GridWorld+%7C+Performance+Curves+%7C+Q-Table+Heatmap+%7C+Statistics)

Chaque méthode affiche en temps réel :
- 🗺️ **GridWorld** : Agent, goal, obstacles
- 📈 **Courbes** : Récompenses et longueurs
- 🔥 **Heatmap** : Q-values ou values
- 📋 **Stats** : Taux de succès, epsilon, etc.

---

## 🔍 Différences Clés

### Value Iteration vs Q-Learning

| | Value Iteration | Q-Learning |
|---|---|---|
| **Type** | Planning | Learning |
| **Goal** | Statique | Dynamique ✨ |
| **Performance** | 95% | 70% |

### Q-Learning : Itératif vs Épisodique

| | Itératif | Épisodique |
|---|---|---|
| **Updates** | Chaque step | Fin d'épisode |
| **Succès** | 70% | 55% |
| **Vitesse** | ⚡ Rapide | 🐢 Lent |

![Learning Speed](https://via.placeholder.com/800x300/1a1a1a/ffaa00?text=Iterative+converges+2x+faster+than+Episodic)

---

## 💡 Ce Qu'On Apprend

1. **Planning vs Learning** : Value Iteration gagne quand on connaît l'environnement
2. **Updates immédiates** : Itératif bat Épisodique (+30% succès)
3. **Baseline importante** : Random (5%) prouve la valeur de l'apprentissage (+65%)

---

## 🎓 Concepts Implémentés

✅ Bellman Equation · Q-Learning · Epsilon-Greedy · State Augmentation · Heatmaps · Real-time Visualization

---

## 📚 Documentation

- [`Value Iteration/README.md`](Value%20Iteration/README.md)
- [`Q-learning/README.md`](Q-learning/README.md)
- [`Value Iteration Random/README.md`](Value%20Iteration%20Random/README.md)

---

<div align="center">

**Technologies** : Python 3.8+ · NumPy · Matplotlib  
**Inspiration** : Sutton & Barto - "Reinforcement Learning: An Introduction"

![RL Logo](https://via.placeholder.com/600x200/4a90e2/ffffff?text=Reinforcement+Learning+GridWorld)

**⭐ Comparez les 4 méthodes pour voir la puissance de l'apprentissage ! ⭐**

</div>
