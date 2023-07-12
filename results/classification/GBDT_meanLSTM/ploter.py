import numpy as np
import re
import matplotlib.pyplot as plt

learning_rates = [0.01, 0.1, 1]
n_estimators = [100, 200, 300]

# Scores F1 moyens correspondants à chaque combinaison de learning_rate et N_estimator
f1_scores_mean = np.array([
    [0.9574, 0.967, 0.97],
    [0.9726, 0.9746, 0.9748],
    [0.9572, 0.957, 0.9552]
])

# Écarts types correspondants à chaque combinaison de learning_rate et N_estimator
std_dev_matrix = np.array([
    [0.005642694391866359, 0.0020591260281974015, 0.0024166091947189165],
    [0.0018547236990991425, 0.002059126028197402, 0.009541488353501259],
    [0.00755248303539969, 0.009086253353280447, 0.0]
])

# Trouver les indices des valeurs maximales
max_indices = np.argwhere(f1_scores_mean == np.max(f1_scores_mean))
max_lr_idx, max_ne_idx = max_indices[0]

# Création des grilles de learning_rate et N_estimator
X, Y = np.meshgrid(n_estimators, learning_rates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Afficher la surface des scores F1 moyens
ax.plot_surface(X, Y, f1_scores_mean, cmap='viridis')

# Afficher la surface supérieure des écarts types
ax.plot_surface(X, Y, f1_scores_mean + std_dev_matrix, alpha=0.3, color='gray')

# Afficher la surface inférieure des écarts types
ax.plot_surface(X, Y, f1_scores_mean - std_dev_matrix, alpha=0.3, color='gray')

# Afficher le point rouge pour la valeur maximale
ax.scatter(n_estimators[max_ne_idx], learning_rates[max_lr_idx], f1_scores_mean[max_lr_idx, max_ne_idx], color='red', s=50)

ax.set_xlabel('N Estimators')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('F1 Score Mean')

plt.title('F1 Score Mean with Standard Deviation')
plt.show()