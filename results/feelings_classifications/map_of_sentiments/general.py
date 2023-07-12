import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Charger les données
df = pd.read_csv(r'C:\Users\decroux paul\Documents\code_stage_uk\data\feelings\relation_feelings_hate speech\anger_x_sarcasm.csv')

# Créer une nouvelle colonne qui est 1 - l'indice de sarcasme
df['inverse_sarcasm_index'] = 1 - df['sarcasm_irony_index']

# Créer un dictionnaire de couleurs et de labels pour chaque classe
label_dict = {0: 'Hate Speech', 1: 'Offensive Speech', 2: 'Neither'}
color_dict = {0: 'red', 1: 'yellow', 2: 'green'}  # Changement des couleurs ici

# Initialiser une grille de sous-plots avec une rangée de trois graphiques
fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # Changement de configuration de subplot

# Boucler sur chaque sous-plot
for i, (label, color) in enumerate([(1, 'yellow'), (2, 'green'), (0, 'red')]):  # Changement des couleurs ici
    # Créer une carte de densité pour chaque classe
    if label == 0:  # use inverse_sarcasm_index for Hate Speech
        data = df[df['class'] == label]
        sns.kdeplot(data=data, x='inverse_sarcasm_index', y='anger_index', fill=True, color=color, ax=axs[i])
        axs[i].set_xlabel('sarcasm_irony_index')
    else:
        data = df[df['class'] == label]
        sns.kdeplot(data=data, x='sarcasm_irony_index', y='anger_index', fill=True, color=color, ax=axs[i])

    # Calculer le centre du cluster et le dessiner sur le graphique
    center_x = np.mean(data['sarcasm_irony_index'])
    center_y = np.mean(data['anger_index'])
    axs[i].plot(center_x, center_y, 'ko')  # dessine le centre du cluster en noir
    axs[i].annotate(f"({center_x:.2f}, {center_y:.2f})", (center_x, center_y))  # ajoute les coordonnées du centre

    axs[i].set_title(f'irony vs Anger for {label_dict[label]}')
    axs[i].set_xlim(0, 1)  # Set x-axis limits
    axs[i].set_ylim(0, 1)  # Set y-axis limits

# Afficher le graphique
plt.tight_layout()
plt.show()
