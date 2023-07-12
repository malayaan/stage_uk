import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
print("Loading data...")
df = pd.read_csv(r'data\hate_speech\LSTM_random_embedding\mean_no_emiji.csv', encoding='utf-8')

# Conserver la colonne de classe pour plus tard
X = df.drop('class', axis=1).values
original_classes = df['class'].values

# Déterminer le nombre de clusters
k = 3  # Définir le nombre de clusters que vous voulez ici

# Exécuter KMeans
print("Running KMeans...")
kmeans = KMeans(n_clusters=k, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_

# Renommer les labels de cluster
print("Renaming clusters...")
label_names = {0: 'A', 1: 'B', 2: 'C'}
labels = np.vectorize(label_names.get)(labels)

# Obtenir les labels uniques et leur nombre d'occurrences
unique_labels, counts = np.unique(labels, return_counts=True)

# Afficher chaque classe et le nombre de représentants
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} members")

# Créer un DataFrame pour stocker les résultats
df_results = pd.DataFrame()
df_results['original_class'] = original_classes
df_results['cluster_label'] = labels

# Ajouter les labels de cluster au DataFrame original sous le nom de colonne 'cluster'
df['cluster'] = labels

# Exporter le DataFrame dans un nouveau fichier CSV
df.to_csv(r'data\clustered_data\3_means_clusters.csv', index=False)

# Regrouper par classe originale et label de cluster et compter le nombre de membres
print("Grouping data...")
cluster_distribution = df_results.groupby(['original_class', 'cluster_label']).size().unstack(fill_value=0)

# Créer un figure avec plusieurs subplots
fig, axs = plt.subplots(1, len(cluster_distribution.index), figsize=(18, 6))

# Pour chaque classe originale, tracer un graphique à secteurs de la répartition des labels de cluster
for i, original_class in enumerate(cluster_distribution.index):
    wedges, texts, autotexts = axs[i].pie(cluster_distribution.loc[original_class], labels=cluster_distribution.columns, autopct='%1.1f%%')
    plt.setp(autotexts, size=10, weight="bold")
    for j, autotext in enumerate(autotexts):
        autotext.set_text(f"{autotext.get_text()} ({cluster_distribution.loc[original_class][j]})")
    axs[i].set_title(f'Cluster distribution for original class {original_class}')

# Afficher tous les graphiques à secteurs
plt.show()
