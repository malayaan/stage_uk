import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("data\LSTM_random_embedding\mean.csv", encoding='utf-8')

# Ne pas supprimer la colonne de classe, mais la conserver pour plus tard
original_classes = df['class'].values
X = df.drop('class', axis=1).values

# Estimer le nombre de min_samples en utilisant la formule 2*dim
dim = X.shape[1]
min_samples = 2000

# Utiliser NearestNeighbors pour trouver la distance au k-ième voisin le plus proche
k = min_samples
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
distances, _ = nn.kneighbors(X)

# Trier les distances et les tracer
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

# Visualiser le graphique et estimer une bonne valeur pour eps
plt.show()

# Une fois que vous avez déterminé une valeur appropriée pour eps, vous pouvez exécuter DBSCAN
eps = 4100  # Mettez votre estimation ici
db = DBSCAN(eps=eps, min_samples=min_samples)
db.fit(X)

labels = db.labels_

# Créer un DataFrame pour stocker les résultats
df_results = pd.DataFrame()
df_results['original_class'] = original_classes
df_results['cluster_label'] = labels

# Regrouper par classe originale et label de cluster et compter le nombre de membres
cluster_distribution = df_results.groupby(['original_class', 'cluster_label']).size().unstack(fill_value=0)

# Pour chaque classe originale, tracer un graphique à secteurs de la répartition des labels de cluster
for original_class in cluster_distribution.index:
    plt.figure(figsize=(6, 6))
    plt.pie(cluster_distribution.loc[original_class], labels=cluster_distribution.columns, autopct='%1.1f%%')
    plt.title(f'Cluster distribution for original class {original_class}')
    plt.show()
