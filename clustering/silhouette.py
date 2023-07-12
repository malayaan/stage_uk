import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv(r' data\hate_speech\select_data\tf_idf_0.1.csv', encoding='utf-8')

# Ne pas supprimer la colonne de classe, mais la conserver pour plus tard
X = df.drop('class', axis=1).values
original_classes = df['class'].values

# Déterminer le nombre de clusters
range_n_clusters = list(range(2,11)) # Testez par exemple de 2 à 10 clusters
silhouette_scores = []

for n_clusters in range_n_clusters:
    # Exécuter KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_

    # Calculer le score de silhouette pour le nombre actuel de clusters
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

# Tracer le score de silhouette pour différents nombres de clusters
plt.figure(figsize=(7, 4))
plt.plot(range_n_clusters, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette score')
plt.title('Silhouette Method')
plt.show()
