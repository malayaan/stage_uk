import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv(r'data\hate_speech\pca\tf_idf_PCA_100.csv', encoding='utf-8')

# Ne pas supprimer la colonne de classe, mais la conserver pour plus tard
X = df.drop('class', axis=1).values
original_classes = df['class'].values

# Déterminer le nombre de clusters
range_k = np.arange(1,100,10)  # Définir la plage de valeurs pour k que vous voulez tester ici
inertias = []  # Liste pour stocker les inerties pour chaque k

# Exécuter KMeans pour chaque k et stocker l'inertie
for k in range_k:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Tracer l'inertie en fonction de k
plt.figure(figsize=(8, 6))
plt.plot(range_k, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal number of clusters')
plt.grid(True)
plt.show()
