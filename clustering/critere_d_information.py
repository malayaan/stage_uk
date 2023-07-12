import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv(r'data\hate_speech\LSTM_random_embedding\mean_no_emiji.csv', encoding='utf-8')

# Ne pas supprimer la colonne de classe, mais la conserver pour plus tard
X = df.drop('class', axis=1).values
original_classes = df['class'].values

# Liste pour stocker les résultats
n_clusters = np.arange(1, 10)
models = [None for i in range(len(n_clusters))]
aics = [None for i in range(len(n_clusters))]
bics = [None for i in range(len(n_clusters))]

# Calcule les AIC et BIC pour différents nombres de clusters
for index, clusters in enumerate(n_clusters):
    print(index)
    gmm = GaussianMixture(n_components=clusters, n_init=10).fit(X)
    models[index] = gmm
    aics[index] = gmm.aic(X)
    bics[index] = gmm.bic(X)

# Tracer l'AIC et le BIC
plt.figure(figsize=(8, 6))
plt.plot(n_clusters, aics, label='AIC')
plt.plot(n_clusters, bics, label='BIC')
plt.xlabel('Nombre de clusters')
plt.ylabel('Valeur')
plt.legend(loc='best')
plt.title('AIC et BIC pour différents nombres de clusters')
plt.show()
