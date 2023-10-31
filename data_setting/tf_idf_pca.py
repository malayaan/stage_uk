import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Charger les données
df = pd.read_csv("data/select_data/tf_idf_full.csv", encoding='utf-8')
y = df['class'].astype(int)
X = np.loadtxt("data/select_data/tf_idf_full.csv", delimiter=',', skiprows=1)
print(X.shape)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#list of components to test
num_components_list = [10, 50, 100, 500, 1000, 5000, 10000]

#progress print
progress_bar = tqdm(num_components_list, desc="Progress", leave=True)

#cycle over the number of components
for num_components in progress_bar:
    # Mettre à jour le message de la barre de progression
    progress_bar.set_description(f"Num. Components: {num_components}")
    
    # Appliquer PCA pour obtenir les composantes principales
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Concatenate y and X_ into a single DataFrame
    df_output = pd.concat([y, pd.DataFrame(X_pca, columns=[f'feature_{i}' for i in range(num_components)])], axis=1)
    
    # Save to CSV
    df_output.to_csv(f"data/pca/tf_idf_PCA_{num_components}.csv", index=False)
