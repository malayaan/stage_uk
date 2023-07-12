from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("Loading the data...")

# Charger les données
df_test = pd.read_csv(r'C:\Users\decroux paul\Documents\code_stage_uk\anger_mean_test.csv', encoding='utf-8')
y_test = df_test['anger_index'].astype(float)
X_test = df_test.drop('anger_index', axis=1).values

df_train = pd.read_csv(r'C:\Users\decroux paul\Documents\code_stage_uk\anger_mean_train.csv', encoding='utf-8')
y_train = df_train['anger_index'].astype(float)
X_train = df_train.drop('anger_index', axis=1).values

print("Data loaded successfully.")

# Combining the training and test data
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

print("Building and training the model...")

# Créer un modèle de régression avec SVM et C=1
svm = SVR(C=1)

# Ajuster le modèle aux données
svm.fit(X, y)

print("Loading the hate data...")

# Charger les données de discours de haine
df_hate = pd.read_csv(r'C:\Users\decroux paul\Documents\code_stage_uk\anger_hate_2.csv', encoding='utf-8')
X_hate = df_hate.values

print("Predicting labels for hate data...")

# Prédire les étiquettes pour les données de discours de haine
y_hate_pred = svm.predict(X_hate)

# Enregistrer les prédictions dans un DataFrame
df_hate_labels = pd.DataFrame(y_hate_pred, columns=['predicted_label'])

# Enregistrer les prédictions dans un fichier CSV
df_hate_labels.to_csv(r'hate_anger_labels_2.csv', index=False)

print("Predicted labels saved successfully.")

print("All tasks successfully completed!")