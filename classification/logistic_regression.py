import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv(r"C:\Users\decroux paul\Documents\code_stage_uk\data\hate_speech\LSTM_random_embedding\mean_no_emiji.csv", encoding='utf-8')
y = df['class'].astype(int)
X = df.drop('class', axis=1).values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de régression logistique
logreg = LogisticRegression(solver='saga', class_weight='balanced', max_iter=10000, penalty='l2')

# Créer un objet de recherche par grille
param_grid = {'C': [ 0.01, 0.1, 1]}
grid_search = GridSearchCV(logreg, param_grid, cv=2, verbose=3)

# Ajuster le modèle aux données
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print("Best parameters: ", grid_search.best_params_)

# Prédire les labels des données de test
y_pred = grid_search.predict(X_test)

# Afficher les statistiques de classification
print(classification_report(y_test, y_pred))

# Enregistrer les résultats dans un fichier .txt
with open('results/classification/logistic_regression_tf_idf/txt_tf_idf_0.01.txt', 'w') as f:
    print("Best parameters: ", grid_search.best_params_, file=f)
    print("\nClassification report:\n", classification_report(y_test, y_pred), file=f)

    # Obtenir les résultats de la recherche par grille sous forme de DataFrame pour faciliter la manipulation
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Extraire les valeurs moyennes et les écarts-types des scores F1 pour chaque combinaison d'hyperparamètres
    mean_scores = results_df['mean_test_score']
    std_scores = results_df['std_test_score']
    params = results_df['params']

    for param, mean_score, std_score in zip(params, mean_scores, std_scores):
        f.write("\nParameters: {}, Mean Score: {}, Standard Deviation: {}\n".format(param, mean_score, std_score))

# Obtenir les résultats de la recherche par grille sous forme de DataFrame pour faciliter la manipulation
results_df = pd.DataFrame(grid_search.cv_results_)

# Extraire les valeurs moyennes et les écarts-types des scores F1 pour chaque combinaison d'hyperparamètres
mean_scores = results_df['mean_test_score']
std_scores = results_df['std_test_score']
params = [ 0.01, 0.1, 1]

# Tracer les scores F1 avec des bandes pour l'écart-type
plt.figure(figsize=(10, 6))
plt.plot(params, mean_scores, label='Mean F1 Score')
plt.fill_between(params, mean_scores - std_scores, mean_scores + std_scores, color='gray', alpha=0.2, label='Standard Deviation')
plt.title('F1 Score vs. C')
plt.xlabel('C')
plt.ylabel('F1 Score')
plt.xscale('log')
plt.legend(loc='best')
plt.grid()

# Enregistrer le graphique
plt.savefig('results/classification/logistic_regression_tf_idf/png_tf_idf_0.01.png')
plt.show()
