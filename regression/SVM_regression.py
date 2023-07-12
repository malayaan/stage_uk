from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd

print("Loading the data...")

# Charger les données
df_test = pd.read_csv(r'data\feelings\LSTM_random_embedding\anger_mean_test.csv', encoding='utf-8')
y_test = df_test['anger_index'].astype(float)
X_test = df_test.drop('anger_index', axis=1).values

# Charger les données
df_train = pd.read_csv(r'data\feelings\LSTM_random_embedding\anger_mean_train.csv', encoding='utf-8')
y_train = df_train['anger_index'].astype(float)
X_train = df_train.drop('anger_index', axis=1).values

print("Data loaded successfully. Building the model...")

# Créer un modèle de régression avec SVM
svm = SVR()

# Définir le paramètre de la grille
#colere
param_grid = {'C': [0.05, 0.1, 1, 10,15, 20,30]}
#peur
#param_grid = {'C': [0.05, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

print("Performing Grid Search...")

# Ajuster le modèle aux données avec recherche de grille et validation croisée
grid = GridSearchCV(svm, param_grid, cv=10, scoring='neg_mean_squared_error', return_train_score=True)
grid.fit(X_train, y_train)

print("Grid Search complete. Making predictions...")

# Prédire les scores des données de test
y_pred = grid.predict(X_test)

# Afficher les statistiques de régression
print("Best Parameters: ", grid.best_params_)
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

# Enregistrer les résultats dans un fichier .txt
with open(r'results\feelings_classifications\LSTM_mean\anger_svm_C_performance.txt', 'w') as f:
   print("Best Parameters: ", grid.best_params_, file=f)
   print("Mean Squared Error: ", mean_squared_error(y_test, y_pred), file=f)

print("Creating and saving plots...")

# Plotting cross-validated scores as a function of C
plt.figure(figsize=(10, 6))
C_values = param_grid['C']
test_scores = -grid.cv_results_['mean_test_score']
train_scores = -grid.cv_results_['mean_train_score']  # get the mean train scores
std_test_scores = grid.cv_results_['std_test_score']  # get the standard deviations
std_train_scores = grid.cv_results_['std_train_score']  # get the standard deviations for the train scores

# we negate the scores because we used 'neg_mean_squared_error' as the scoring metric
plt.errorbar(C_values, test_scores, yerr=std_test_scores, capsize=5, fmt='-o', label='Test')  # add error bars to the plot
plt.errorbar(C_values, train_scores, yerr=std_train_scores, capsize=5, fmt='-o', label='Train')  # add error bars for the train scores to the plot

plt.xlabel('C')
plt.ylabel('Mean Squared Error')
plt.title('Performance as a function of C')
plt.legend()  # add a legend to the plot
plt.grid()

# Save the plot
plt.savefig(r'results\feelings_classifications\LSTM_mean\anger_svm_C_performance.png')

print("Plots saved successfully.")

plt.show()

print("All tasks successfully completed!")
