from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("Loading the data...")

# Charger les données
df_test = pd.read_csv(r"data\feelings\tf_idf\fear\0.005_fear_test_select.csv", encoding='utf-8')
y_test = df_test['fear_index'].astype(float)
X_test = df_test.drop('fear_index', axis=1).values

# Charger les données
df_train = pd.read_csv(r"data\feelings\tf_idf\fear\0.005_fear_train_select.csv", encoding='utf-8')
y_train = df_train['fear_index'].astype(float)
X_train = df_train.drop('fear_index', axis=1).values

print("Data loaded successfully. Building the model...")

# Redéfinir la forme des données pour qu'elles soient compatibles avec un CNN
n_samples, n_features = X_train.shape
n_channels = 1  # Nous supposons que nous avons des "images" 1D
X_train = X_train.reshape(n_samples, n_features, n_channels)
X_test = X_test.reshape(X_test.shape[0], n_features, n_channels)

def create_model(learning_rate=0.01):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(n_features, n_channels)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=10, verbose=0)

# Définir le paramètre de la grille
param_grid = {'learning_rate': [0.00001,0.0001,0.0005,0.001, 0.01]}

# Ajuster le modèle aux données avec recherche de grille et validation croisée
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, scoring='neg_mean_squared_error', return_train_score=True)
grid_result = grid.fit(X_train, y_train)

print("Grid Search complete. Making predictions...")

# Prédire les scores des données de test
y_pred = grid.predict(X_test)

# Afficher les statistiques de régression
print("Best Parameters: ", grid.best_params_)
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

# Enregistrer les résultats dans un fichier .txt
with open('results/feelings_classifications/tf_idf/0.005_fear_cnn_regression.txt', 'w') as f:
    print("Best Parameters: ", grid.best_params_, file=f)
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred), file=f)

print("Creating and saving plots...")

# Plotting cross-validated scores as a function of learning_rate
plt.figure(figsize=(10, 6))
learning_rate_values = param_grid['learning_rate']
scores_test = -grid.cv_results_['mean_test_score']
scores_train = -grid.cv_results_['mean_train_score']

std_test_scores = grid.cv_results_['std_test_score']  # get the standard deviations
std_train_scores = grid.cv_results_['std_train_score']

# we negate the scores because we used 'neg_mean_squared_error' as the scoring metric
plt.errorbar(learning_rate_values, scores_test, yerr=std_test_scores, capsize=5, fmt='-o', label='Test')  # add error bars to the plot
plt.errorbar(learning_rate_values, scores_train, yerr=std_train_scores, capsize=5, fmt='-o', label='Train')

plt.xlabel('Learning Rate')
plt.ylabel('Mean Squared Error')
plt.title('Performance as a function of Learning Rate')
plt.legend()
plt.grid()

# Save the plot
plt.savefig('results/feelings_classifications/tf_idf/0.005_fear_cnn_learning_rate_performance.png')

print("Plots saved successfully.")

plt.show()

print("All tasks successfully completed!")
