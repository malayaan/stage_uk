import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# Charger les données
df = pd.read_csv("data/select_data/tf_idf_0.1_emoji.csv", encoding='utf-8')
y = df['class'].astype(int)
X = df.drop('class', axis=1).values.tolist()

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Valeurs de C à tester
C_values = [0.01, 0.1, 1.0, 10, 100]

# Listes pour stocker les scores F1
train_scores = []
test_scores = []

# Indicateur de progression
pbar = tqdm(total=len(C_values), desc='Progression')

# Parcourir les différentes valeurs de C
for C in C_values:
    # Créer le modèle SVM avec la valeur de C
    model = LinearSVC(class_weight='balanced',C=C, penalty='l2', loss='squared_hinge',multi_class='ovr', max_iter=10000)
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Prédire les étiquettes sur les ensembles d'entraînement et de test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculer le score F1 sur les ensembles d'entraînement et de test
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    # Ajouter les scores à la liste respective
    train_scores.append(train_f1)
    test_scores.append(test_f1)
    
     # Mettre à jour l'indicateur de progression
    pbar.update(1)

# Terminer l'indicateur de progression
pbar.close()

# Tracer le graphique
plt.plot(C_values, train_scores, label='Train F1 Score')
plt.plot(C_values, test_scores, label='Test F1 Score')
plt.xlabel('Valeur de C')
plt.ylabel('Score F1')
plt.title("Évolution du score F1 en fonction de C")
plt.legend()
plt.show()
"""
# Enregistrer le graphique
plt.savefig('resultats/svm regarding to pca/png_svm_5000.png')

# Enregistrer les données dans un fichier texte
resultats_df = pd.DataFrame({'C': C_values, 'Train F1 Score': train_scores, 'Test F1 Score': test_scores})
resultats_df.to_csv('resultats/svm regarding to pca/txt_svm_5000.txt', index=False, sep='\t')
"""