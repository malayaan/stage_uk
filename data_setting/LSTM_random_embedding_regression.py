from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from keras.utils import pad_sequences
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import tf_idf_linear_select as tf
from sklearn.model_selection import train_test_split

def prepare_data(df, word_to_index=None):
    # Préparation des données
    tweets = df.Tweet
    X = []
    for t in tweets:
        tokens = tf.hand_tokenize(tf.preprocess(t))
        X += [tokens]

    if word_to_index is None:
        # Création d'un vocabulaire avec tous les mots uniques dans les tweets
        vocab = set(word for tweet in X for word in tweet)
        # Création d'un dictionnaire qui attribue un indice unique à chaque mot
        word_to_index = {word: i for i, word in enumerate(vocab)}
    
    # Conversion des tweets en listes d'indices de mots
    X_indices = [[word_to_index[word] for word in tweet if word in word_to_index] for tweet in X]
    
    # Padding
    X_pad = pad_sequences(X_indices)
    
    return X_pad, word_to_index

def prepare_data_test(df, word_to_index):
    return prepare_data(df, word_to_index=word_to_index)

# Chargement des données d'entraînement et de test
df = pd.read_csv(r"data\general\anger_tweets.csv", encoding='utf-8')
df_hate = pd.read_csv(r"gpt_label\tweet.csv", encoding='utf-8')
tweets=df.Tweet
hate_tweets = df_hate.Tweet
df_train, df_test = train_test_split(df, test_size=0.4, random_state=42)
    
# Récupération de l'indice de colère
y_train = df_train['Score'].astype(float)
y_test = df_test['Score'].astype(float)


X_train, vocab_train = prepare_data(df_train)
word_to_index = vocab_train.copy()  # Copie du dictionnaire pour une utilisation dans les données de test
X_test, _ = prepare_data_test(df_test, word_to_index=word_to_index)
X_hate,_ = prepare_data_test(df_hate, word_to_index=word_to_index)

# Création d'un modèle LSTM avec une couche d'embedding initialisée de manière aléatoire
model = Sequential()
model.add(Embedding(input_dim=len(vocab_train) + 1, output_dim=50))  # Modifier la dimension de l'embedding si nécessaire
model.add(Dropout(0.25))
model.add(LSTM(50))  # Vous pouvez modifier le nombre de neurones LSTM
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))  # Une seule sortie linéaire pour l'indice de colère
model.compile(loss='mean_squared_error', optimizer='adam')  # Remplacez par votre fonction de perte et votre optimiseur

# Entraînement du modèle LSTM sur les données d'entraînement
model.fit(np.array(X_train), np.array(y_train), epochs=10)  # Vous pouvez modifier le nombre d'époques

# Obtention des embeddings appris pour chaque tweet
embeddings = model.layers[0].get_weights()[0]

# Fonction pour obtenir les embeddings
def get_embeddings(X_pad):
    embeddings_means = []
    for indices in X_pad:
        # Check if indices array is not empty
        if indices.size:
            embeddings_mean = np.mean(embeddings[indices], axis=0)
        else:
            # If it is empty, use a vector of zeros
            embeddings_mean = np.zeros(shape=(len(embeddings[0]), ))
            print(1)
        embeddings_means.append(embeddings_mean)
    return embeddings_means

# Obtention des embeddings pour les données d'entraînement et de test
X_train_embeddings = get_embeddings(X_train)
X_test_embeddings = get_embeddings(X_test)
X_hate_embeddings = get_embeddings(X_hate)

# Conversion des listes en DataFrames
X_train_df = pd.DataFrame(X_train_embeddings)
X_test_df = pd.DataFrame(X_test_embeddings)
X_hate_df = pd.DataFrame(X_hate_embeddings)

# Concaténation des DataFrames y et X
df_output_train = pd.concat([y_train.reset_index(drop=True), X_train_df], axis=1)
df_output_test = pd.concat([y_test.reset_index(drop=True), X_test_df], axis=1)

column_names = ['fear_index'] + [f'feature_{i}' for i in range(len(embeddings[0]))]

df_output_train.columns = column_names
df_output_test.columns = column_names
X_hate_df.columns = [f'feature_{i}' for i in range(len(embeddings[0]))]


# Enregistrement au format CSV
df_output_train.to_csv('anger_mean_train.csv', index=False)
df_output_test.to_csv('anger_mean_test.csv', index=False)
X_hate_df.to_csv('anger_hate_2.csv', index=False)