"""libraries"""

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

#preparation of the data
df = pd.read_csv("data/general/labeled_data.csv", encoding='utf-8')
tweets=df.tweet
X=[]
for t in tweets:
    tokens = tf.hand_tokenize(tf.preprocess(t))
    X+=[tokens]

#recuperation of the label
y = df['class'].astype(int)
# Create a vocabulary of all unique words in your tweets
vocab = set(word for tweet in X for word in tweet)

# Create a dictionary that assigns a unique index to each word
word_to_index = {word: i for i, word in enumerate(vocab)}

# Convert tweets into lists of word indices
X_indices = [[word_to_index[word] for word in tweet] for tweet in X]

# Padding
X_pad = pad_sequences(X_indices)

# Create a LSTM model with a randomly initialized embedding layer
model = Sequential()
model.add(Embedding(input_dim=len(vocab) + 1, output_dim=50))  # Change the dimension of the embedding if necessary
model.add(Dropout(0.25))
model.add(LSTM(50))  # You can change the number of LSTM neurons
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # There are 3 classes: hateful, offensive, neutral
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')  # Replace with your loss function and optimizer

# Train the LSTM model
model.fit(np.array(X_pad), np.array(y), epochs=10)  # You can change the number of epochs
# Get the learned embeddings for each tweet
embeddings = model.layers[0].get_weights()[0]

embeddings_means = []
for indices in X_pad:
    embeddings_mean = np.mean(embeddings[indices], axis=0)
    embeddings_means.append(embeddings_mean)

X=embeddings_means

# Convert the list X to a DataFrame
X_df = pd.DataFrame(X)

# Concatenate y and X_df
df_output = pd.concat([y, X_df], axis=1)
column_names = ['label'] + [f'feature_{i}' for i in range(len(embeddings))]
# Save to CSV
df_output.to_csv('data/LSTM_random_embedding/mean.csv', index=False)