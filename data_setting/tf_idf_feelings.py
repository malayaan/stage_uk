"""libraries"""
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import emoji 
import warnings
import html
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# Ignorer les avertissements spécifiques
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'")
warnings.filterwarnings("ignore", category=UserWarning, message="Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens ['b', 'c', 'e', 'f', 'g', 'h', 'j', 'l', 'n', 'p', 'r', 'u', 'v', 'w'] not in stop_words.")

"""variables"""

df = pd.read_csv("data/general/fear_tweets.csv", encoding='utf-8')
stopwords = nltk.corpus.stopwords.words("english")
tweets=df.Tweet
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
stemmer = PorterStemmer()
sentiment_analyzer = VS()
nltk.download('punkt')

"""functions"""


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) html entities with their corresponding characters.
    5) emojis with words

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    # convert HTML entities to their corresponding characters
    # parsed_text = html.unescape(parsed_text)
    # translate emojis to words
    parsed_text = emoji.demojize(parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

def hand_tokenize(tweet):
    # Set to lowercase
    tweet = tweet.lower()
    
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(tweet)
    
    # Remove excess whitespace
    tokens = [token.strip() for token in tokens]
    
    # Stem tokens
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return stemmed_tokens

#Now get other features
def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))
"""
# Display preprocessing results for the first 15 tweets
for i in range(15):
    print(f'Original tweet:\n{tweets[i]}\n')
    print(f'Preprocessed tweet:\n{preprocess(tweets[i])}\n')
    print('-----------------------------------------------------\n')
"""
def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    
    words = preprocess(tweet) #Get text only
    
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    #features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats=[]
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

if __name__ == '__main__':
    
    vectorizer = TfidfVectorizer(
        tokenizer=hand_tokenize,
        preprocessor=preprocess,
        ngram_range=(1, 3),
        stop_words=stopwords,
        use_idf=True,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=10000,#Ne considère que les 10 000 termes les plus fréquemment rencontrés dans le corpus.
        min_df=0,#les mots qui apparaissent dans moins que min_df docs sont ignoré
        max_df=0.75 #Les mots qui apparaissent dans plus de 75% des documents sont également ignorés.
        )

    # Construire la matrice TF-IDF
    tfidf = vectorizer.fit_transform(tweets).toarray()

    # cette partie met en place les pos tag pour determminer les fonctions gramaticales de mots
    # Récupérer le vocabulaire et les valeurs IDF
    vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names_out())}
    idf_vals = vectorizer.idf_
    idf_dict = {i:idf_vals[i] for i in vocab.values()}
    #Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in tweets:
        tokens = hand_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)

    #We can use the TFIDF vectorizer to get a token matrix for the POS tags
    pos_vectorizer = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.75,
        )

    #Construct POS TF matrix and get vocab dict
    pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
    pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names_out())}

    other_features_names = ["FKRA", "FRE","num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total", \
                        "num_terms", "num_words", "num_unique_words", "vader neg","vader pos","vader neu", \
                        "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]
    #mets en place des éléments d'interprétabilité du tweet
    feats = get_feature_array(tweets)
    #Now join them all up
    M = np.concatenate([tfidf,pos,feats],axis=1)
    #running the model
    X = pd.DataFrame(M)
    y = df['Score'].astype(float)
    
    
    # Séparation des données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Utilisation de Lasso comme modèle de base pour SelectFromModel
    select = SelectFromModel(Lasso(alpha=0.005))

    # Fit le SelectFromModel sur les données d'entraînement
    select.fit(X_train, y_train)

    # Transformer les données d'entraînement et de test
    X_train_select = select.transform(X_train)
    X_test_select = select.transform(X_test)

    # Convertir les matrices transformées en DataFrames
    X_train_select_df = pd.DataFrame(X_train_select)
    X_test_select_df = pd.DataFrame(X_test_select)

    # Concaténation des DataFrames y et X
    df_output_train = pd.concat([y_train.reset_index(drop=True), X_train_select_df], axis=1)
    df_output_test = pd.concat([y_test.reset_index(drop=True), X_test_select_df], axis=1)

    # Assigner les noms de colonnes
    column_names = ['fear_index'] + [f'feature_{i}' for i in range(X_train_select_df.shape[1])]
    df_output_train.columns = column_names
    df_output_test.columns = column_names

    # Enregistre les données transformées
    df_output_train.to_csv('data/feelings/tf_idf/fear/0.005_fear_train_select.csv', index=False)
    df_output_test.to_csv('data/feelings/tf_idf/fear/0.005_fear_test_select.csv', index=False)
