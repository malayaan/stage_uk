"""libraries"""
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from collections import Counter
import joblib
import test as tt
from nltk.tokenize import word_tokenize
import string

with open("data/trump_tweets.csv", 'r', encoding='utf-8', errors='ignore') as file:
    df = pd.read_csv(file)
df = df.dropna(subset=['Text'])

stopwords = tt.stopwords

tweets=df.Text


stemmer = PorterStemmer()

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

#Now get other features
sentiment_analyzer = VS()

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

def load_model(file_name):
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model

def get_tweets_predictions(tweets, perform_prints=True):
    fixed_tweets = []
    for i, t_orig in enumerate(tweets):
        s = t_orig
        try:
            s = s.encode("latin1")
        except:
            try:
                s = s.encode("utf-8")
            except:
                pass
        if type(s) != str:
            fixed_tweets.append(str(s, errors="ignore"))
        else:
            fixed_tweets.append(s)
    assert len(tweets) == len(fixed_tweets), "shouldn't remove any tweets"
    tweets = fixed_tweets
    if perform_prints: print(len(tweets), " tweets to classify")

    if perform_prints: print("Loading trained classifier... ")
    model = joblib.load('study1/model.pkl')

    if perform_prints: print("Loading other information...")
    vectorizer = joblib.load('study1/vectorizer.pkl')
    pos_vectorizer = joblib.load('study1/pos_vectorizer.pkl')
    #select = joblib.load('select_model.pkl')

    if perform_prints: print("Transforming inputs...")
    tfidf = vectorizer.transform(tweets).toarray()
    tweet_tags = []
    for t in tweets:
        tokens =hand_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    pos = pos_vectorizer.transform(pd.Series(tweet_tags)).toarray()
    feats = get_feature_array(tweets)
    M = np.concatenate([tfidf,pos,feats],axis=1)
    X = pd.DataFrame(M)
    #X = select.transform(X)

    if perform_prints: print("Running classification model...")
    predicted_class = model.predict(X)

    return predicted_class

def hand_tokenize(tweet):
    # Set to lowercase
    tweet = tweet.lower()
    
    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '',string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(tweet)
    
    # Remove excess whitespace
    tokens = [token.strip() for token in tokens]
    
    # Stem tokens
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return stemmed_tokens

model = load_model('study1/model.pkl')

# Charger le SelectFromModel
select = load_model('study1/select_model.pkl')

# Charger le vectorizer
vectorizer = load_model('study1/vectorizer.pkl')
# Construire la matrice TF-IDF
tfidf = vectorizer.transform(tweets).toarray()

# Charger le pos_vectorizer
pos_vectorizer = load_model('study1/pos_vectorizer.pkl')
#Get POS tags for tweets and save as a string
tweet_tags = []
for t in tweets:
    tokens = hand_tokenize(preprocess(t))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)
pos = pos_vectorizer.transform(pd.Series(tweet_tags)).toarray()

# Prétraiter les tweets
trump_tweets_preprocessed = [preprocess(t) for t in tweets]
#mets en place des éléments d'interprétabilité du tweet
feats = get_feature_array(tweets)
#Now join them all up
M = np.concatenate([tfidf,pos,feats],axis=1)
#running the model
X = pd.DataFrame(M)

