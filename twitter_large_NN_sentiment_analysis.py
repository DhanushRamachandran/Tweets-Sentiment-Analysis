# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:53:14 2025

@author: sudha
"""

# twitter sentiment analysis - using all neural networks and comparison
# Neural networks included are: RNN, GRU,LSTM, Bi-directional layer

# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Dropout
from gensim.models import Word2Vec
import re
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_lg")
stop_words = stopwords.words("english")


# load the dataset
df = pd.read_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\datasets\sentiment140.csv",encoding="utf-8",encoding_errors="ignore")
print(df.columns)
df_mini = df[:1000]
df.describe()
df.drop(columns=["1467810369"],inplace=True)

df.rename(columns={"0":"target","NO_QUERY":"query","Mon Apr 06 22:19:45 PDT 2009":"date","_TheSpecialOne_":"user_id","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D":"tweet"},inplace=True)
df["date"]

def split(date):
    #print("d")
    date = date["date"]
    date = date.split(" ")
    
    return pd.Series([date[2],date[1],date[5]])
    

def get_len(tweet):
    return pd.Series([len(tweet.split(" ")),len(tweet)])

df_mini = df[:100]
df[["day","month","year"]] = df.apply(lambda x: split(x),axis=1)
df[["len","char_len"]] = df["tweet"].apply(lambda x: get_len(x))

df.drop(columns=["date"],inplace=True)

# splitting , clean the dataset
def clean_tweet(tweet):
    char_list = [chr(97+i) for i in range(0,26)]
    tweet = tweet.lower()
    tweet = " ".join(re.findall(r"[a-z]+", tweet))
    return tweet
    

df_mini["tweet"] = df_mini["tweet"].apply(clean_tweet) 

df["tweet"] = df["tweet"].apply(clean_tweet) 

# prepocess the dataset


# exploratory analysis
#univariate
df["user_id"].nunique()
df["user_id"].value_counts()
df["year"].value_counts()
df["month"].value_counts()
day_dist_df = pd.DataFrame(df["day"].value_counts())
day_dist_df["day"] = day_dist_df.index.tolist()
day_dist_df.reset_index(drop=True, inplace=True)
['target', 'user_id', 'tweet', 'day', 'month', 'year', 'len',
       'char_len']
df["target"].value_counts() 
df["target"] = df["target"].replace(4,1)
df["query"].value_counts() 
df.drop(columns = ["query"],inplace=True)

for col in df.columns:
    plt.figure(figsize=(8, 4))
    plt.plot(df[col], label=col)
    plt.title(f"Line Plot for '{col}'")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def merge_tweet(user_id,tweet):
    return pd.Series([user_id+" "+tweet])

df["tweet"] = df[["tweet","user_id"]].apply(lambda x: merge_tweet(x["user_id"],x["tweet"]),axis=1)

df.drop(columns=["user_id"],inplace=True)

df.to_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\datasets/sentiment140_preprocessed.csv")
# bivariate


#multi-variate

df = pd.read_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\datasets/sentiment140_preprocessed.csv")
# neural network methods

# getting features

chunk_size = 10000
n_chunks = np.ceil(len(df)/chunk_size).astype("int")

# building model
model = Sequential()
model.add(Dense(512,activation="relu",input_dim = 20000))
model.add(Dense(128,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.summary()
# traing the model


for i in range(n_chunks-64):
    print("Training")
    print("-"*50)
    print("Trainig chunk no: ",i+1)
    print("-"*50)
    chunk = df.iloc[i*chunk_size: (i+1)*chunk_size]
    tweets = chunk["tweet"].tolist()
    
    vectorizer = CountVectorizer(max_features=20000)    
    vectorizer.fit(tweets)
    
    input_features = vectorizer.transform(tweets).toarray()
    
    del tweets
    target = np.array([tar for tar in chunk["target"].tolist()])
    del chunk
    
    history_1 = model.fit(input_features[:8000],target[:8000],epochs=2,batch_size=100,validation_data=(input_features[8000:],target[8000:]) )
    

model.save(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\models/twt_NN_large.")


arr = [3,2,5,6,1,4,4]
all_areas = []
for i in range(0,len(arr)-1):
    inc_list = []
    if arr[i]<arr[i+1]:
        inc_list += [arr[i]]
        for j in range(0,len(inc_list)):
            all_areas += [inc_list[j]*(len(inc_list)-j+1)]
    else:
        inc_list = []
        start_index = -1
        
print(max(all_areas))

