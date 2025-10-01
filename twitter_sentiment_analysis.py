# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 19:26:06 2025

@author: sudha
"""

# twitter sentiment analysis naive bayes model

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
import spacy
nlp = spacy.load("en_core_web_lg")


# preparing the dataset
df_twitter = pd.read_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\datasets\twitter_training.csv",encoding="utf-8")
df_twitter = pd.concat([df_twitter,pd.DataFrame([df_twitter.columns.tolist()],columns=df_twitter.columns)])

df_twitter = df_twitter[df_twitter.columns.tolist()[2:]]
df_twitter.rename(columns ={"Positive":"sentiment",df_twitter.columns.tolist()[1]:"tweet"},inplace=True)
df_twitter.fillna("-",inplace=True)


# text preprocessing
stop_words = stopwords.words()
char_list = [chr(i) for i in range(0,200) if i<97 or i>123]
stop_words += char_list


count = 0
for i,row in df_twitter.iterrows():
    string = word_tokenize(row["tweet"])
    string = [word.lower() for word in string if word.lower() not in stop_words]
    row["tweet"] = " ".join(string)
    
    print("\n\ncount: ",count)
    count += 1
    

count = 0
for i,row in df_twitter.iterrows():
    new_tweet = ""
    for char in row["tweet"]:
        if char != " ":
            if not (ord(char)<97 or ord(char)>123):
                new_tweet += char
        else:
            new_tweet += " "
    row["tweet"] = new_tweet
    print("\ncount: ",count)
    count+=1
            

count = 0
for i,row in df_twitter.iterrows():
    new_tweet = []
    for word in row["tweet"].split(" "):
        if len(word)>2:
            new_tweet+=[word]
            
    
    row["tweet"] = " ".join(new_tweet)
    print("\ncount: ",count)
    count+=1
            

sentiments = df_twitter['sentiment'].unique().tolist()
sent_dict = {"Positive":3,"Neutral":2,"Negative":0,"Irrelevant":1}

df_twitter["sentiment"] = df_twitter["sentiment"].apply(lambda x: sent_dict[x])
df_twitter = df_twitter[~df_twitter["sentiment"].isin([1, 0])]



# text encoding
vectorizer = TfidfVectorizer()
x_sparse = vectorizer.fit_transform(list(df_twitter["tweet"])) 


df_broken = pd.DataFrame.sparse.from_spmatrix(x, columns=vectorizer.get_feature_names_out())
df_broken = df_broken.reset_index(drop=True)
df_twitter = df_twitter.reset_index(drop=True)





df_broken["sentiment_y"] = df_twitter["sentiment"]

y = df_broken.sum()
y = list(y)
y_dict = {}
cols = df_broken.columns
for i in range(0,len(df_broken.columns)):
    y_dict[y[i]] = cols[i]


y = sorted(y)    
del_cols = []
for val in y[:5000]:
    col = y_dict[val]
    del_cols+=[col]

chunk_size = 1000
for i in range(0, len(del_cols), chunk_size):
    df_broken.drop(columns=del_cols[i:i+chunk_size], inplace=True)
    print(i)


df_broken = df_broken.astype('float32')

df_broken = df_broken.drop(columns="sentimen")

df_broken = df_broken[:30000]


df_broken.isnull().sum()
df_twitter = df_twitter[["sentiment"]]    
    

# model training

y = df_twitter["sentiment"]
model = MultinomialNB()
x_train,x_test,y_train,y_test =  train_test_split(x_sparse ,y,train_size=0.8)
model.fit(x_train,y_train)



# Step 5: Evaluate
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

cols = vectorizer.get_feature_names_out()

cols = list(cols)
final_cols = []
for col in cols:
    if col in nlp.vocab:
        final_cols += [col]
        
len(final_cols)
len(cols) 
"acknowledge" in nlp.vocab.vectors["acknowledge"]  

from nltk.corpus import wordnet

def is_legit_wordnet(word):
    return len(wordnet.synsets(word)) > 0

cols = list(cols)
final_cols = []
for col in cols:
    if is_legit_wordnet(col):
        final_cols += [col]

df_broken = df_broken[final_cols]


# model training
from scipy.sparse import csr_matrix

x_sparse = csr_matrix(df_broken.values)



y = df_twitter["sentiment"]
model = MultinomialNB()
x_train,x_test,y_train,y_test =  train_test_split(df_broken.values() ,y,train_size=0.8)
model.fit(df_broken[:30000],y[:30000])



# Step 5: Evaluate
y_pred = model.predict(df_broken[10000:20000])
print(classification_report(y[10000:20000], y_pred))

df_broken.to_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\twitter_sentiment_analysis/train_test_processed_dataset.csv")

