# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 00:21:15 2025

@author: sudha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:37:00 2025

@author: sudha
"""

# sentiment analysis usig LSTM architecture
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense,LSTM,Embedding,Dropout,GRU,Bidirectional
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_lg")
stop_words = stopwords.words()

df = pd.read_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\datasets\sent_analysis_new_train.csv",on_bad_lines="skip",encoding="latin",header=None)
df.head(5)
df = df[[1,3]]

df=df.rename(columns={1:"text",3:"labels"})
df=df[["labels","tweets"]]

df["labels"] = df["labels"].replace({"neutral":2,"positive":4,"negative":0})
df.isna().sum()
df.dropna(inplace=True)

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)  # remove links/mentions/hashtags
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text

def clean_text_2(text):
    text = [token.lemma_ for token in nlp(text)]
    text = [token for token in text if ((token not in stop_words) and (len(token)>1))]
    text = " ".join(text)
    #print("final text: ",text)
    return text

df["text"] = df["text"].apply(clean_text)

df["clean_text"] = df["text"].apply(clean_text_2)


df_sample = df.sample(n=40000,random_state=42)

df_sample["clean_tweets"] = df_sample["tweets"].apply(clean_text_2)


df.drop(columns=["user_id"],inplace=True)

# get the most importtant 20000 features/ words
df_sample = df.sample(n=40000, random_state=42)  # random_state ensures reproducibility
df_sample = df.head(40000)

df =df[~df["clean_text"].isin(["text"])]
tweets = df["clean_text"].tolist()


# Step 1: Fit tokenizer on your texts
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(tweets)  # raw_texts is your list of strings

# Step 2: Convert texts to sequences
sequences = tokenizer.texts_to_sequences(tweets)


# Step 3: Pad sequences to the same length
x_data = pad_sequences(sequences, maxlen=50, padding='post')  # shape: (samples, 50)


vocab_size = len(tokenizer.word_index) +1

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=50))
model.add(GRU(32))
model.add(Dense(1, activation="sigmoid"))


# feed sequence into LSTM/RNN
# Then rebuild the model from scratch with correct input shape
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=50))  # input_dim = num unique tokens
model.add((GRU(units=128,dropout=0.2, return_sequences=True)))
model.add((GRU(units=50)))
model.add(Dropout(0.4))
model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer= Adam(learning_rate=0.0001) , metrics=["accuracy"])

target_all = df_sample["target"].tolist()

target_all = df["labels"].tolist()

target_all = np.array(target_all)
target_all = target_all[1:]

target_all = np.array([
    ([1, 0, 0]) if num == 0 else
    [0, 1, 0] if num == 2 else
    [0, 0, 1] 
    for num in target_all
])

#target_all = np.array([1 if t == 4 else t for t in target_all]) 
x_train,x_test,y_train,y_test = train_test_split(x_data,target_all,test_size = 0.2,random_state=42)

print("x_train shape:", x_train.shape)  # should be (samples, timesteps, 1)
print("y_train shape:", y_train.shape)
    
model.fit(x_train,y_train,epochs = 10,batch_size=128,validation_data=(x_test, y_test))

model.save(r'C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\models/tt_LSTM_model.h5')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

y_train_ml = np.array([
    0 if num == [1, 0, 0] else
    2 if num == [0, 1, 0] else
    4 
    for num in target_all
])
index_to_label = {0: 0, 2: 1, 4: 2}

# Use argmax to get index of the 1 in each one-hot row
y_train_ml = np.array([index_to_label[idx] for idx in np.argmax(target_all, axis=1)])

clf = LogisticRegression()
clf.fit(x_train, y_train)

pred = clf.predict(x_test)

print("Accuracy:", accuracy_score(y_test, pred))



df["tweet"] = df["tweet"].apply(clean_text)


del df,df_sample
