# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:37:00 2025

@author: sudha
"""

# sentiment analysis usig LSTM architecture
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dense,LSTM,Embedding,Dropout
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



df = pd.read_csv(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\datasets/sentiment140_preprocessed.csv")

df.drop(columns=["user_id"],inplace=True)

# get the most importtant 20000 features/ words

tweets = df["tweet"].tolist()

vectorizer = CountVectorizer(max_features=20000)    
vectorizer.fit(tweets)
words = list(vectorizer.get_feature_names_out())
word_index = dict()
i =1
for word in words:
    word_index[word] = i
    i+=1

del words


# encode the tweets in sequences

def sequence_tweets(tweet):
    tweet_seq = []
    for word in word_tokenize(tweet):
        #print("word: ",word)
        if word in word_index.keys():
            tweet_seq += [str(word_index[word])]
    print("seq:" ,tweet_seq)
    return ",".join(tweet_seq)

df["tweet_enc"] = df["tweet"].apply(lambda x: sequence_tweets(x))

# preprocess input features
max_len = 0
i=1
for seq in df["tweet_enc"]:
    print(i)
    i+=1
    cur_len = len(seq[1:-1].split(","))
    max_len = max(cur_len,max_len)

i=1
no_seq = []
final_inp_features = []
for seq in df["tweet_enc"].tolist():
    print(seq)
    print("-"*30)
    print(i)
    i+=1
    if seq != "":
        inp_feature = seq.split(",")
        inp_feature = [int(num) for num in inp_feature]
        print(inp_feature)
        cur_len = len(inp_feature)
        inp_feature += [0 for i in range(max_len-cur_len)]
        print(inp_feature)
        final_inp_features += [inp_feature]
    else:
        inp_feature = [0 for i in range(50)]
        no_seq.append(i)


# Step 1: Fit tokenizer on your texts
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["tweet"].tolist()[:20000])  # raw_texts is your list of strings

# Step 2: Convert texts to sequences
sequences = tokenizer.texts_to_sequences(df["tweet"].tolist()[:20000])


# Step 3: Pad sequences to the same length
x_data = pad_sequences(sequences, maxlen=50, padding='post')  # shape: (samples, 50)


vocab_size = len(tokenizer.word_index) +1


# feed sequence into LSTM/RNN
# Then rebuild the model from scratch with correct input shape
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=50))  # input_dim = num unique tokens
model.add(LSTM(units=100,dropout=0.2, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dropout(0.6))
model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer= Adam(learning_rate=0.0001) , metrics=["accuracy"])

final_inp_features = [np.array(arr) for arr in final_inp_features]
target_all = df["target"].tolist()

final_inp_features = [np.reshape(arr, (len(arr), 1)) for arr in final_inp_features]
final_inp_features = np.array(final_inp_features)


final_target = []
for i in range(len(target_all)):
    if i not in no_seq:
        final_target += [target_all[i]]
        
del target_all
print(len(final_inp_features))
print(len(final_target))

x_train,x_test,y_train,y_test = train_test_split(x_data,final_target[:20000])


x_train = x_data[:16000]
x_test = x_data[16000:]

y_train = final_target[:1000000]
y_test = final_target[1000000:]
y_train = np.array(y_train)
y_test = np.array(y_test)


for i in range(len(y_train)):
    if y_train[i] == 4:
        y_train[i] = 1
        

for i in range(len(y_test)):
    if y_test[i] == 4:
        y_test[i] = 1
        
print("x_train shape:", x_train.shape)  # should be (samples, timesteps, 1)
print("y_train shape:", y_train.shape)

chunk_size = 100000
n_chunks = int(len(x_train)/chunk_size)

for i in range(0,n_chunks-1):
    
    model.fit(x_train[i*chunk_size:(i+1)*chunk_size],y_train[i*chunk_size:(i+1)*chunk_size],epochs = 2,batch_size=128)
    
model.fit(x_train,y_train,epochs = 10,batch_size=128,validation_data=(x_test, y_test))

model.save(r'C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\models/tt_LSTM_model.h5')


