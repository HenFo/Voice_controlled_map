import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from nltk.stem.snowball import GermanStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix


os.chdir("classifier")
# load data
with open(os.path.join("Data", "commands","Training", "combined_commands.json"), "rt") as f:
    data = json.load(f)

with open(os.path.join("Data", "class_tag.csv"), "rt") as f:
    class_tag = pd.read_csv(f)

cv = CountVectorizer()

new_data = data
# for tag, i in data.items():
#     new_data[tag] = [c[0] for c in i["commands"]]


documents = []
for doc in new_data.values():
    documents.append(" ".join(command for command in doc))


cv.fit(documents)
# pickle.dump(cv, open("cv.p", "wb"))

def transform_command(command:str):
    return cv.transform([command]).toarray()[0]


tag_to_int = {}
int_to_tag = {}
for i,j in enumerate(new_data):
    tag_to_int[j] = i
    int_to_tag[i] = j


num_tags = len(new_data)
X = []
y = []
y_test = []

for tag, commands in new_data.items():
    for command in commands:
        X.append(transform_command(command))
        y.append(to_categorical(tag_to_int[tag], num_tags))
        y_test.append(tag_to_int[tag])

X = np.array(X)
y = np.array(y)

input_shape = X.shape[1:]
output_shape = y.shape[1]

print(X)
print(input_shape)
print(output_shape)

model = Sequential()
model.add(Dense(64, activation="relu", input_shape=input_shape))
model.add(Dense(64, activation="relu"))
model.add(Dense(output_shape, activation="softmax"))

model.compile("rmsprop", "categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor="loss", min_delta=0.01, patience=10, restore_best_weights=True)
model.fit(X,y, epochs=200, batch_size=16, callbacks=[early_stopping])

# model.save(os.path.join("Data", "models", "dnn_intent_classification.h5"))

predictions = model.predict(X)
locations = np.argmax(predictions, 1)

print(confusion_matrix(y_test, locations))
print(accuracy_score(y_test, locations))


stemmer = GermanStemmer()
while True:
    c = input("Your Input:")

    if c == "q":
        break

    c = " ".join(sorted([stemmer.stem(x) for x in word_tokenize(c.lower())]))

    c = np.array([transform_command(c)])
    prediction = model.predict(c)

    out_index = np.argmax(prediction)

    print(f"acc: {prediction[0, out_index]}")

    real_tag = int_to_tag[out_index]
    print(real_tag)
    print(f"tag: {class_tag[class_tag.Class == int(real_tag)].Tag}")
