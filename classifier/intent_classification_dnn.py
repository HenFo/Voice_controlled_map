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
with open(os.path.join("Data", "commands","Training", "all_directions", "final_vocabulary_all_f.json"), "rt") as f:
    t_data = json.load(f)
# with open(os.path.join("Data", "commands","Test", "testingdata.json"), "rt") as f:
#     e_data = json.load(f)

with open(os.path.join("Data", "class_tag.csv"), "rt") as f:
    class_tag = pd.read_csv(f)

cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")

training_data = {}
# for t, cs in t_data.items():
#     com = [c["c"] for c in cs]
#     training_data[t] = com
for t, cs in t_data.items():
    com = [c[0] for c in cs["commands"]]
    training_data[t] = com
    


documents = []
for doc in training_data.values():
    documents.append(" ".join(command for command in doc))

cv.fit(documents)
pickle.dump(cv, open("cv.p", "wb"))

def transform_command(command:str):
    return cv.transform([command]).toarray()[0]


tag_to_int = {}
int_to_tag = {}
for i,j in enumerate(training_data):
    tag_to_int[j] = i
    int_to_tag[i] = j

with open(os.path.join("Data", "models", "tag_to_int.json"), "wt") as f:
    json.dump(tag_to_int, f)
with open(os.path.join("Data", "models", "int_to_tag.json"), "wt") as f:
    json.dump(int_to_tag, f)

num_tags = len(training_data)
X_training = []
y_training_one_hot = []
y_training = []

for tag, commands in training_data.items():
    for command in commands:
        X_training.append(transform_command(command))
        y_training_one_hot.append(to_categorical(tag_to_int[tag], num_tags))
        y_training.append(tag_to_int[tag])

# X_test = []
# y_test = []

# for tag, commands in e_data.items():
#     for command in commands:
#         X_test.append(transform_command(command))
#         y_test.append(tag_to_int[tag])

X_training = np.array(X_training)
y_training_one_hot = np.array(y_training_one_hot)
# X_test = np.array(X_test)
# y_test = np.array(y_test)

input_shape = X_training.shape[1:]
output_shape = y_training_one_hot.shape[1]

model = Sequential()
model.add(Dense(64, activation="relu", input_shape=input_shape))
model.add(Dense(output_shape, activation="softmax"))

model.compile("rmsprop", "categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor="loss", min_delta=0.01, patience=10, restore_best_weights=True)
model.fit(X_training,y_training_one_hot, epochs=200, batch_size=16, callbacks=[early_stopping])

model.save(os.path.join("Data", "models", "dnn_intent_classification.h5"))

predictions = model.predict(X_training)
locations = np.argmax(predictions, 1)

cm = confusion_matrix(y_training, locations)
print(cm)
print(accuracy_score(y_training, locations))

print(np.where(cm == 1))
# predictions = model.predict(X_test)
# locations = np.argmax(predictions, 1)

# print(confusion_matrix(y_test, locations))
# print(accuracy_score(y_test, locations))

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
    # print(f"tag: {class_tag[class_tag.Class == int(real_tag)].Tag}")
