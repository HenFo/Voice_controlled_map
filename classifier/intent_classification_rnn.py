import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer


os.chdir("classifier")
# load data
with open(os.path.join("Data", "commands", "Training", "final_vocabulary.json"), "rt") as f:
    data = json.load(f)

with open(os.path.join("Data", "class_tag.csv"), "rt") as f:
    class_tag = pd.read_csv(f)

# print(class_tag.head())

new_data = data
for tag, i in data.items():
    new_data[tag] = [c[0] for c in i["commands"]]


allCommands = []
for commands in new_data.values():
    allCommands.extend(commands)
text = " ".join(allCommands)

unique_chars = set(text)

int_to_char = {}
char_to_int = {}
for i,j in enumerate(unique_chars):
    int_to_char[i] = j
    char_to_int[j] = i

tag_to_int = {}
int_to_tag = {}
for i,j in enumerate(data):
    tag_to_int[j] = i
    int_to_tag[i] = j

with open(os.path.join("Data", "models", "char_to_int.json"), "w") as f:
    json.dump(char_to_int, f)

with open(os.path.join("Data", "models", "int_to_char.json"), "w") as f:
    json.dump(int_to_char, f)

with open(os.path.join("Data", "models", "tag_to_int.json"), "w") as f:
    json.dump(tag_to_int, f)

with open(os.path.join("Data", "models", "int_to_tag.json"), "w") as f:
    json.dump(int_to_tag, f)

max_length = 30
def transform_command(command:str):
    output = []
    i = 0
    for char in reversed(command):
        if i >= max_length:
            break

        i += 1

        char = char.lower()
        try:
            bag = np.zeros(len(unique_chars))
            bag[char_to_int[char]] = 1
            output.append(bag)
        except KeyError:
            continue
    
    while len(output) < max_length:
        output.append(np.zeros(len(unique_chars)))
    
    return np.array(output)

num_tags = len(data)
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

print(type(X[0]))
print(y.shape)

input_shape = X.shape[1:]
output_shape = y.shape[1]

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Dense(output_shape, activation="softmax"))

model.compile("rmsprop", "categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(monitor="loss", min_delta=0.01, patience=10, restore_best_weights=True)
model.fit(X,y, epochs=200, batch_size=32, callbacks=[early_stopping])

model.save(os.path.join("Data", "models", "rnn_intent_classification.h5"))

predictions = model.predict(X)
locations = np.argmax(predictions, 1)

print(confusion_matrix(y_test, locations))
print(accuracy_score(y_test, locations))


stemmer = GermanStemmer()
while True:
    c = input("Your Input:")

    if c == "q":
        break

    # print(f"requested: {c}")

    c = " ".join(sorted([stemmer.stem(x) for x in word_tokenize(c.lower())]))

    c = np.array([transform_command(c)])
    prediction = model.predict(c)

    out_index = np.argmax(prediction)

    print(f"acc: {prediction[0, out_index]}")

    real_tag = int_to_tag[out_index]
    print(real_tag)
    print(f"tag: {class_tag[class_tag.Class == int(real_tag)].Tag}")














# neural_net = FFNN()

# tags = sorted(list(data.keys()))
# X, y, y_normal = [], [], []
# pos_to_tag = {}
# for i, tag in enumerate(tags):
#     for command in data.get(tag):
#         X.append(cv.transform([command]).toarray().reshape(-1,))
#         y.append(to_categorical(i, len(tags)))
#         y_normal.append(i)
#     pos_to_tag[i] = tag

# X = np.array(X)
# y = np.array(y)
# y_normal = np.array(y_normal)


# input_shape = X.shape[1:]
# output_size = y.shape[1]

# neural_net.build(input_shape, output_size)

# # earlyStopping = EarlyStopping(monitor="loss", min_delta=0.005, restore_best_weights=True, patience=10)
# neural_net.fit(X, y, epochs=50, batch_size = 16)

# predict = neural_net.predict(X)
# locations = np.argmax(predict, 1)

# print(confusion_matrix(y_normal, locations))
# print(accuracy_score(y_normal, locations))
