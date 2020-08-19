import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
from keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer


os.chdir("classifier")
# load data
with open(os.path.join("Data", "commands", "Training", "all_directions", "final_vocabulary_all_f.json"), "rt") as f:
    t_data = json.load(f)

with open(os.path.join("Data", "class_tag.csv"), "rt") as f:
    class_tag = pd.read_csv(f)


training_data = {}
# for t, cs in t_data.items():
#     com = [c["c"] for c in cs]
#     training_data[t] = com
for t, cs in t_data.items():
    com = [c[0] for c in cs["commands"]]
    training_data[t] = com


allCommands = []
for commands in training_data.values():
    allCommands.extend(commands)
text = " ".join(allCommands)

unique_chars = set(text)

int_to_char = {}
char_to_int = {}
for i, j in enumerate(unique_chars):
    int_to_char[i] = j
    char_to_int[j] = i

tag_to_int = {}
int_to_tag = {}
for i, j in enumerate(training_data):
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


def transform_command(command: str):
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


num_tags = len(training_data)
X_training = []
y_training_one_hot = []
y_training = []

for tag, commands in training_data.items():
    for command in commands:
        X_training.append(transform_command(command))
        y_training_one_hot.append(to_categorical(tag_to_int[tag], num_tags))
        y_training.append(tag_to_int[tag])


X_training = np.array(X_training)
y_training_one_hot = np.array(y_training_one_hot)

input_shape = X_training.shape[1:]
output_shape = y_training_one_hot.shape[1]

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
# model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(output_shape, activation="softmax"))

model.compile("rmsprop", "categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stopping = EarlyStopping(
    monitor="loss", min_delta=0.01, patience=10, restore_best_weights=True)
model.fit(X_training, y_training_one_hot, epochs=200, batch_size=64, callbacks=[early_stopping])

model.save(os.path.join("Data", "models", "rnn_intent_classification.h5"))

predictions = model.predict(X_training)
locations = np.argmax(predictions, 1)

print(confusion_matrix(y_training, locations))
print(accuracy_score(y_training, locations))


stemmer = GermanStemmer()
# with open(os.path.join("Data", "commands", "stopwords.txt"), "rt") as f:
#     stopwords = set(f.read().splitlines())
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