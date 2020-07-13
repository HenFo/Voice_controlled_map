from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
from keras.models import load_model
import json
import os
import numpy as np
import pandas as pd

os.chdir("classifier")

path = os.path.join("Data", "models", "rnn_conflict_free")
model_name = "rnn_intent_classification.h5"

with open(os.path.join(path, "char_to_int.json"), "rt") as f:
    char_to_int = json.load(f)
with open(os.path.join(path, "int_to_char.json"), "rt") as f:
    int_to_char = json.load(f)
with open(os.path.join(path, "tag_to_int.json"), "rt") as f:
    tag_to_int = json.load(f)
with open(os.path.join(path, "int_to_tag.json"), "rt") as f:
    int_to_tag = json.load(f)
with open(os.path.join("Data", "class_tag.csv"), "rt") as f:
    class_tag = pd.read_csv(f)

print(int_to_tag)

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
            bag = np.zeros(len(char_to_int))
            bag[char_to_int[char]] = 1
            output.append(bag)
        except KeyError:
            continue
    
    while len(output) < max_length:
        output.append(np.zeros(len(char_to_int)))
    
    return np.array(output)


model = load_model(os.path.join(path, model_name))

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

    real_tag = int_to_tag[str(out_index)]
    print(real_tag)
    print(f"tag: {class_tag[class_tag.Class == int(real_tag)].Tag}")