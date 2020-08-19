from keras.models import load_model
import json
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from nltk.stem.snowball import GermanStemmer
from nltk.tokenize import word_tokenize

def transform_command_BoW(command:str, vectorizer):
    return vectorizer.transform([command]).toarray()[0]

def evaluate_dnn(path:str):
    with open(os.path.join(path, "tag_to_int.json"), "rt") as f:
        tag_to_int = json.load(f)
    with open(os.path.join(path, "int_to_tag.json"), "rt") as f:
        int_to_tag = json.load(f)  

    cv = pickle.load(open(os.path.join(path, "cv.p"), "rb"))
    stemmer = GermanStemmer()
    model_name = "dnn_intent_classification.h5"
    model = load_model(os.path.join(path, model_name))

    with open(os.path.join("Data", "commands", "Test", "testingdata.json"), "rt") as f:
        val_data = json.load(f)

    X = []
    y = []

    for tag, commands in val_data.items():
        for command in commands:
            command = " ".join(stemmer.stem(c) for c in sorted(word_tokenize(command)))
            X.append(transform_command_BoW(command, cv))
            y.append(tag_to_int[tag])

    X = np.array(X)
    y = np.array(y)

    predictions =  model.predict(X)
    predicted_indices = np.argmax(predictions, 1)

    print("acc: ", accuracy_score(y, predicted_indices))
    cm = confusion_matrix(y, predicted_indices)
    cm = pd.DataFrame(cm, index=int_to_tag.values(), columns=int_to_tag.values())
    print(cm)

    return (accuracy_score(y, predicted_indices), cm)



def transform_command_sequence(command:str, char_to_int:dict):
    max_length = 30
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

def evaluate_rnn(path:str):
    with open(os.path.join(path, "char_to_int.json"), "rt") as f:
        char_to_int = json.load(f)
    with open(os.path.join(path, "int_to_char.json"), "rt") as f:
        int_to_char = json.load(f)
    with open(os.path.join(path, "tag_to_int.json"), "rt") as f:
        tag_to_int = json.load(f)
    with open(os.path.join(path, "int_to_tag.json"), "rt") as f:
        int_to_tag = json.load(f)  

    model_name = "rnn_intent_classification.h5"
    model = load_model(os.path.join(path, model_name))
    stemmer = GermanStemmer()

    with open(os.path.join("Data", "commands", "Test", "testingdata.json"), "rt") as f:
        val_data = json.load(f)

    X = []
    y = []

    for tag, commands in val_data.items():
        for command in commands:
            command = " ".join(stemmer.stem(c) for c in sorted(word_tokenize(command)))
            X.append(transform_command_sequence(command, char_to_int))
            y.append(tag_to_int[tag])

    X = np.array(X)
    y = np.array(y)

    predictions =  model.predict(X)
    predicted_indices = np.argmax(predictions, 1)

    print("acc: ", accuracy_score(y, predicted_indices))
    cm = confusion_matrix(y, predicted_indices)
    cm = pd.DataFrame(cm, index=int_to_tag.values(), columns=int_to_tag.values())
    print(cm)
    return (accuracy_score(y, predicted_indices), cm)


if __name__ == "__main__":
    os.chdir("classifier")

    def saveFile(path:str, name:str, *data):
        with open(os.path.join(path, name), "wt") as f:
            f.writelines([str(x) + "\n" for x in data])


    model_path = os.path.join("Data", "models")
    eval_path = os.path.join(model_path, "evaluation")

    result = evaluate_rnn(os.path.join(model_path, "rnn_conflict_free_AD"))
    saveFile(eval_path, "rnn_conflict_free_AD", result[0], result[1])
    result = evaluate_rnn(os.path.join(model_path, "rnn_vocab_all_f"))
    saveFile(eval_path, "rnn_vocab_all_f", result)
    # result = evaluate_rnn(os.path.join(model_path, "rnn_conflict_free"))
    # saveFile(eval_path, "rnn_conflict_free", result)
    # result = evaluate_rnn(os.path.join(model_path, "rnn_vocabulary"))
    # saveFile(eval_path, "rnn_vocabulary", result)

    result = evaluate_dnn(os.path.join(model_path, "dnn_conflict_free_AD"))
    saveFile(eval_path, "dnn_conflict_free_AD", result[0], result[1])
    result = evaluate_dnn(os.path.join(model_path, "dnn_vocab_all_f"))
    saveFile(eval_path, "dnn_vocab_all_f", result)
    # result = evaluate_dnn(os.path.join(model_path, "dnn_conflict_free"))
    # saveFile(eval_path, "dnn_conflict_free", result)
    # result = evaluate_dnn(os.path.join(model_path, "dnn_vocabulary"))
    # saveFile(eval_path, "dnn_vocabulary", result)
    
