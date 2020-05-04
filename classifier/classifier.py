import os
from pprint import pprint

import numpy as np
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from preprocessing import normalize_command
from fuzzywuzzy import fuzz, process

class Fuzzy_Classifier():
    def __init__(self, vocabulary:dict):
        self.vocab = vocabulary
        self.tags = sorted(list(vocabulary.keys()))
        self.tag_to_index = {k:v for v,k in enumerate(self.tags)}

    def predict(self, command:str, threshold:int = 50):
        predictions = process.extractBests(command, self.vocab, score_cutoff=threshold)
        return [self.tag_to_index.get(tag) for _, _, tag in predictions]
        



class NN():
    def __init__(self):
        self.model = None
        self.input_shape = None
        self.num_outputs = None

    def build(self, input_shape: tuple, num_outputs: int):
        raise NotImplementedError

    def transform_data(self, data: dict, vectorizer):
        raise NotImplementedError

    def transform_input(self, commands:list, vectorizer):
        raise NotImplementedError

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y, epochs=1, **kwargs):
        self.model.fit(X, y, epochs=epochs, **kwargs)

    def save_weights(self, weights_name: str):
        path = os.path.join(
            "weights",
            f"{weights_name}_input={self.input_shape}_output={self.num_outputs}.h5",
        )
        if os.path.exists(path):
            print("override existing file")
        self.model.save_weights(path, overwrite=True)
        return path

    def load_weights(self, weight_path):
        if self.model:
            self.model.load_weights(weight_path)
        else:
            print("build model first using .build()")


class FFNN(NN):
    def build(self, input_shape: tuple, num_outputs: int):
        model = Sequential()
        model.add(Dense(64, input_shape=input_shape))
        model.add(Dense(32))
        model.add(Dense(num_outputs, activation="sigmoid"))

        model.compile("rmsprop", "binary_crossentropy")
        model.summary()

        self.model = model
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def transform_data(self, data: dict, vectorizer) -> tuple:
        tags = sorted(list(data.keys()))
        X, y = [], []
        pos_to_tag = {}
        for i, tag in enumerate(tags):
            for command in data.get(tag):
                X.append(vectorizer.transform([command]).toarray().reshape(-1,))
                y.append(to_categorical(i, len(tags)))
            pos_to_tag[i] = tag

        return (np.array(X), np.array(y), pos_to_tag)

    def transform_input(self, commands:list, vectrizer):
        return vectrizer.transform(normalize_command(commands)).toarray()


class RNN(NN):
    def build(self, input_shape: tuple, num_outputs: int):
        model = Sequential()
        model.add(LSTM(32,input_shape=input_shape, return_sequences=True, dropout=0.3))
        model.add(Dense(32))
        model.add(Dense(num_outputs, activation="sigmoid"))

        model.compile("rmsprop", "binary_crossentropy")
        model.summary()

        self.model = model
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def transform_data(
        self, data: dict, vectorizer, max_command_length: int = None
    ) -> tuple:
        tags = sorted(list(data.keys()))
        
        X, y = [], []
        pos_to_tag = {}
        for i, tag in enumerate(tags):
            for command in data.get(tag):
                word_list = [
                    vectorizer.transform([word]).toarray().reshape(-1,) for word in word_tokenize(command)]

                X.append(word_list)
                y.append(to_categorical(i, len(tags)))
            pos_to_tag[i] = tag
        
        X = pad_sequences(X, maxlen=max_command_length, padding="post")

        return (np.array(X), np.array(y), pos_to_tag)

    def transform_input(self, commands:list, vectorizer) -> np.ndarray:
        if not self.input_shape:
            raise AttributeError("Build RNN first!")
        
        commands = normalize_command(commands)
        X = []
        depth = self.input_shape[0]
        for command in commands:
            word_list = np.array([
                vectorizer.transform([word]).toarray().reshape(-1,) for word in word_tokenize(command)
            ])

            word_list = word_list[~np.all(word_list == 0, axis=1)]
            # print(word_list)
            X.append(word_list)

        X = pad_sequences(X, maxlen=depth, padding="post")

        return np.array(X)

