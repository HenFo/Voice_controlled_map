from sklearn.feature_extraction.text import CountVectorizer
from classifier import RNN, FFNN, Fuzzy_Classifier
import json
import os, sys
import pandas as pd
from preprocessing import normalize_command
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

os.chdir("classifier")
# load data
with open(os.path.join("Data", "commands", "combined_training_data.json"), "rt") as f:
    data = json.load(f)

# sw = pd.read_csv(os.path.join("Data", "stopwords.txt"), sep="\n", header=None)

# sw = set(sw.values.reshape(-1,))
cv = CountVectorizer()

# documents = [" ".join(doc) for doc in data.values()]
# cv.fit(documents)

chars = set()
for commands in data.values():
    for command in commands:
        for char in command:
            chars.add(char)


















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
