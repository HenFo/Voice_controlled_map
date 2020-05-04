from sklearn.feature_extraction.text import CountVectorizer
from classifier import RNN, FFNN
import json
import os
import pandas as pd
from preprocessing import normalize_command
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from keras.callbacks import EarlyStopping

# load data
with open(os.path.join("Data", "commands", "combined_training_data.json"), "rt") as f:
    data = json.load(f)

sw = pd.read_csv(os.path.join("Data", "stopwords.txt"), sep="\n", header=None)

sw = set(sw.values.reshape(-1,))
cv = CountVectorizer(stop_words=sw)

documents = [" ".join(doc) for doc in data.values()]
cv.fit(documents)

neural_net = FFNN()
X,y,t = neural_net.transform_data(data, cv)

input_shape = X.shape[1:]
output_size = y.shape[1]

neural_net.build(input_shape, output_size)

earlyStopping = EarlyStopping(monitor="loss", min_delta=0.005, restore_best_weights=True, patience=50)
neural_net.fit(X, y, epochs=500, batch_size = 16, callbacks=[earlyStopping])

text = ["und nochmal nach oben"]
x = neural_net.transform_input(text, cv)

prediction = np.round(neural_net.predict(x), 1)
print(prediction)
res = np.argwhere(prediction >= 0.1)

for i in res.T[1,:]:
    print(t[i])


# score = f1_score(np.argmax(y, axis=1), np.argmax(prediction, axis=1), average="micro")   

# print(score)