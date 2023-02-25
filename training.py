import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json

# Load the intents file
with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# Loop through each intent and tokenize the words
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)
        docs_x.append(tokenized_words)
        docs_y.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Stem and sort the words
words = [stemmer.stem(w.lower()) for w in words if w not in '?']
words = sorted(list(set(words)))
labels = sorted(labels)

# Create training data
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    stemmed_words = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in stemmed_words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Convert to numpy arrays
training = np.array(training)
output = np.array(output)

# Define the model architecture
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

# Train the model
model = tflearn.DNN(net)
hist = model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.h5')

import os
print(os.getcwd())


