import os
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np

import tensorflow as tf
import random
import json

# Get the path to the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the model and words/labels
model_path = os.path.join(dir_path, 'model.h5')
model = tf.keras.models.load_model(model_path)

with open(os.path.join(dir_path, 'intents.json')) as file:
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

# Generate a bag of words for the input
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Start the chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    results = model.predict([bag_of_words(user_input, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

    print(random.choice(responses))
