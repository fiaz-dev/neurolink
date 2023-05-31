import os
import pickle
import random
import warnings

import nltk
import numpy as np
import tensorflow as tf
import tflearn
from nltk import LancasterStemmer

from neurolink.include.__load__ import load_data
from neurolink.include.__process__ import preprocess_data

warnings.filterwarnings("ignore")



def load_model(model_path, training, output):
    tf.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.load(model_path)
    return model


def classify_input(sentence, words, labels, model):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [LancasterStemmer().stem(word.lower()) for word in sentence_words]
    input_bag = [0 for _ in range(len(words))]
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                input_bag[i] = 1
    results = model.predict([input_bag])[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    return tag, results


def get_response(intents, intent, user_input):
    for i in intents:
        if i["tag"] == intent:
            result = random.choice(i["responses"])
            break
        else:
            result = "Sorry, I don't understand. Please try again."
    return result


def chat(message, accuracy, intents_path, model_path, data_path, train_model):
    # Load the intents data
    intents_data = load_data(intents_path)

    # Preprocess the data
    words, labels, training, output = preprocess_data(intents_data)

    if train_model:
        # Train the model or load the saved model
        tf.compat.v1.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)
        model = tflearn.DNN(net)
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save(model_path)
        with open(data_path, "wb") as f:
            pickle.dump((words, labels, training, output), f)
    else:
        # Load the saved model
        model = load_model(model_path, training, output)

    # Classify the user input
    tag, results = classify_input(message.lower(), words, labels, model)

    # Generate a response based on the predicted intent
    if results[results.argmax()] > accuracy:
        response = get_response(intents_data["intents"], tag, message.lower())
    else:
        # Generate predicted sentence
        predicted_sentence = predict(message, model, words)
        response = "I'm sorry, I'm not sure I understand what you are asking. Could you please provide me " \
                   "with more details or clarify your question? Predicted Sentence: " + predicted_sentence

    return response


def predict(sentence, model, words):
    input_bag = [0 for _ in range(len(words))]
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [LancasterStemmer().stem(word.lower()) for word in sentence_words]
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                input_bag[i] = 1
    input_data = np.array(input_bag).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    predicted_sentence = " ".join([words[i] for i, val in enumerate(prediction) if val > 0.5])
    return predicted_sentence


class initialize:
    """
    these are the test case for the chat function
intents_path = os.path.abspath("intents.json")
model_path = os.path.abspath("model.tflearn")
data_path = os.path.abspath("data.pickle")

response = chat("Hi", 0.5, intents_path, model_path, data_path, train_model=False)
print(response)
"""

    def __init__(self, intents_path, model_path, data_path, train_model):
        self.intents_path = intents_path
        self.model_path = model_path
        self.data_path = data_path
        self.train_model = train_model

    def chat(self, message, accuracy):
        response = chat(message, accuracy, self.intents_path, self.model_path, self.data_path, self.train_model)
        return response
