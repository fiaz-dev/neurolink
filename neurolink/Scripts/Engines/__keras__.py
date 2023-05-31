import json
import numpy as np
from tensorflow import keras
from nltk.stem.lancaster import LancasterStemmer
import nltk
import os
import pickle


def preprocess_data(data):
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = [LancasterStemmer().stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [LancasterStemmer().stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)
    training = np.array(training)
    output = np.array(output)
    return words, labels, training, output


def build_model(input_shape, output_shape):
    model = keras.Sequential([
        keras.layers.Dense(8, input_shape=input_shape, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, training, output, epochs, verbose=1):
    model.fit(training, output, epochs=epochs, verbose=verbose)


def predict_intent(sentence, model, words, labels):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [LancasterStemmer().stem(word.lower()) for word in sentence_words]
    input_bag = [0 for _ in range(len(words))]
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                input_bag[i] = 1
    input_data = np.array(input_bag).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    intent_index = np.argmax(prediction)
    intent = labels[intent_index]
    return intent


def chat(message, confidence_threshold, intents_path, model_path, data_path, should_train_model):
    if should_train_model:
        # Load and preprocess the data
        with open(intents_path, "r") as file:
            data = json.load(file)
        words, labels, training, output = preprocess_data(data)

        # Build and compile the model
        input_shape = (len(training[0]),)
        output_shape = len(output[0])
        model = build_model(input_shape, output_shape)

        # Train the model
        epochs = 1000
        train_model(model, training, output, epochs, verbose=0)

        # Save the model in the native Keras format
        model.save(model_path + '.keras')

        # Save words and labels using pickle
        with open(data_path, 'wb') as file:
            pickle.dump((words, labels), file)

    else:
        model = keras.models.load_model(model_path + '.keras')
        with open(data_path, 'rb') as file:
            words, labels = pickle.load(file)

    response = predict_intent(message, model, words, labels)
    if response:
        return response
    else:
        return "Sorry, I don't understand. Please try again."


class initialize_keras_model:
    """
    these are the parameters that are required to initialize the chatbot
intents_path = os.path.abspath("../../intents.json")
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "model")
data_path = os.path.join(model_dir, "data.pickle")

initialize = initialize(intents_path, model_path, data_path, should_train_model=True)
response = initialize.keras("Hi", 0.5)
print(response)
    """
    def __init__(self, intents_path, model_path, data_path, should_train_model):
        self.intents_path = intents_path
        self.model_path = model_path
        self.data_path = data_path
        self.should_train_model = should_train_model

    def keras(self, message, confidence_threshold):
        response = chat(
            message,
            confidence_threshold,
            self.intents_path,
            self.model_path,
            self.data_path,
            self.should_train_model
        )
        return response





