import numpy as np
from tensorflow import keras
from nltk.stem.lancaster import LancasterStemmer
import nltk
import pickle
from neurolink.include import __load__
from neurolink.include.__process__ import *


# Build and compile the model
def build_model(input_shape, output_shape):
    model = keras.Sequential([
        keras.layers.Dense(8, input_shape=input_shape, activation='relu'),
        keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train the model
def train_model(model, training, output, epochs, verbose=1):
    model.fit(training, output, epochs=epochs, verbose=verbose)


# Predict the intent
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


# Use the model to predict intents
def chat(message, confidence_threshold, intents_path, model_path, data_path, should_train_model):
    if should_train_model:
        # Load and preprocess the data
        data = __load__.load_data(intents_path)
        words, labels, training, output = preprocess_data(data)

        # Build and compile the model
        input_shape = (len(training[0]),)
        output_shape = len(output[0])
        model = build_model(input_shape, output_shape)

        # Train the model
        epochs = 1000
        train_model(model, training, output, epochs, verbose=0)

        # Save the model
        model.save(model_path)

        # Save words and labels using pickle
        with open(data_path, 'wb') as file:
            pickle.dump((words, labels), file)

    else:
        model = keras.models.load_model(model_path)
        with open(data_path, 'rb') as file:
            words, labels = pickle.load(file)

    response = predict_intent(message, model, words, labels)
    if response:
        return response
    else:
        return "Sorry, I don't understand. Please try again."


class botModel_keras:
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
