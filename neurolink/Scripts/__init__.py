import os

from neurolink.Scripts.__tflearn__ import chat


class initialize:
    """
    you can pass the data in this format => path of the intents.json, model.tflearn, data.pickle and train_model as
    True or False default is True
    intents_path = os.path.abspath("intents.json")
    model_path = os.path.abspath("model.tflearn")
    data_path = os.path.abspath("data.pickle")
    initialize_obj = initialize(intents_path, model_path, data_path, train_model=True)
    response = initialize_obj.chat("Hi", 0.5)
    print(response)

    """
    def __init__(self, intents_path, model_path, data_path, train_model=True):
        self.intents_path = intents_path
        self.model_path = model_path
        self.data_path = data_path
        self.train_model = train_model

    def chat(self, message, accuracy):
        response = chat(message, accuracy, self.intents_path, self.model_path, self.data_path, self.train_model)
        return response


