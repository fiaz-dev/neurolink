import os

import neurolink

"""
Sample Test Case to test the output and  get a response from the chatbot
"""
intents_path = "intents.json"
model_path = "model.tflearn"
data_path = "data.pickle"
initialize_obj = neurolink.initialize(intents_path, model_path, data_path, train_model=True)
response = initialize_obj.chat("Hi", 0.5,'keras')
print(response)

