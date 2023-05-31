import json


def load_data(data_path):
    """
    Load data from a JSON file
    the data should be loaded here with the json path and return it to process the data
    """
    with open(data_path) as file:
        data = json.load(file)
    return data
