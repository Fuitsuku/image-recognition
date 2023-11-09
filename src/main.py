import numpy as np
import tensorflow as tf

from tensorflow.python.keras import datasets, layers, models
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def main():
    image_file_path = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/images/data_batch_1'
    data = unpickle(image_file_path)
    data_decoded = {}

    for i, j in data.items():
        data_decoded[i.decode('UTF-8')] = j

    print(data_decoded)

    

if __name__ == "__main__":
    main()