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
    file_path = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/images/data_batch_1'
    unPickledData = unpickle(file_path)
    print(unPickledData)
    

if __name__ == "__main__":
    main()