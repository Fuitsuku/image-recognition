import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def unpackage_images(filepath):
    label_key = 'labels'
    data_intermediate = unpickle(filepath)
    data_decoded = {}

    for i, j in data_intermediate.items():
        data_decoded[i.decode('utf8')] = j
    data_intermediate = data_decoded

    data = data_intermediate["data"]
    labels = data_intermediate[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32) #Restructures data into RGB, 32 by 32 cells.
                                                  # 10000 x 3072 -> 10000 x 3 x 32 x 32

    return data, labels

def main():
    image_file_path = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/images/data_batch_1'
    (data, labels) = unpackage_images(image_file_path)


    

if __name__ == "__main__":
    main()