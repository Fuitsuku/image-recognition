import numpy as np
from PIL import Image
import io

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def readImages(image_data):
    for image_bytes in image_data[b'data']:
        # Create a BytesIO object to read image bytes
        image_io = io.BytesIO(image_bytes)
        im = Image.open(image_io)
        print(im.format)
        print(im.size)
        print(im.mode)
        im.show()

def main():
    file_path = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/images/data_batch_1'
    unPickledData = unpickle(file_path)
    print(unPickledData)
    readImages(unPickledData)
    

if __name__ == "__main__":
    main()