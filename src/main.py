import numpy as np
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt


filepath = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/captcha_images'

def showImage(fpath):
    image = Image.open(fpath)
    image_array = np.array(image)

    print(image_array.shape)
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()

def partitionImages(fpath):
    SRC_DIRECTORY = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/'

    current_folder = 0
    folder_names = ['batch_1', 'batch_2', 'batch_3', 
                    'batch_4', 'batch_5', 'batch_6', 
                    'batch_7', 'batch_8', 'batch_9', 
                    'batch_10']

    counter = 0
    BATCH_CAPACITY = 1000

    file_names = os.listdir(fpath)
    for file_name in file_names:
        if counter == BATCH_CAPACITY:
            counter = 0
            current_folder += 1

        source_file = SRC_DIRECTORY + 'captcha_images/' + file_name
        destination_file = SRC_DIRECTORY + 'images/' + folder_names[current_folder]
        
        shutil.move(source_file, destination_file)
        counter += 1

def saveImageNames(fpath):
    file_names = os.listdir(fpath)
    print(len(file_names))

    with open('image_names.txt', 'w') as txt_file:
        for name in file_names:
            txt_file.write(name + '\n')

def main():
    return

if __name__ == "__main__":
    main()