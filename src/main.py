import numpy as np
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt


SRC_DIRECTORY = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/'

def showImage( fpath ):
    image = Image.open( fpath )
    image_array = np.array( image )

    print( image_array.shape )
    plt.imshow( image_array )
    plt.axis( 'off' )
    plt.show()


# Was used to create 9 batch folders containing 1000 images each. And a test_batch containing the remaining images.
def partitionImages( fpath ):
    SRC_DIRECTORY = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/'

    current_folder = 0
    folder_names = ['batch_1', 'batch_2', 'batch_3', 
                    'batch_4', 'batch_5', 'batch_6', 
                    'batch_7', 'batch_8', 'batch_9', 
                    'test_batch']

    counter = 0
    BATCH_CAPACITY = 1000

    file_names = os.listdir( fpath )
    for file_name in file_names:
        if counter == BATCH_CAPACITY:
            counter = 0
            current_folder += 1

        source_file = SRC_DIRECTORY + 'captcha_images/' + file_name
        destination_file = SRC_DIRECTORY + 'images/' + folder_names[ current_folder ]
        
        shutil.move( source_file, destination_file )
        counter += 1


# DEPRECIATED -> Going to create labels during run time as we convert image -> Numpy Array
def createLabels( fpath ):  
    folder_names = ['batch_1', 'batch_2', 'batch_3', 
                    'batch_4', 'batch_5', 'batch_6', 
                    'batch_7', 'batch_8', 'batch_9', 
                    'test_batch']

    for folder_name in folder_names:
        file_names = os.listdir( fpath + "images/" + folder_name )
        file_names = sorted( file_names )
        
        os.chdir( fpath + "images/" + folder_name )
        with open( '0image_names.txt', 'w' ) as txt_file:
            for file_name in file_names:
                txt_file.write( file_name[ :-4 ] + '\n' )

# Takes each image from the specified batch and returns a list of 2-Tuples. 
# Each 2-Tuple contains an image from the batch converted to a numpy array and
# its associated label.
def preProcessBatch( fpath, batch_number ):
    BATCH_DIRECTORY = fpath + "images/batch_" + str( batch_number ) + "/"
    batch_data = []

    image_names = os.listdir( BATCH_DIRECTORY ) # Loads all file names from the selected batch folder
    for image_name in image_names:
        data_singular = Image.open( BATCH_DIRECTORY + image_name )
        data_singular = np.array( data_singular )
        label_singular = image_name[ :-4 ]
        batch_data += [ [data_singular, label_singular] ]

    return batch_data                         # A list of lists. Each sublist contains the numpy array and its label

def main():
    preProcessBatch( SRC_DIRECTORY, 1 )

if __name__ == "__main__":
    main()