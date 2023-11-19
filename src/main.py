import numpy as np
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt


ROOT_DIRECTORY = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/'

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

    current_folder = 0                                  #Counter for batch fpath creation
    folder_names = ['batch_1', 'batch_2', 'batch_3', 
                    'batch_4', 'batch_5', 'batch_6', 
                    'batch_7', 'batch_8', 'batch_9', 
                    'test_batch']

    image_counter = 0                                         #Counter for making sure 1000 images are in each batch
    BATCH_CAPACITY = 1000

    file_names = os.listdir( fpath )
    for file_name in file_names:
        if image_counter == BATCH_CAPACITY:
            image_counter = 0
            current_folder += 1

        source_file = SRC_DIRECTORY + 'captcha_images/' + file_name
        destination_file = SRC_DIRECTORY + 'images/' + folder_names[ current_folder ]
        
        shutil.move( source_file, destination_file )
        image_counter += 1


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

def loadData(fpath):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # PROCESSING TRAINING IMAGES [9000 TOTAL]
    # Go through each of the 9 test batches and consolidate into 1 large training set. Creates 2-Tuple (numpyArrays, labels) -> (x_train, y_train)
    for batch_number in range(1, 10):
        # Determine current Batch directory
        BATCH_DIRECTORY = fpath + "images/batch_" + str( batch_number ) + "/"

        # Load in all image file names
        image_names = os.listdir( BATCH_DIRECTORY )                      
        image_names.sort()                                                # Ensures consistent ordering

        # Convert each image file into a numpy.darray (Height, Length, Channels) and grab label from file name.
        for image_name in image_names:
            data_singular = Image.open( BATCH_DIRECTORY + image_name )
            data_singular = np.array( data_singular )                     # Height X Length X Channel (24, 72, 3)

            label_singular = image_name[ :-4 ]
            
            # Add the numpy array and label into their respective lists
            x_train.append(data_singular)
            y_train.append(label_singular)

    # PROCESSING TEST IMAGES [955 TOTAL]
    # Determine test batch directory
    TEST_BATCH_DIRECTORY = fpath + "images/test_batch/"

    # Load in all test_image file names
    test_image_names = os.listdir( TEST_BATCH_DIRECTORY )
    test_image_names.sort()                                              # Ensures consistent ordering

    # Convert each test_image into a numpy.darray (Height, Length, Channels) and grab label from file name.
    for test_image_name in test_image_names:
        data_singular = Image.open( TEST_BATCH_DIRECTORY + test_image_name )
        data_singular = np.array( data_singular )                        # Height X Length X Channel (24, 72, 3)

        label_singular = test_image_name[ :-4 ]
        
        # Add the numpy array and label into their respective lists
        x_test.append(data_singular)
        y_test.append(label_singular)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train, (y_train.size, 1))

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = np.reshape(y_test, (y_test.size, 1))

    return (x_train, y_train), (x_test, y_test)

def main():
    loadData(ROOT_DIRECTORY)

if __name__ == "__main__":
    main()