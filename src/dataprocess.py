import cv2
import matplotlib as plt
import shutil
import os
import numpy as np

# Sample Method that displays an imported image via PIL.Image
def showImage( fpath ):
    image = cv2.imread( fpath, cv2.IMREAD_GRAYSCALE )
    image_array = np.array( image )

    print( image_array.shape )
    plt.imshow( image_array )
    plt.axis( 'off' )
    plt.show()


# Input: FPath to project repo. 
# Output: Changes bulk picture dump into a partitioned collection of 10
#         9 Training Batches + 1 Test Batch
def partitionImages( fpath ):
    IMAGE_DIRECTORY = fpath + '/captcha_images/'
    BATCH_DIRECTORY = fpath + '/images/'
    current_folder = 0    #Counter for batch fpath creation
    folder_names = ['batch_1', 'batch_2', 'batch_3', 
                    'batch_4', 'batch_5', 'batch_6', 
                    'batch_7', 'batch_8', 'batch_9', 
                    'test_batch']
    image_counter = 0     #Counter for making sure 1000 images are in each batch
    BATCH_CAPACITY = 1000

    # Extract the names of each image
    file_names = os.listdir( IMAGE_DIRECTORY )

    # Go through the list of images and sort them into 10 batches
    # 9 Training + 1 Test
    for file_name in file_names:
        # Whenever the current batch is full, reset the counter and iterate the batch index
        if image_counter == BATCH_CAPACITY:
            image_counter = 0
            current_folder += 1

        source_file = IMAGE_DIRECTORY + file_name
        destination_file = BATCH_DIRECTORY + folder_names[ current_folder ]
        
        shutil.move( source_file, destination_file )
        image_counter += 1

# Input: Fpath to Project Repo
# Output: [ ( training-image-matrices, training-labels ), ( testing-image-matrices, testing-labels ) ]
#         2-Tuples of 2-Tuples ((2-Tuple), (2-Tuple))
def loadData(fpath, CHARACTERS):
    counter = 0

    x_train = np.zeros((9000, 24, 72, 1))
    y_train = np.zeros((4, 9000, len(CHARACTERS)))
    x_test = np.zeros((955, 24, 72, 1))
    y_test = np.zeros((4, 955, len(CHARACTERS)))

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
            image_data = cv2.imread( BATCH_DIRECTORY + image_name, cv2.IMREAD_GRAYSCALE ) # Reads image in greyscale (24, 72) No Channels
            image_data = np.array( image_data )              # Height X Length X Channel (24, 72). 
            image_data = np.reshape( image_data, (24, 72, 1))

            label_singular = image_name[ :-4 ] # Drop the .png from file name to extract image label

            if len(label_singular) < 5: #Only 4 characters in the label
                image_data = image_data / 255 # Normalize pixel values to be between 0 and 1.

                label = np.zeros((4, 36)) # Segment the target into its individual 4 characters, we will mark which of the 36 possible characters each one is.

                for j, k in enumerate(label_singular):
                # j represents which character in the image label we are referring to
                # k represents the character at the specific index j
                # character_index represents which of the 36 characters is active
                    character_index = CHARACTERS.find(k) # Finds the index of the letter in the constant variable CHARACTER.
                    label[j, character_index] = 1        # Marks the character index at the ith value in the test label with a 1.

            # Add the numpy array and label into their respective lists
            x_train[counter] = image_data
            y_train[:,counter] = label
            counter += 1

    # PROCESSING TEST IMAGES [955 TOTAL]
    # Determine test batch directory
    TEST_BATCH_DIRECTORY = fpath + "images/test_batch/"

    # Load in all test_image file names
    test_image_names = os.listdir( TEST_BATCH_DIRECTORY )
    test_image_names.sort()                                              # Ensures consistent ordering

    # Convert each test_image into a numpy.darray (Height, Length, Channels) and grab label from file name.
    for i, test_image_name in enumerate(test_image_names):
        image_data = cv2.imread( BATCH_DIRECTORY + image_name, cv2.IMREAD_GRAYSCALE ) # Reads image in greyscale (24, 72) No Channels
        image_data = np.array( image_data )              # Height X Length X Channel (24, 72). 
        image_data = np.reshape( image_data, (24, 72, 1)) #Reshapes to be (24, 72, 1)

        label_singular = test_image_name[ :-4 ] # Drop the .png from file name to extract image label

        # Convert the string into a 2-D array specifying which character was found. Helps when constructing the output layer.
        if len(label_singular) < 5: #Only 4 characters in the label
                image_data = image_data / 255 # Normalize pixel values to be between 0 and 1.

                label = np.zeros((4, 36)) # Segment the target into its individual 4 characters, we will mark which of the 36 possible characters each one is.

                for j, k in enumerate(label_singular):
                    character_index = CHARACTERS.find(k) # Finds the index of the letter in the constant variable CHARACTER.
                    label[j, character_index] = 1        # Marks the character index at the ith value in the test label with a 1.
        
        # Add the numpy array and label into their respective lists
        x_test[i] = image_data
        y_test[:,i] = label

    return (x_train, y_train), (x_test, y_test)