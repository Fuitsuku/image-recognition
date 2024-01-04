from dataprocess import *
from model import *
from evaluate import *
import matplotlib.pyplot as plt

REPO_DIRECTORY = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/'
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVQXYZ" # Each of the possible 36 characters are indexed based on their location in this string.

def main():
    (x_train, y_train), (x_test, y_test) = loadData(REPO_DIRECTORY, CHARACTERS) #Definitely not correct. Will fix later

    model = createModel( ( 24, 72, 1) )
    # model.summary() # Prints out high-level explanation of model architecture.

    hist = model.fit( x_train,  [ y_train[0], y_train[1], y_train[2], y_train[3] ], batch_size = 32, epochs = 5, validation_split = 0.4)
    evaluateOnSamples( model, x_train, y_train, x_test, y_test )
    

if __name__ == "__main__":
    main()