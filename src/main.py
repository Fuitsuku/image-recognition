from dataprocess import *
from model import *
import matplotlib.pyplot as plt

REPO_DIRECTORY = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/'

def main():
    train, test = loadData(REPO_DIRECTORY) #Definitely not correct. Will fix later
    x_train, y_train = train
    x_test, y_test = test

    model = createModel( ( 24, 72, 1) )
    model.summary()

    hist = model.fit( x_train,  [ y_train[0], y_train[1], y_train[2], y_train[3] ], batch_size = 50, epochs = 30, validation_split = 0.2)

    for label in ["accuracy"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.show()

if __name__ == "__main__":
    main()