import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


fpath = '/Users/mitsuakifukuzaki/Desktop/Hub/Programming/Python_Project/Image_Recognition/images/potato.jpg'

def showImage(fpath):
    image = Image.open(fpath)
    image_array = np.array(image)

    plt.imshow(image_array)
    plt.axis('off')
    plt.show()


def main():
    showImage(fpath)
    

if __name__ == "__main__":
    main()