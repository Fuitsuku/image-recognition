import cv2
import numpy as np

# Used to evaluate the overall loss over the training and testing sample sets.
# The code was taken from "Captcha-recognition-using-CNN/Captcha_Recognition_Project_Final.ipynb" Created by Manvi Goel
def evaluateOnSamples( model, x_train, y_train, x_test, y_test):
    print("\n--RUNNING EVALUATION MODULE--\n")

    print("Evaluating loss on training set:\n\n")
    predictions = model.evaluate(x_train, [y_train[0], y_train[1], y_train[2], y_train[3]])
    print("Loss: " + str(predictions[0]))

    print("Evaluating loss on testing set:\n\n")
    predictions = model.evaluate(x_test, [y_test[0], y_test[1], y_test[2], y_test[3]])
    print("Loss: " + str(predictions[0]))

# Used to predict a specified captcha
def predictCAPTCHA( fpath, model, CHARACTERS ):
    img = cv2.imread( fpath, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = img / 255.0
    else:
        print("Image not found")
    
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))

    result = np.reshape(res, ( 4, 36 ))
    k_ind = []
    for i in result:
        k_ind.append(np.argmax(i))
    
    captcha_prediction = ''
    for k in k_ind:
        captcha_prediction += CHARACTERS[k]

    return captcha_prediction
