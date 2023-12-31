# First Image Processing Project

### Motivation
- I am currently taking a course on linear algebra and wanted to see a real-world application of what I am currently learning.
- Neural Networks are fascinating and I want to have more hands-on experiences working with it.
- The better I get at things, the more I will enjoy them.
- I want a future career in ML Engineering and felt that this was the perfect starting point.

### Goals
- Understand how a picture gets converted into usable data for training a Convolutional Neural Network (CNN).
- Learn more about how image processing works from a conceptual standpoint and the libraries used during "it".
- Gain experience with Machine Learning Python Libraries.

### References
* Captcha Image Dataset was taken from Kaggle, User "Ami".
* Link : 'https://www.kaggle.com/datasets/fanbyprinciple/captcha-images/'
* Image Data : 9955 Labeled CAPTCHA Images. Image Dim = 24 x 72.

**Resources I Used When Learning About CNNs**
* 1. "CNNs, Part 1: An Introduction to Convolutional Neural Networks" by Victor Zhou
* Link: 'https://victorzhou.com/blog/intro-to-cnns-part-1/'
* 2. TensorFlow - Keras Source Code
* Link: 'https://github.com/keras-team/keras/tree/master/keras'
* 3. " Complete Guide to Prevent Overfitting in Neural Networks (Part-1) "
* Link: 'https://www.analyticsvidhya.com/blog/2021/06/complete-guide-to-prevent-overfitting-in-neural-networks-part-1/'

**Resources I Used to Learn More About This Specific Project**
* 1. "CAPTCHA Recognition using Convolutional Neural Network"
* Link: 'https://medium.com/@manvi./captcha-recognition-using-convolutional-neural-network-d191ef91330e'

## LINE BREAK LINE BREAK LINE BREAK LINE BREAK

### Journal Entries
[01/04/2024]
- Long hiatus as I have been working on a different venture. I recently converted a personal management tool into a side business, and I have consequently been pouring most of my free time into it.
- Today, I realized that my current implementation is experiencing overfitting. The training accuracy is very high, but it suffers from very low testing accuracy. Textbook overfitting. I spent today reading into what overfitting was, why it happens, and how I can mitigate the amount of overfitting without going as far as underfitting the model.
  - The link to the resource I read was : https://www.analyticsvidhya.com/blog/2021/06/complete-guide-to-prevent-overfitting-in-neural-networks-part-1/
  - Link is also posted in the summary above.

[12/19/2023]
- Fixed the training issue. Turns out my preprocessing method was broken, it was replacing the data over and over again, basically making the training set 1000 images instead of the full 1000. After  fixing it, the model is training past 3% accuracy.
- I am starting to mess with the parameters to reduce the testing set loss and overall accuracy. Really starting to see what people mean by "overfitting". The accuracy for the training set is so high, but so awful for the testing set. The better the accuracy on the training set actually, the worse it gets for the testing set.
- Will continue to mess with the parameters for trainings + the model to see if I can improve the model at all.

[12/17/2023]
- The fitting is running without errors, but it appears that the accuracy is not improving. I believe something about my understanding of Goel's code is incorrect. Will continue to look into it over the next few days.

[12/16/2023]
- Finished implementing Goel's model architecture. This will serve as my baseline as I tweak different parts of his model to see how the final accuracy is impacted.
- I also modified the way I pre-processed the image data as I liked how Goel's model reserves a final output branch for each of the characters within the CAPTCHA. (Switched to using OpenCV instead of Pillow, and segmented the image label into its individual parts)
  - This likely improves the accuracy of the model as we have more areas for training. (training an independent branch for each character index instead of training one branch for the entire 4-length character label). 
  - Additionally, from a design consideration perspective, Goel's model architecture better aligns with the goals of CNNs. 
    - CNNs achieve image recognition via feature-based identification (Find the important parts and ignore the rest). His model identifies each character in the CAPTCHA image, while mine only identified the entire CAPTCHA, slightly diminishing the purpose of using a CNN.
- I will begin to test if my implementation of Goel's model is correct during the next session.

[12/15/2023]
- Began to create the model for the data set. I'm following the article written by Manvi Goel (link posted below and above) as a baseline, and will experiment once I get a working model to understand all of the moving parts.
- Currently, Goel has three convolution layers, 3 Max-Pooling layers, 1 Batch Normalization layer, and 1 Flattening layer. Goel picked 16 filters of 3x3, not entirely sure why, as well as an activation function of RElu for the Conv layers, which again, not sure why yet.
- Will continue to document what I learn as I experiment with the model creation process more.
- As a side note, Goel's article introduced me to python libraries that allow you to visuall represent the model's accuracy over "time". I put time in quotations as its really over Epochs, which although not a unit of time, do act as a form of iteration over time. I will be using this to measure the accuracy and performance of my own model once I begin to deviate from Goel's baseline.

[12/14/2023]
- Found an online resource that is doing something very similar to my project, reading on what their approach was and what steps they decided to take.
   - Link: https://medium.com/@manvi./captcha-recognition-using-convolutional-neural-network-d191ef91330e
- Realized that I may need a way to segment each CAPTCHA into the separate characters. Otherwise it would be highly inaccurate when attempting to figure out the entire CAPTCHA all at once. Will begin to follow the write up linked above on how I can implement that in my project.

[12/11/2023]
- Finished reading Victor Zhou's CNN Part 1 Blog. I learned that CNNs are NNs with added Convolutional Layers that only highlight important features, Pooling Layers that shrink the dimensions of the data without losing important features, and an optional Softmax layer that allows us to measure "loss" (accuracy) by normalizing the output to a decimal value between 0 and 1. 
- I moved the data-processing methods into a new python file for better organization. Codebase is becoming more modular.

[12/10/2023]
- Continued reading. Goal is to understand what a CNN is on a conceptual layer so that as I understand what I am implementing within code.

[12/06/2023]
- Another very long hiatus due to finals week. Only one final remains, starting back up lightly by doing some light reading. I also read through my codebase to determine where I left off and began to reread the CNN intro written by Victor Zhou to dust off my knowledge.

[11/19/2023]
- Converted preProcessBatch() -> loadData()
  - As expected, the formatting of how preProcessBatch returns the data was problematic. I converted it to match load_data() as defined in keras/keras/datasets/cifar10.py
  - Also changed it to consolidate all batches into one large training data set. This will occur during run time. File storage remains the same for portability.
- Learned more about how I can manipulate and interpret numpy.darray shapes.
  - Specifically learned that reshaping can be done via numpy.reshape(arrayToBeShape, (newShapeDimensions), additionalOrderingDetails) 
- Will likely begin CNN layer creation next session.

[11/18/2023]
- Started reading an article on basics of CNNs, written by Victor Zhou.
  - I learned about why CNNs are used over classical NNs when working with image processing
    - (Image dimensions are often too large and NNs perform poorly when image data is not "perfect")
  - Link: https://victorzhou.com/blog/intro-to-cnns-part-1/
- Brief hiatus due to school. Finished writing the preProcessBatch() method that, when given a batch number, will convert each image into a numpy array, and attach the label. It creates a 2-Tuple for each image in the batch, then collects all 2-Tuples into one long list of 1000 length and returns it.
- Will most likely need to change the structure of this method since it doesn't fully match the format of Keras. And i'm sure there is a reason why the developers of Keras decided to structure their data in a certain way.

[11/10/2023]
 - Began working out how I want to process the image data so it's formatted well. Keras is designed to return a 2-Tuple List containing image data and the associated data labels

[11/09/2023]
- Found a CAPTCHA Dataset on Kaggle
  - > Replaced the CIFAR 10 dataset and reduced project scope
  - Link: https://www.kaggle.com/datasets/fanbyprinciple/captcha-images/
- Learned how to load .png data into python using Pillow-Image.open
- Messed around with a russet potato image, showed my girlfriend my progress haha.
- Realized that I should partition the 9955 images into smaller batches like how CIFAR 10 was structured
  - Wrote a script that partitions the 9955 images into 9 batches of 1000, leaving the remaining 955 images as a test_batch.
  - I believe this will come in handy once I begin training my CNN.
 
[11/5-8/2023]
- Started messing around with different image processing libraries.
  - Experimented with Pillow, Tensorflow-Keras, Numpy, and io. Decided to only use pillow and numpy and and use the Keras source code as learning material.
    - I'm reading through Keras Source Code and studying how it loads, processes, and manipulates image data.
      - Link: https://github.com/keras-team/keras/tree/master/keras
 
[11/05/2023]
- Continued to format main method and overall repository structure. Learned that most image recognition python libraries follow the "Images, source code, unit test" structure.

[11/03/2023]
- Still learning the ropes of Github, looked up a few sources to understand how branches work

[11/02/2023]
- Started Project, Created Repo
- Found the CIFAR 10 Dataset -> This was later changed to a CAPTCHA dataset to reduce project scope.
- Created README
- Loaded CIFAR 10 images into this repository
