# -*- coding: utf-8 -*-
"""
CASE STUDY 1 : EMOTION DETECTION USING CNN

Original file is located at https://colab.research.google.com/drive/1JTlczsua-U_bI-dl0ssM_UuDmC2Gm6z0

CNN MODEL IMPLEMENTATION
We briefly touched upon the libraries that python offers machine learning in chapter 2. Let's take a hands-on approach to deepen our understanding of these concepts. Machine learning's subfield, Emotion Detection, includes evaluating and identifying human emotions from a variety of sources, including text, audio, facial expressions, and physiological signs. Psychology, healthcare, customer service, and marketing are just a few of the industries where emotion detection has applications. On labelled data sets, machine learning algorithms are taught to identify patterns in the input data that correspond to various emotional states. For instance, algorithms for face recognition may be trained to recognise expressions that are indicative of various emotions, such as surprise, rage, grief, or happiness. Here we are particulary interested in exploring the dataset using a deep learning model.

DEEP LEARNING
In order to learn hierarchical representations of data, deep learning entails training artificial neural networks with several layers. The ability of these deep neural networks to autonomously learn and extract complex characteristics from high-dimensional data enables them to achieve state-of-the-art performance on a number of tasks, including speech and picture recognition, natural language processing, and game playing.

DATASET
We are using an image-based dataset that shows 7 different human emotions - Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise.

You can download the dataset from https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
"""

from google.colab import drive
drive.mount('/content/drive')

"""
CONVOLUTIONAL NEURAL NETWORK
Deep neural networks like CNN (Convolutional Neural Networks) are typically employed to analyse visual data. For tasks like image classification, object recognition, and image segmentation, it employs a specific architecture with convolutional layers that can automatically learn features from photos and other forms of multidimensional data. The code in the following cells define the architecture of a Convolutional Neural Network (CNN) model for human emotion detection.
"""


# Importing necessary libraries

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization



# Unzipping the dataset

"""
NOTE : 
The fastest way to import data into Google Colab is to upload the zipped version of the dataset into Google Drive and use the following line of code to unzip and load it into the runtime. "> /dev/null" is used to disable the unzip command's output.
"""
!unzip /content/drive/MyDrive/Emotion_Detection/images.zip > /dev/null



# Setting the paths for our training and validation data:

folder = "/content/images"
train_data_path = os.path.join(folder, "train")
test_data_path = os.path.join(folder, "validation")



# Viewing images in the dataset

"""
The emotion variable's value is shown in a grid of 9 pictures using the following code. Using the load_img() method from the  keras.preprocessing.image module, the pictures are loaded from the directory provided in the folder variable. The imshow() function of the  Matplotlib library is then used to display the pictures. The plt.show() method is used to display the plot. The plot's backdrop is made dark by  using the plt.style.use('dark background') function. The plot's size is adjusted to 12x12 inches using the command plt.figure(figsize = (12, 12)).
"""

emotion = 'happy'
plt.style.use('dark_background')
plt.figure(figsize = (12, 12))
for i in range(1, 10, 1):
    plt.subplot(3, 3, i)
    img = load_img(folder + "/train/" + emotion + "/" +
                   os.listdir(folder + "/train/" + emotion)[i], target_size=(48, 48))
    plt.imshow(img)   
plt.show()



# DATA GENERATORS

# Create data generators for our train and validation datasets

batch_size = 32
img_size = (48, 48)

# The following lines of code define an ImageDataGenerator object for data augmentation during training of a neural network for image classification.
train_datagen = ImageDataGenerator(    
    rescale=1./255,                       # Scales the pixel values of the image to be between 0 and 1.
    rotation_range=20,                    # Randomly rotates the image by a specified number of degrees in the given range.
    zoom_range=0.2,                       # Randomly zooms into the image by a specified factor in the given range.
    width_shift_range=0.2,                # Randomly shifts the image horizontally by a specified fraction of the total image size.
    height_shift_range=0.2,               # Randomly shifts the image vertically  by a specified fraction of the total image size.
    horizontal_flip=True,                 # Determines how the empty space created by the above transformations is filled.
    fill_mode='nearest'                   # Fills it with the nearest pixel value.
)

# The next line of code defines an ImageDataGenerator object for data augmentation during validation of a neural network for image classification.
val_datagen = ImageDataGenerator(rescale=1./255)

""" 
The following lines of code create a data generator for the training and validation datasets, which can be used to load images in batches during model training. The train_datagen object's flow_from_directory method requests the directory path containing the training images, the goal size for the images, the batch size  for loading the images, the colour mode (in this case, grayscale), and the class mode (categorical in this case). When a model is being trained, it returns a generator that can be used to load batches of photos and the labels that go with them. 
"""

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    test_data_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

"""

HYPER-PARAMETER DESCRIPTION

ACTIVATION
relu :The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.

MODEL
sequential : A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

MAXPOOLING: Maximum pooling, or max pooling, is a pooling operation that calculates the maximum, or largest, value in each patch of each feature map. The results are down sampled or pooled feature maps that highlight the most present feature in the patch, not the average presence of the feature in the case of average pooling.

PADDING: The padding parameter of the Keras Conv2D class can take one of two values: 'valid' or 'same'. Setting the value to "valid" parameter means that the input volume is not zero-padded and the spatial dimensions are allowed to reduce via the natural application of convolution.

BATCH NORMALIZATION: Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.

DROPOUT: Dropout is a technique used to prevent a model from overfitting. Dropout works by randomly setting the outgoing edges of hidden units (neurons that make up hidden layers) to 0 at each update of the training phase.

ADAM: Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum. It uses the squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using moving average of the gradient instead of gradient itself like SGD with momentum.

SGD: Stochastic Gradient Descent (SGD) addresses both of these issues by following the negative gradient of the objective after seeing only a single or a few training examples. The use of SGD In the neural network setting is motivated by the high cost of running back propagation over the full training set

RMSprop: RMSprop is a gradient based optimization technique used in training neural networks. This normalization balances the step size (momentum), decreasing the step for large gradients to avoid exploding, and increasing the step for small gradients to avoid vanishing.

"""

# DEFINE CNN MODEL 

# An empty sequential model is created in the first line, to which successive layers are added.
model = Sequential()

""" 
* The next lines add Convolutional layers to the model with increasing depth and reducing spatial dimensions of the feature maps through MaxPooling layers.
* The Rectified Linear Unit (ReLU), which is well known for performing well in image recognition tasks, is the activation function utilized in all convolutional layers.
* When the padding is set to "same," zeros are appended to the input to provide the output the same spatial dimensions as the input.
* Each Conv2D layer is followed by a layer of BatchNormalization to normalize the activations from the preceding layer.
* To minimize overfitting, dropout layers are introduced after each MaxPooling layer.
"""

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_size[0], img_size[1], 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

# Convolutional layer output is transformed into a one-dimensional vector by the Flatten layer, which can then be fed into a fully linked layer.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

""" 
To extract the probability distribution of 7 emotions, 2 dense layers are added at the end, one with 128 units and ReLU activation (mentioned above) and 
the other with 7 units and softmax activation.
"""
model.add(Dense(7, activation='softmax'))

# To print the model's architecture, including the number of parameters in each tier, the summary method is invoked.
model.summary()


""" 
COMPILE AND TRAIN THE MODEL

The model is set up for training using the model.compile() function. In this line of code, we are building the model using the Adam optimizer, a well-liked stochastic gradient descent optimizer. Moreover, categorical cross-entropy, which is frequently employed for multiclass classification issues, is the loss function that we have specified. Finally, we define accuracy as the statistic that will be used to assess the model's performance throughout training.

The model is trained using the model.fit() function. We are fitting the model to our training data by using train_generator as the input data, train_generator.samples // batch_size as the number of steps_per_epoch (the number of batches of samples to use in each epoch), 50 as the number of epochs, val_generator as the validation data, and val_generator.samples // batch_size as the number of validation steps. The model will be tuned during training to reduce category cross-entropy loss and increase the accuracy metric. The history object contains the training history.

"""

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=45,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)


"""
EVALUATING THE MODEL AND PLOTTING A GRAPH TO VISUALIZE THE ACCURACY & LOSS

These lines of code create two graphs, one for the accuracy of a machine learning model during training and validation and the other for the loss during training and validation, using the Python module Matplotlib. Here is a detailed explanation of the code:

"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

# This line displays the values from the training and validation sets over the epochs that were recorded in the history object and correspond to the input parameters.
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

# This line changes the graph's title
plt.title('Training and validation accuracy') 
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

# This line changes the graph's title
plt.title('Training and validation loss') 

# This line adds a legend to the graph
plt.legend() 

# This line displays the graph.
plt.show()

# Save the model
model.save('emotion_detection_model.h5')

"""
OPEN CV
A software library for computer vision and machine learning is called OpenCV (Open Source Computer Vision Library). It provides an array of algorithms and methods that can be applied to a number of computer vision tasks, including processing of images and videos, object detection and recognition, and more.
"""

"""
This code loads a model that has already been trained to recognise facial expressions in a video stream. It captures the video stream using OpenCV, uses a pre-trained face detection model to identify faces in each frame, and then employs a pre-trained CNN model to identify the emotions associated with each detected face. The edited frames are then written to a new video file along with the predicted emotion label that was drawn on the frame. Finally, it closes all open windows and releases all consumed resources.

"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained facial emotion detection model
model = load_model('/content/emotion_detection_model.h5') # This is my custom path, it may be different from your file path

# Define a dictionary to map emotion labels to their names
emotions = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Create a VideoCapture object to capture the video stream
cap = cv2.VideoCapture('/content/drive/MyDrive/Emotion_Detection/Emotions.mp4') # This is my custom path, it may be different from your file path

# Define the face detection model
face_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/Emotion_Detection/haarcascade_frontalface_default.xml') # This is my custom path, it may be different from your file path

# Define the output video codec and frame rate
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the output video writer
out = cv2.VideoWriter('Emotions_Output.mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Loop through each frame in the video stream
while cap.isOpened():
    # Read the next frame from the video stream
    ret, frame = cap.read()
    
    # If there's an error reading the frame, break out of the loop
    if not ret:
        break
        
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame using the face detection model
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # For each face detected, predict the emotion using the trained model
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_image = gray[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (48, 48))
        face_image = np.reshape(face_image, (1, 48, 48, 1))
        
        # Normalize the pixel values to be between 0 and 1
        face_image = face_image / 255.0
        
        # Predict the emotion using the trained model
        emotion_probabilities = model.predict(face_image)[0]
        predicted_emotion = emotions[np.argmax(emotion_probabilities)]
        
        # Draw the predicted emotion label on the frame
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw a rectangle around the face on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Write the frame to the output video
    out.write(frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()