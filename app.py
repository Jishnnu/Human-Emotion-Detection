import gradio as gr
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
model = tf.keras.models.load_model('/content/Trained Model/Trained_Model.h5')

# Define the emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Create the image generator for preprocessing
img_gen = ImageDataGenerator(rescale=1./255)

# Define the function to predict emotions
def predict_emotion(file):
    # Load the image or video
    cap = cv2.VideoCapture(file.name)
    if cap.isOpened():
        ret, frame = cap.read()
        # Check if it's an image or video
        if frame is not None:
            # Preprocess the image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            img = img.astype('float32')
            img = img_gen.standardize(img)
            # Predict the emotion
            prediction = model.predict(img)
            label = emotion_labels[np.argmax(prediction)]
        else:
            label = "No frames found in the video"
    else:
        label = "Could not open the file"
    return label

# Create the Gradio interface
input_type = gr.inputs.File(label="Input File")
output_type = gr.outputs.Textbox(label="Predicted Emotion")
title = "Emotion Detection"
description = "Upload an image or video to predict the corresponding emotion"
iface = gr.Interface(fn=predict_emotion, inputs=input_type, outputs=output_type, title=title, description=description)
if __name__ == '__main__':
    iface.launch()