import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
from flask import Flask, render_template, Response

from predict import predict

model = keras.models.load_model('model.h5')

face_cs = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eye_cs = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def process_image(data):

    #read image file string data
    filestr = data.read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_ANYCOLOR)
    
    text_face = "Undetected"
    text_eye = "Undetected"
    
    frame = tf.keras.utils.img_to_array(img)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')

    faces = face_cs.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        prediction = predict(model, face)
        if (prediction == 0) :
            text_face = "Yawn"
        elif (prediction == 1):
            text_face = "No Yawn"

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cs.detectMultiScale(roi_gray, scaleFactor = 1.3, minNeighbors = 6)
        height2 = h/2

        for (x,y,w,h) in eyes:
            if (y < height2):
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (0, 0, 255), 2)
                eye = roi_color[y:y+h, x:x+w]
                prediction = predict(model, eye)
                if (prediction == 2) :
                    text_eye = "Closed"
                elif (prediction == 3):
                    text_eye = "Opened"

    return {"Face": text_face, "Eye": text_eye}