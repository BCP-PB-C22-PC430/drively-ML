import cv2
import tensorflow
from tensorflow import keras
from flask import Flask, render_template, Response

from predict import predict

model = keras.models.load_model('model.h5')

face_cs = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eye_cs = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def gen():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, (0,height-50), (width,height), (0,0,0), thickness=cv2.FILLED)

        faces = face_cs.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        text_face = " "
        text_eye = " "

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
                
            cv2.putText(frame, text_face, (10, height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, text_eye, (200, height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

        # Showing window
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release capture object
    cap.release()
    cv2.destroyAllWindows()