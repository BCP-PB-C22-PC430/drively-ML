import cv2
import numpy as np

labels = ["Yawn", "No_Yawn", "Open", "Closed"]

def preprocess(frame):
    frame = cv2.resize(frame,(128,128))
    frame = np.array(frame, dtype = float)
    frame = np.expand_dims(frame, axis=0)
    frame /= 255.
    return frame.reshape((1, 128, 128, 3))

def predict(model, frame):
    image = preprocess(frame)
    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    # label = labels[index]
    return index