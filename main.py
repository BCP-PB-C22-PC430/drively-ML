from flask import Flask, render_template, Response, request
import json
import numpy as np
import cv2

from process_image import process_image

# from predict import predict

app = Flask(__name__)

@app.route('/')
def index():
    return "Hi :*"

@app.route('/predict', methods=['GET','POST'])
def predict_images():

    data = request.files.get("file")
    if data == None:
        return 'Got Nothing'
    else:
        prediction = process_image(data)
        

    return json.dumps(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)