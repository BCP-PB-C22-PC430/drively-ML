from flask import Flask, render_template, Response
import json

from process_video import gen

# from predict import predict

app = Flask(__name__)

@app.route('/')
def index():
    return "Hi :*"

@app.route('/video_feed')
def video_feed():
    global cap
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)