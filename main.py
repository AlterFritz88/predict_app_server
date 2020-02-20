from flask import Flask, jsonify, request
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = load_model('model.h5f')
model._make_predict_function()


@app.route('/api/predict_photo', methods=['POST', "GET"])
def predict_photo():
    payload = json.load(request.files['json'])
    file = request.files['file']
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    list_of_pred = model.predict(image)
    return jsonify({'type': "incorrect tank on photo", "percents": str()})


@app.route('/api/check', methods=['POST', "GET"])
def check():
    payload = json.loads(request.data)
    if payload["check"] == "check":
        return jsonify({'answer': "ok"})