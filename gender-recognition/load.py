from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from PIL import Image
import  numpy as np
from io import BytesIO
import json
import requests
CLASS_INDEX = None
import keras

input_image_size=224
class_num=2

model_jsonPath='./gender-recognition/model/model_class.json'


def preprocess_input(x):
    x *= (1./255)
    return x


def decode_predictions(preds, top=5, model_json=""):

    global CLASS_INDEX

    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)
    return results


prediction_results = []

prediction_probabilities = []


url='./gender-recognition/james.png'

# response=requests.get(url).content

image_input = Image.open(url)
image_input = image_input.convert('RGB')
image_input = image_input.resize((input_image_size,input_image_size))
image_input = np.expand_dims(image_input, axis=0)
image_to_predict = image_input.copy()
image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
image_to_predict = preprocess_input(image_to_predict)


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow import keras

def create_model():
    base_model=ResNet50(include_top=True, weights=None,classes=class_num)
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    return model


model=create_model()


# 编译模型
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('./gender-recognition/model/model_ex.h5')

prediction = model.predict(x=image_to_predict)


try:
    predictiondata = decode_predictions(prediction, top=int(class_num), model_json=model_jsonPath)

    for result in predictiondata:
        prediction_results.append(str(result[0]))
        prediction_probabilities.append(result[1] * 100)
except:
    raise ValueError("An error occured! Try again.")


print(prediction_results[0],prediction_probabilities[0])
