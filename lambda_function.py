#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite



interpreter = tflite.Interpreter(model_path='./saved_model/bean-leaf-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299,299))

classes = [
        'angular leaf spot',
        'bean rust',
        'healthy',
    ]



def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index,X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    
    return dict(zip(classes,preds[0]))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



