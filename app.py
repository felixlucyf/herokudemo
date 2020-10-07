# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:11:51 2020

@author: felix
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #for rendering results on HTML GUI
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if (prediction==1):
        output = 'The student may have symptoms of depression.'
    else:
        output = 'The student does not have symptoms of depression'
        
    return render_template('index.html',prediction_text=output)

if __name__=="__main__":
    app.run(debug=True)