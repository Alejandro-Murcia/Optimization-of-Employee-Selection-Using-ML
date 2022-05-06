# -*- coding: utf-8 -*-

import numpy as np
import pickle
import os
from flask import Flask, request, render_template, send_from_directory


# Load ML model
model = pickle.load(open('SVM.pkl','rb')) 

# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route('/')
def home():
    return render_template('SeleccionPersonal.html')

@app.route('/about')
def stanford_page():
    return render_template('_about.html')

@app.route('/seleccion')
def selec():
    return render_template('SeleccionPersonal.html')

# Icon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    last_feature = features[-1]
    features.pop()
    if((features[2]==1.0 and features[3]==1.0 and features[4]==1.0 and features[5]==1.0) or (features[5]==2.0)):
        features.append(1.0)
    else:
        features.append(0.0)
    if(features[2]==2.0 and features[3]==2.0 and features[4]==2.0 and features[5]==2.0):
        features.append(1.0)
    else:
        features.append(0.0)
    features.append(last_feature)

    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction
    
    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('SeleccionPersonal.html', 
                               result = 'Se recomienda contratar por ser destacado')
    else:
        return render_template('SeleccionPersonal.html', 
                               result = 'No apto para ser contratado')

if __name__ == '__main__':
#Run the application
    app.run()
