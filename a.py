import json
from flask import Flask, render_template, request, app, jsonify, url_for, Request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('stacking_regressor_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')


# @app.route('/predict', methods = ['POST'])
# def predict_placement():
#     Overall_quality = request.form.get('OverallQuality')
#     Year_built = request.form.get('YearBuilt')
#     Total_Basement = request.form.get('TotalBasement')
#     Garden_Area = request.form.get('GardenArea')
#     Garage_Area = request.form.get('GarageArea"')

#     #prediction 
#     result = model.predict(np.array([Overall_quality,Year_built, Total_Basement, Garden_Area, Garage_Area]).reshape(1,5))
#     return result

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug= True)

