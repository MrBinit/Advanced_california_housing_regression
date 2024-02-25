import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler objects
model = pickle.load(open('stacking_regressor_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    data = [float(x) for x in request.form.values()]
    
    # Apply scaling to the input data
    scaled_data = scaler.transform(np.array(data).reshape(1, -1))
    
    # Make predictions using the model
    predictions = model.predict(scaled_data)
    
    # Inverse transform the predictions to the original scale
    inverse_predictions = np.exp(predictions)
    
    # Format the predictions as dollars
    formatted_predictions = ['${:,.2f}'.format(pred * 1000) for pred in inverse_predictions]
    
    # Render the template with the prediction text
    return render_template("index.html", prediction_text="Predicted house prices: {}".format(", ".join(formatted_predictions)))

if __name__ == "__main__":
    app.run(debug=True)
