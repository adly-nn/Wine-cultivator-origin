from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Model
try:
    model = load_model('wine_model.h5')
    scaler = joblib.load('wine_scaler.pkl')
except:
    print("Error: Run create_wine_model.py first!")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Get 5 inputs from the form
            alcohol = float(request.form['alcohol'])
            flavanoids = float(request.form['flavanoids'])
            color = float(request.form['color'])
            proline = float(request.form['proline'])
            dilution = float(request.form['dilution'])

            # Prepare array
            features = np.array([[alcohol, flavanoids, color, proline, dilution]])
            features_scaled = scaler.transform(features)

            # Predict
            prediction = model.predict(features_scaled)
            
            # Find which class has the highest probability (0, 1, or 2)
            predicted_class = np.argmax(prediction)
            
            # Map number to a readable name
            cultivator_names = ["Cultivator A (Barolo)", "Cultivator B (Grignolino)", "Cultivator C (Barbera)"]
            result_name = cultivator_names[predicted_class]
            
            prediction_text = f"Predicted Origin: {result_name}"

        except Exception as e:
            prediction_text = f"Error: {e}"

    # IMPORTANT: Ensure your HTML file is named 'index_wine.html' inside the templates folder
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)