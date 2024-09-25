from flask import Flask,render_template,request,redirect
app = Flask(__name__)
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
import pandas as pd

model = load_model("iris_classification_model.h5")
encoder = joblib.load("encoder.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
      
        sepal_length = request.form.get("sepal_length")
        sepal_width = request.form.get("sepal_width")
        petal_length = request.form.get("petal_length")
        petal_width = request.form.get("petal_width")


        try:
            
            input_data = pd.DataFrame({
                "sepal_length": [float(sepal_length)],
                "sepal_width": [float(sepal_width)],
                "petal_length": [float(petal_length)],
                "petal_width": [float(petal_width)]
            })
            
        
            predictions = model.predict(input_data)
            class_index = np.argmax(predictions, axis=1)
            prediction = encoder.inverse_transform(class_index)
            prediction = prediction[0]
            print(prediction)
            return render_template("prediction.html",
                                   sepal_length=sepal_length,
                                   sepal_width=sepal_width,
                                   petal_length=petal_length,
                                   petal_width = petal_width,
                                   prediction=prediction)
            
        except ValueError as ve:
            print("ValueError:", ve)
            return "Invalid input. Please ensure all fields contain valid numbers.", 400
        except Exception as e:
            print("Error during prediction:", e)
            return "An error occurred during prediction.", 500
    return redirect('/')


@app.route("/prediction")
def prediction():
    return
if __name__ == "__main__":
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
