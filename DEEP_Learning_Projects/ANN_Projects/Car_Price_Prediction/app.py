from flask import Flask,render_template,request,redirect
import pandas as pd
import numpy as np
import os
from keras.models import load_model
import joblib
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    
    if request.method == "POST":
        year = int(request.form.get("year"))
        present_price = request.form.get("present_price")
        fuel_type = request.form.get("fuel_type")
        seller_type = request.form.get("seller_type")

        age1 = 2024-year
        present_price1=float(present_price)
        fuel_type1 = 1 if fuel_type == "Petrol" else 0
        seller_type1 = 1 if seller_type == "Individual" else 0

        data = {
            "Present_Price":[present_price1],
            "Seller_Type_Individual":[seller_type1],
            "Fuel_Type_Petrol":[fuel_type1],
            "age":[age1]
        }
        data=pd.DataFrame(data,columns=["Present_Price","Seller_Type_Individual","Fuel_Type_Petrol","age"])
        scaled_df = scaler.transform(data)
        input_data = pd.DataFrame(scaled_df,columns=data.columns)

        prediction = model.predict(input_data)[0]
        prediction = np.round(prediction,2)[0]
        
        return render_template("prediction.html",
                               year=year,
                               age=age1,
                               present_price=present_price,
                               fuel_type=fuel_type,
                               seller_type=seller_type,
                               prediction=prediction)
        


#     return Predicted_Price

if __name__ == "__main__":
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
