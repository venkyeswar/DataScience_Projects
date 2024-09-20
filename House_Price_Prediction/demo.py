import pandas as pd
import joblib
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler_new.pkl")
inputs={
    "number_of_bedrooms":[4],
    "number_of_bathrooms":[3],
    "living_area":[3400],
    "waterfront_present":[1],
    "number_of_views":[2],
    "grade_of_the_house":[5],
    "area_of_house_excluding_basement":[3400],
    "Area_of_the_basement":[0],
    "Built_Year":[2002],
    "Lattitude":[53],
    "Longitude":[-114]
}
inputs = pd.DataFrame(inputs)
print(model.predict(inputs))