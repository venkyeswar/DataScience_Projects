import streamlit as st
import pickle
import pandas as pd
loaded_model = pickle.load(open("model.sav","rb"))
loaded_scaler = pickle.load(open("scaler.sav","rb"))

st.markdown("<center><h1 style='color:#522258'>Logistic Regression</h1></center>",unsafe_allow_html=True)
st.markdown("<center><h2 style='color:#8C3061'>Titanic Survival Prediction</h2></center>",unsafe_allow_html=True)
st.sidebar.header("User Input Parameters")

def user_inputs():
    Age = st.sidebar.number_input("Age",max_value=90,step=1,min_value=0)
    SibSp = st.sidebar.number_input("No.of Siblings (SibSp)",step=1,value=0,min_value=0)
    Parch = st.sidebar.number_input("No.of parent childs(Parch)",step=1,value=0,min_value=0)
    Pclass = st.sidebar.selectbox("Pclass",("1","2","3"))
    Sex = st.sidebar.selectbox("Sex- male(1) female(0)",("1","0"))
    Embarked = st.sidebar.selectbox("Embarked",("Q","S","C"))
    
   
    Pclass_2 = 1 if Pclass == "2"  else 0
    Pclass_3 = 1 if Pclass == "3" else 0

    Sex = int(Sex)

    Embarked_Q = 1 if Embarked == "Q" else 0
    Embarked_S = 1 if Embarked == "S" else 0
  

    data = {
        "Age":Age,
        "SibSp":SibSp,
        "Parch":Parch,
        "Pclass_2":Pclass_2,
        "Pclass_3":Pclass_3,
        "Sex":Sex,
        "Embarked_Q":Embarked_Q,
        "Embarked_S":Embarked_S,
        
       
    }

    features = pd.DataFrame([data])
    return features



inputs =user_inputs()

def Prediction(inputs):
    inputs = inputs
    scaled_df = loaded_scaler.transform(inputs)
    input_df = pd.DataFrame(scaled_df,columns=inputs.columns)
    prediction = loaded_model.predict(input_df)[0].astype(int)

    return prediction




st.subheader("User Input parameters")
st.write(inputs)

result_container = st.container()

def predict():
    prediction = Prediction(inputs)
    if prediction == 0 : 
        predict = "Not Survived"
    else:
        predict = "Survived"

    with result_container:
        st.subheader("Prediction")
        st.write(predict)

if st.button("Predict"):
    predict()
