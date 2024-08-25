
# Titanic Survival Predictor
### Using Logestic Regression Model




## Overview
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques(logestic Regression). By analyzing historical passenger data, the project builds a predictive model to estimate the likelihood of survival for each passenger based on various features.

## Project Objectives
-  Analyze Titanic passenger data to understand survival patterns.
- Build and evaluate machine learning models to predict survival chances.
- Deploy a predictive tool that can estimate survival probabilities for new data.

## Dataset
The dataset used in this project is the Titanic dataset, which contains information about passengers on the Titanic. Key features include:

- **PassengerId:** Unique identifier for each passenger.
- **Pclass:** Passenger's class (1st, 2nd, or 3rd).
- **Name:** Name of the passenger.
- **Sex:** Gender of the passenger.
- **Age:** Age of the passenger.
- **SibSp:** Number of siblings or spouses aboard the Titanic.
- **Parch:** Number of parents or children aboard the Titanic.
- **Ticket:** Ticket number.
- **Fare:** Ticket fare.
- **Cabin:** Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
- **Survived:** Whether the passenger survived (0 = No, 1 = Yes).

## Installation:

To run this project, you will need to have the following libraries installed:
- *numpy*
- *pandas*
- "scikit-learn*
- *streamlit* (for the web app interface)
- *pickle* ( for model serialization)

## Project Structure
The project is organized as follows:

- **assignment_7.ipynb:** Jupyter notebook file for exploratory data analysis (EDA) and model building.
- **app.py: **Python file for deploying the model and to create the user interface for easy testing of the model.
- **Logestic Regression.docx:** This is the text file that explains the assignment.
- **Titanic_test.csv/Titanic_train.csv : ** These are the datasets used for the training and testing the model.
- **model.sav:** This is the model build in the assignment_7 file it is saved as .sav file using *pickle* library by using these model we are predicting the target feature.
- **scaler.sav:** This is an .sav file that is used to standardize the input data.
- **requirements.text:** This is a text file that contains all the required libraries.
- **README.md:** It describes about the project.


## Model Development
**1. Data Preparation:**

- Clean and preprocess the dataset.
- Handle missing values, encode categorical features, and normalize numerical features.
**2. Exploratory Data Analysis (EDA):**

- Analyze feature distributions and correlations.
- Visualize survival rates across different features.
**3. Model Training:**

- Split the dataset into training and testing sets.
- Train logestic regression machine learning model.
- Evaluate model performance using metrics like accuracy, precision, recall, and F1 score.

**4. Model Serialization:**

- Save the trained model using `pickle`for later use in the Streamlit app.
## Usage
**1. Run the Streamlit App:**

Navigate the project directory and  run

```bash
  streamlit run app.py
```
This will start a local server and open the Streamlit app in your browser.



## Demo

Use the following link to test the Model: 

https://titanic-survival-prediction-wksz.onrender.com/

**Interact with the App:**

- Open the App using the above provided link.
- Provide the passenger data.
- At present i have used only basic concepts so the accuracy of the model is only 80 %.
- Give the data on your own or by using the dataset in the project.
- click on the `predict` button then the prediction is displayed whether the passenger is "survived" or "not survived".


## Acknowledgements
- **`Streamlit`** for creating interactive web applications.
- **`render`** for hosting the websites.
- **`scikit-learn`** for providing machine learning algorithms.

