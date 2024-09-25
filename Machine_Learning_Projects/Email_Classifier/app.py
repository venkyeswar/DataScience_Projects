import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download('wordnet')
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


st.markdown(
    """
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/venkyeswar/DataScience_Projects/blob/main/Email_Classifier/)
    """,
    unsafe_allow_html=True
)
stop_words = stopwords.words("english")

 
st.markdown("<center><h1 style='color:#E26EE5'>Email Classifier</h1></center>",unsafe_allow_html=True)
def clean_text(text):
    text = text.lower()
    text = text.strip()
    text = text.translate(str.maketrans("","",string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [ lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

def predict(email):
    email = clean_text(email)
    email = [email]
    processed_email = vectorizer.transform(email)
    prediction = model.predict(processed_email)
    return prediction[0]

 
email_input = st.text_area("Enter the email text:")

prediction = predict(email_input)
 
if st.button("Classify"):
    if email_input:
        prediction = predict(email_input)
        if prediction == "spam":
            st.write("### The email is **Spam**.")
        else:
            st.write("### The email is **Ham**.")
    else:
        st.write("Please enter the text of the email to classify.")
