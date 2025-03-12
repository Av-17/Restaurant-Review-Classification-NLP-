import streamlit as st
import pickle 
import nltk
import re
from nltk.corpus import stopwords

model = pickle.load(open("model1.pkl","rb"))
vector = pickle.load(open("vector.pkl","rb"))

custom_stopwords = {
    'don', "don't", 'ain', "aren", "aren't", 'couldn', "couldn't", "didn",
    "didn't", "doesn", "doesn't", 'hadn', "hadn't", "hasn", "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', "mightn", "mightn't", 'mustn', "mustn't",
    "needn", "needn't", "shan", "shan't", 'no', "nor", 'not', "shouldn", 
    "shouldn't", "wasn", "wasn't", "weren", "weren't", 'won', "won't", "wouldn", "wouldn't"
}
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
stop_word = set(stopwords.words("english")) - custom_stopwords





st.markdown("""
    <style>
        .stApp {
            background-color: #1e1e1e; /* Light Blue */
            padding: 20px;
            color: white !important;
        }
        .title {
            text-align: center;
            font-size: 32px;
            color: white !important;
        }
        .stTextInput > div > div > input {
            border: 2px solid #0078D7;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px;
        }
        p {
            color: white !important;
            font-size: 16px;
        }

        .stButton > button {
            background-color: #0078D7;
            color: white;
            font-size: 18px;
            padding: 8px 15px;
            border-radius: 8px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="title">🍽️ Restaurant Review Sentiment Analysis</h1>', unsafe_allow_html=True)



input_msg = st.text_input("Enter Your Review 🍕")

if st.button("🔍 Predict Sentiment"):
    def clean_text(text):
        text = text.lower()
        text = re.sub("[^a-zA-Z]", " ", text)
        text = nltk.word_tokenize(text)
        text = [lemma.lemmatize(word,pos="v") for word in text if word not in stop_word]
        text = " ".join(text)
        return text

    clean_sms = clean_text(input_msg)

    transformed_sms = vector.transform([clean_sms])
    result = model.predict(transformed_sms)[0]

    if result == 1:
        st.markdown('<h2 style="color: green; text-align: center;">😊 Positive Review</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color: red; text-align: center;">😞 Negative Review</h2>', unsafe_allow_html=True)




