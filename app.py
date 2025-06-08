import streamlit as st
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Fake News Detector")
st.subheader("Enter a news article below to check whether it is Fake or Real.")
st.markdown("(This project is created by Abhishek Kumar Pandey, Dilkhush Kumar and Himanshu.)")

article = st.text_area("Enter your News Article here", height=300)

if st.button("Click Here to Check"):
    if not article.strip():
        st.warning("Please enter a news article.")
    else:
        vec = vectorizer.transform([article])
        pred = model.predict(vec)[0]
        label = " Real News" if pred == 1 else "Fake News"
        st.success(f"Prediction: {label}")
