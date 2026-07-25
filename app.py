import streamlit as st
import tensorflow as tf 
from gensim.models import Word2Vec
import joblib
from utils import clean_text, input_converter

w2v_model = Word2Vec.load("models/w2v_model.model")
best_model = joblib.load("models/best_model.pkl")
ann_model = tf.keras.models.load_model("models/ann_model.keras")

st.title("Fake News Detection")
st.info("""
**Important: How to use this prediction tool**

This app is a machine-learning text classifier trained on political, world, and government news articles from 2016–2018. It detects patterns similar to its training data; it does not verify facts or search the internet.

For more meaningful results:
- Enter complete news-style headlines and article text.
- Use political, world, or government news topics similar to 2016–2018 coverage.
- Avoid short, made-up sentences.
- Always verify important claims using trusted news sources or fact-checking organizations.

Predictions may be unreliable for current events, satire, opinions, very short text, or topics outside the training data.
""")
title=st.text_input("Enter the news title:")
text=st.text_area("Enter the news text:")

if st.button("Detect Fake News"):
    cleaned_title = clean_text(title)
    cleaned_text = clean_text(text)

    input_value=input_converter(cleaned_title, cleaned_text, w2v_model)
    
    y_pred1 = best_model.predict(input_value)
    if y_pred1[0] == 1:
        st.write("The news is likely to be a fake news according to the ML model")
        st.write(f"The confidence level is: {best_model.predict_proba(input_value)[0][1]:.2f}")
    else:
        st.write("The news is likely to be real according to the ML model")
        st.write(f"The confidence level is: {best_model.predict_proba(input_value)[0][0]:.2f}")

    y_pred2 = ann_model.predict(input_value)
    if y_pred2[0][0] > 0.5:
        st.write("The news is likely to be a fake news according to the ANN model")
        st.write(f"The confidence level is: {y_pred2[0][0]:.2f}")
    else:
        st.write("The news is likely to be real according to the ANN model")
        st.write(f"The confidence level is: {1-y_pred2[0][0]:.2f}")


