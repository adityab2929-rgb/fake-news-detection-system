import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(sentence):
    # Keep only capital and small letters
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Convert to lowercase
    sentence = sentence.lower()

    # Split sentence into words
    words = sentence.split()

    # Remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Join words back into sentence
    sentence = " ".join(words)

    cleaned_sentence=[]
    cleaned_sentence.append(sentence)

    return cleaned_sentence

def input_converter(cleaned_titles,cleaned_texts,w2v_model):
    
    tokenized_titles = [sentence.split() for sentence in cleaned_titles]
    tokenized_texts = [sentence.split() for sentence in cleaned_texts]

    def average_word2vec(words, model, vector_size=100):
        valid_words = [word for word in words if word in model.wv]

        if len(valid_words) == 0:
            return np.zeros(vector_size)

        return np.mean([model.wv[word] for word in valid_words], axis=0)
    
    title_vectors = [
        average_word2vec(words, w2v_model, 100)
        for words in tokenized_titles
    ]

    text_vectors = [
        average_word2vec(words, w2v_model, 100)
        for words in tokenized_texts
    ]

    combined_vector = np.concatenate([title_vectors, text_vectors], axis=1)

    scaler=joblib.load("models/standard_scaler.pkl")
    scaled_combined_vector=scaler.transform(combined_vector)

    return scaled_combined_vector


