#import pandas as pd
#import spacy
#import en_core_web_sm
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')







# Define a function to preprocess the text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Return the preprocessed tokens as a string
    return lemmatized_tokens

def get_topics(review_text=''):

    # Extract the tokens based on specified conditions
    tokens = preprocess_text(review_text)

    # Load and Check the model
    lda_model_loaded = LdaMulticore.load('./LDA_model/lda_model_10topics')
    dictionary = Dictionary.load('./GensimDictionary/dictionary.dict')


    # Predict
    new_doc_bow = dictionary.doc2bow(tokens)
    topics_probabilities = lda_model_loaded.get_document_topics(new_doc_bow)

    return topics_probabilities



def main():
    st.title('Topic Recognition for Restaurants')
   
   # Labels associated with each index
    labels = [
        "Family friendly",
        "Family owned business - Small and Cozy",
        "Desserts Quality",
        "Menu variety",
        "Waiting time (take order, answer questions and bring food)",
        "Food Quality (International food)",
        "Food Quality (Tasteful and generous)",
        "Traditional Location",
        "Fun night Atmosphere",
        "Customer Support, enjoyable atmosphere and Affordable"
    ]

    # Define a box for text input
    review_text = st.text_input("Enter your review text:", "Review text")
    ok = st.button('Get Topics')

    if ok :
        predictions = get_topics(review_text=review_text)

        #for prediction in predictions:
        #    print(f' Topic N°{prediction[0]} has an assigned probability of {prediction[1]:.02f}')


        # Extract probabilities and labels from the data list
        probabilities = [t[1] for t in predictions]
        labels = [labels[t[0]] for t in predictions]

        # PIE PLOT

        # Create a pie plot
        fig, ax = plt.subplots()
        ax.pie(probabilities, labels=labels, autopct='%1.1f%%')
        ax.axis('equal')

        # Display the pie plot using Streamlit
        st.pyplot(fig)

        # BAR PLOT
        # Create a horizontal bar plot
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, probabilities)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')

        # Display the horizontal bar plot using Streamlit
        st.pyplot(fig)

if __name__ == '__main__':
    main()
