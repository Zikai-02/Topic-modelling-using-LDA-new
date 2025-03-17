#Necessary imports
import streamlit as st
import pickle
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from collections import defaultdict
import re
import string


@st.cache_resource
def load_lda_model(): # Loading the trained LDA model from a pickle file.
    with open('lda_model_three.pkl', 'rb') as file:
        lda_model = pickle.load(file)
    return lda_model


@st.cache_resource
def load_id2word():
    return Dictionary.load('id2word_three.dict') # Preprocessesing the input text 

# Preprocessing function for user input with args (text (str): The input text).
def preprocess_text(text):
    text = re.sub(r'\d', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = text.lower()
    
       
    tokens = word_tokenize(text)  # Tokenizing the text
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
     
    lemmatizer = WordNetLemmatizer() 
    tag_map = defaultdict(lambda: 'n')  
    tag_map.update({'J': 'a', 'V': 'v', 'R': 'r'})  
    tokens = [lemmatizer.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(tokens)]

    return tokens


def infer_topic(text, lda_model, id2word, topic_names):
    processed_text = preprocess_text(text) # Preprocess the input text
    bow = id2word.doc2bow(processed_text) # Convert to bag-of-words format
    topics = lda_model.get_document_topics(bow)
    if topics:
        topic_id = max(topics, key=lambda x: x[1])[0]  
        return topic_names.get(topic_id, "Unknown Topic")
    return "No clear topic detected"


# Main function to run the Streamlit app
def main():
    st.title("Topic Detection from Movie Reviews")
    st.write("Enter a movie review, and this app will predict its topic using the trained LDA model.")

    
    lda_model = load_lda_model() 

   
    topic_names = {                       # Defining topic names
        0: "Filmmaking & Audience Engagement",
        1: "Acting & Character Performance",
        2: "Diverse/Eclectic Films",
        3: "General Movie Viewing Experience",
        4: "Television & Musical Entertainment",
        5: "Horror & Monster Films",
        6: "Action & Crime Films",
        7: "Life, Family & Age"
    }

    
    review = st.text_area("Enter your movie review:")
    
    if st.button("Predict Topic"):    # a button to trigger prediction
        if review.strip():
            topic = infer_topic(review, lda_model, id2word, topic_names)
            st.success(f"The review topic is: **{topic}**")  # Display the predicted topic
        else:
            st.error("Please enter a valid review!")

if __name__ == "__main__":
    main()