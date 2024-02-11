import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import uuid

ps = PorterStemmer()

# preprocess
def transform_message(mess):
    message = mess.lower()
    message = nltk.word_tokenize(message)
    
    y = []
    for i in message:
        if i.isalnum():          # remove special characters
            y.append(i)
            
    message = y[:]
    y.clear()
    
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    message = y[:]
    y.clear()
    
    for i in message:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('mnb.pkl','rb'))

st.title('Spam Ham Classifier')

msg_placeholder = st.empty()  # Placeholder for the text area
msg = msg_placeholder.text_area("Enter the message :", value='')

if st.button('Predict'):
    transform_msg = transform_message(msg)

    # vectorize
    vec_input = tfidf.transform([transform_msg])

    # predict
    out = model.predict(vec_input)

    # Display
    if out == 1:
        st.header('Spam')
    else:
        st.header('Ham')   

    # Clear original text area
    msg_placeholder.empty()  # Clear the placeholder
    msg_placeholder.text_area("Enter the message :", value='', key=str(uuid.uuid4()))  # Re-display the text area with an empty value and a unique key
