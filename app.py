import streamlit as st 
import pickle
#transform_txt function for text preprocessing

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split


ps = PorterStemmer()

def transform_text(text):
    
    text = text.lower() #Converting to lowercase
    text = nltk.word_tokenize(text) #tokenizing the text.
    
    #Removing special characters and retaining alphanumeric words.
    text = [word for word in text if word.isalnum()]
    
    #Removing stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    #Applying stemming 
    text = [ps.stem(word) for word in text]
    
    
    return " ".join(text)   


#Loding the saved vectorizer and naive bayes model.
vect = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#x = tfidf.fit_transform(df['transformed_txt']).toarray()
#y = df['target']
#x_train,y_train,x_test,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)


#streamlit code
st.title("Email Spam Classifier :email:")

input_message = st.text_area("Enter the message")

if st.button('Predict'):
    #preprocessing the input message
    transformed_email = transform_text(input_message)
    
    #vectorize the preprocessed message
    vector_input = vect.transform([transformed_email])
    
    #predict
   # model.fit(x_train,y_train)
    result = model.predict(vector_input)[0]
    
    #Display the result
    if result == 1:
        st.header("SPAM!")
    else:
        st.header("Not Spam")
        
        
                    
            
    