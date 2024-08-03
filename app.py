import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    # Load the pre-trained model
    model = joblib.load('model.joblib')


    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    
    # Input from the user
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    
    if st.button("Predict"):
        # Preprocess the input
        preprocessed_input = preprocessor().transform(pd.Series([userinput]))
        
        # Predict the sentiment
        predicted_sentiment = model.predict(preprocessed_input)[0]
        
        # Determine output based on prediction
        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'
        
        # Display the results
        sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)
        

if __name__ == "__main__":
    run()
