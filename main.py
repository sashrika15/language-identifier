import streamlit as st
from run2 import *

opt = st.sidebar.selectbox("",("Home", "Identifier"))

if opt=="Home":
    html_temp = '''
        <div>
        <h2></h2>
        <center><h3>Language Identifier - An NLP Project</h3></center>
        </div>
        <hr>
        This project will involve using the Language Detection dataset for training your machine learning/deep learning algorithm. 
        <br>
        This dataset has two columns: text and language. After performing text preprocessing methods, you can use your preferred algorithm to predict the correct target variable of language for a given text.
        <br>
        '''
    st.markdown(html_temp, unsafe_allow_html=True)
    st.header("")
    
else:
    input_txt = st.text_input('Input Text')
    if st.button("Identify"):
        output = predict(input_txt)
        st.success("The given text has been identified as '{}'".format(output))
