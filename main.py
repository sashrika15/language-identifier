import streamlit as st
import pandas as pd
import numpy as np
from run2 import *

st.title('Language Detector')
input_txt = st.text_input('Input Text')
# input_txt = "hello there"

if input_txt:
    st.write(predict(input_txt))
# output_lang = predict(input_txt)


# st.write(output_lang)