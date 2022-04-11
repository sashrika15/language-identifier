import pandas as pd
import numpy as np
import re         
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter("ignore")

data = pd.read_csv("data/Language Detection.csv")
X = data["Text"]
y = data["Language"]

le = LabelEncoder()
y = le.fit_transform(y)
data_list = []

for text in X:         
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)    
    text = re.sub(r'[[]]', ' ', text)   
    text = text.lower()          
    data_list.append(text)

import pickle
filename = "fmodel.pkl"

# with open(filename, 'wb') as fout:
#     pickle.dump((cv, model), fout)

with open('fmodel', 'rb') as f:
    cv, model = pickle.load(f)

def predict(text):
    x = cv.transform([text]).toarray() 
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The langauge is in",lang[0]) 
    return lang[0]

# predict('People are awesome')