import pandas as pd
from sklearn.preprocessing import LabelEncoder

def dataLoading(path):
    data = pd.read_csv(path)
    data["Language"].value_counts()
    X = data["Text"]
    y = data["Language"]
    return X,y

def labelencoding(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y