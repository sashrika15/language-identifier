from sklearn.feature_extraction.text import CountVectorizer
import re

def text_process(X): 
  data_list = []
  for text in X:
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        data_list.append(text)
  return data_list

def bag_of_words(data_list): 
  cv = CountVectorizer()
  X = cv.fit_transform(data_list).toarray()
  return X