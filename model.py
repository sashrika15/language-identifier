from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

def split(X,y):
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
  return x_train, x_test, y_train, y_test

def model_train(x_train, y_train, x_test):
  model = MultinomialNB()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  filename = 'final_model.sav'
  pickle.dump(model, open(filename, 'wb'))
  return y_pred

