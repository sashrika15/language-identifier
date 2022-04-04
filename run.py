from cgitb import text
from data_loading import *
from data_preprocessing import *
from model import *
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


path = "data/Language Detection.csv"

X,y = dataLoading(path)
y = labelencoding(y)
data_list = text_process(X)
X = bag_of_words(data_list)

x_train, x_test, y_train, y_test = split(X,y)
y_pred = model_train(x_train, y_train, x_test)
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

def predict(text):
    cv = CountVectorizer()
    x = cv.transform([text]).toarray()
    model = MultinomialNB()
    lang = model.predict(x)
    le = LabelEncoder()
    lang = le.inverse_transform(lang)
    print("The langauge is in",lang[0])

predict("Analytics Vidhya provides a community based knowledge portal for Analytics and Data Science professionals")
predict("Analytics Vidhya fournit un portail de connaissances basé sur la communauté pour les professionnels de l'analyse et de la science des données")

