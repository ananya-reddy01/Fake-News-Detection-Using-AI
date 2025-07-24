import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/fake_or_real_news.csv")
x = data["text"]
y = data["label"].map({"FAKE": 0, "REAL": 1})

vectorizer = TfidfVectorizer(stop_words='english')
x_vec = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2)
model = LogisticRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, pred))