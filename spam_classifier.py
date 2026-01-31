import pandas as pd

data = pd.read_csv("spam.csv", sep="\t", header=None)
data.columns = ["label", "message"]

print(data.head())
X = data["message"]
y = data["label"]

print(X.head())
print(y.head())
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

print(X_train.shape)
print(X_test.shape)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed")
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
msg = ["Hey are we meeting today?"]
msg_vec = vectorizer.transform(msg)
result = model.predict(msg_vec)

print("Prediction:", result)

