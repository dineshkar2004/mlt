import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}

accuracy_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc
    print(f"\n{name} Accuracy: {round(acc * 100, 2)}%")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

plt.figure(figsize=(7, 5))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue','green','red','purple','orange'])
plt.title("Performance Comparison of ML Algorithms on Iris Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Algorithm")
plt.ylim(0.9, 1.05)
for i, v in enumerate(accuracy_scores.values()):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center')
plt.grid(True)
plt.show()
