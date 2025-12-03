import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("Square Root of 25:", math.sqrt(25))
print("Sine of 30 degrees:", math.sin(math.radians(30)))
print("Factorial of 5:", math.factorial(5))

arr = np.array([10, 20, 30, 40, 50])
print("\nNumPy Array:", arr)
print("Mean:", np.mean(arr))
print("Standard Deviation:", np.std(arr))

data = {'Name': ['John', 'Alice', 'Bob'], 'Age': [25, 30, 22], 'Salary': [40000, 50000, 35000]}
df = pd.DataFrame(data)
print("\nPandas DataFrame:\n", df)
print("Average Salary:", df['Salary'].mean())

mode_result = stats.mode(df['Age'], keepdims=True)
print("\nMode of Age column using SciPy:", mode_result.mode[0])

plt.figure(figsize=(5,4))
plt.plot(['John', 'Alice', 'Bob'], df['Salary'], color='green', marker='o')
plt.title("Salary Distribution")
plt.xlabel("Employee")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
model = LinearRegression()
model.fit(X, y)
pred = model.predict(np.array([[6]]))
print("\nPredicted value for X=6:", pred[0])
