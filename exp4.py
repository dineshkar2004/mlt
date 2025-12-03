import pandas as pd
import numpy as np

data1 = {
    'ID': [1, 2, 3, 4],
    'Name': ['John', 'Alice', 'Bob', 'Mary'],
    'Age': [25, 30, np.nan, 35]
}

data2 = {
    'ID': [3, 4, 5, 6],
    'Department': ['IT', 'CSE', 'ECE', 'EEE'],
    'Salary': [45000, 50000, 40000, np.nan]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
print("Dataset 1:\n", df1)
print("\nDataset 2:\n", df2)

reshaped = pd.melt(df1, id_vars=['ID'], var_name='Attribute', value_name='Value')
print("\nReshaped Data (Using melt):\n", reshaped)

filtered = df1[df1['Age'] > 25]
print("\nFiltered Data (Age > 25):\n", filtered)

merged = pd.merge(df1, df2, on='ID', how='outer')
print("\nMerged Data (Outer Join):\n", merged)

print("\nChecking for Missing Values:\n", merged.isnull().sum())

merged['Age'] = merged['Age'].fillna(merged['Age'].mean())
merged['Salary'] = merged['Salary'].fillna(0)
merged['Department'] = merged['Department'].fillna('Not Assigned')

print("\nAfter Handling Missing Values:\n", merged)
