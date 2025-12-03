import pandas as pd
import numpy as np
import statistics as st

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print("Dataset Loaded Successfully!\n")
print(df.head())

data = df['sepal_length']

mean_val = np.mean(data)
median_val = np.median(data)
mode_val = st.mode(data)
variance_val = np.var(data)
std_val = np.std(data)

print("\nStatistical Computations for 'sepal_length':")
print("Mean:", round(mean_val, 3))
print("Median:", round(median_val, 3))
print("Mode:", mode_val)
print("Variance:", round(variance_val, 3))
print("Standard Deviation:", round(std_val, 3))
