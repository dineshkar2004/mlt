import numpy as np
import pandas as pd
import pickle

data_manual = {
    'Name': ['John', 'Alice', 'Bob'],
    'Age': [25, 30, 22],
    'Salary': [40000, 50000, 35000]
}
print("Manual Dataset Created:\n", data_manual)

np_data = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
print("\nNumPy Dataset:\n", np_data)

np.savetxt("numpy_data.txt", np_data)
print("\nNumPy data saved to 'numpy_data.txt'")

loaded_np = np.loadtxt("numpy_data.txt")
print("\nLoaded NumPy Dataset:\n", loaded_np)

df = pd.DataFrame(data_manual)
print("\nPandas DataFrame:\n", df)

df.to_csv("dataset.csv", index=False)
print("\nDataset saved to 'dataset.csv'")

df_loaded = pd.read_csv("dataset.csv")
print("\nLoaded DataFrame from CSV:\n", df_loaded)

with open('dataset.pkl', 'wb') as f:
    pickle.dump(df, f)

with open('dataset.pkl', 'rb') as f:
    loaded_pickle = pickle.load(f)
print("\nLoaded Dataset from Pickle File:\n", loaded_pickle)
