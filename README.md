# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries

2.Load and check dataset

3.Encode categorical data

4.Split into features and target

5.Train Decision Tree Regressor

6.Predict and evaluate

7.Predict new value

8.Visualize tree

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: AMAN ALAM 
RegisterNumber:  212224240011
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

data = pd.read_csv(io.BytesIO(uploaded['Salary.csv']))
data.head()
print(data.head())          # View first 5 rows
print(data.info())          # Dataset info
print(data.isnull().sum())  # Check for null values
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())  # View updated dataset
x = data[["Position", "Level"]]  # Features
y = data["Salary"]               # Target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)
print("Predicted Salary for [5,6]:", dt.predict([[5, 6]]))
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/e47547a3-be86-4390-9abe-2ba88346f867)

![image](https://github.com/user-attachments/assets/6bc9502f-a05a-49f5-9747-46ae89d3d418)

![image](https://github.com/user-attachments/assets/349d7540-2185-4d54-b0a2-edcc0b78b44e)

![image](https://github.com/user-attachments/assets/e48e2afd-5a05-45d9-abff-4398e829b2d8)

![image](https://github.com/user-attachments/assets/e076be24-afea-4ffc-9b7f-e2440db132af)

![image](https://github.com/user-attachments/assets/50feb13f-0624-4e4b-bef5-6b83a3e3311f)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
