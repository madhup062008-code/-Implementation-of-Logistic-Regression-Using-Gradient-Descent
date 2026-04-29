# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```/*
1. Load and preprocess the dataset by converting labels to binary values and selecting relevant features, then normalize the features using standard scaling.
2.Initialize model parameters (weights) and define the sigmoid and cost functions for logistic regression.
3.Apply gradient descent iteratively to update the weights by minimizing the cost function.
4.Use the trained model to make predictions and evaluate performance using accuracy, while optionally plotting cost vs iterations.
 */
```

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Placement_Data.csv")

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()

Developed by:MADHU .P 
RegisterNumber:212225040215 
*/
```

## Output:
<img width="802" height="631" alt="WhatsApp Image 2026-04-29 at 9 36 18 AM" src="https://github.com/user-attachments/assets/c2b14d57-f2b2-47d0-95bc-62a47031a866" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

