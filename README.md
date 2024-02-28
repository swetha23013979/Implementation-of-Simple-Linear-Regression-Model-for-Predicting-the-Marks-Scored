# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Swetha D
RegisterNumber:  212223040222
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![dataset](https://github.com/swetha23013979/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153823422/a8725c13-0f70-4ff8-a6e4-fc5bd8e7e579)


![head](https://github.com/swetha23013979/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153823422/af69aaab-c437-4230-b8ab-546be4816505)


![tail](https://github.com/swetha23013979/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153823422/e3403eca-b02d-4b2f-b973-c758970d7154)


![xyvalues](https://github.com/swetha23013979/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153823422/958c771d-5ee2-4514-913d-d1be2a80b8c2)


![predict ](https://github.com/swetha23013979/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153823422/1f9b2b9b-efdc-4e6c-a7cd-78ef934ba903)


![values](https://github.com/swetha23013979/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153823422/569157bb-cc80-493a-bc5d-462af81dd1bc)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
