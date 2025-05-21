# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 212224240176
RegisterNumber:  THIRUMALAI K
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Encoding:
![image](https://github.com/user-attachments/assets/8dfa1bff-519f-4a4b-a223-79587b8bb356)

Head():

![328234925-8e2c3fec-2fe3-40c3-923a-1a1c3719e734](https://github.com/user-attachments/assets/9cb8ac2e-320c-4f92-8da5-e4e65740a7ea)

Info():

![328235099-b48518c5-c983-44d3-9cc2-14924033aa91](https://github.com/user-attachments/assets/f938eaee-c186-42da-b709-9c4d436a5ebb)

isnull().sum():

![328235367-50754f89-e886-48c3-a285-44b76317b605](https://github.com/user-attachments/assets/91fa8710-ecf8-4bb7-9a01-9996ec1fe400)

Prediction of y:


![328235504-8f3a2d63-9aa6-4da2-95c4-d53b87fde998](https://github.com/user-attachments/assets/bc8c2f58-16bf-440a-a90d-cfe3b4f812f5)


Accuracy:

![328235573-d1dcce16-dc32-4ec2-a042-ce25bee461da](https://github.com/user-attachments/assets/6a53acb6-7d89-4fac-a417-b79329383dc6)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
