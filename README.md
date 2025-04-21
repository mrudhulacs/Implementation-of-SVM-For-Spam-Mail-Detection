# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import required libraries and read the dataset using pandas.
 
2. Split the data into train and test data. Fit transform the data using CountVectorizer.
 
3. Predict Y using Support Vector Classifier(SVC) and calculate the accuracy.
 
4. Now print the necessary outputs.

   
## Program:



Program to implement the SVM For Spam Mail Detection.

Developed by: CHITTOOR SARAVANA MRUDHULA

RegisterNumber:  212224040056



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


ENCODING

![image](https://github.com/user-attachments/assets/09a518b8-3635-4b77-b489-0e514608bdde)

HEAD()

![image](https://github.com/user-attachments/assets/a0214131-9761-4ce8-ab07-1b5f7feb0c39)

INFO()

![image](https://github.com/user-attachments/assets/1d674cc2-4fb0-468f-950c-34fa47dacf75)

SUM OF NULL VALUES

![image](https://github.com/user-attachments/assets/0956ebeb-2c25-4576-84f3-96a2c5f01cbf)

PREDICTED Y VALUE

![image](https://github.com/user-attachments/assets/f7d1e9e1-3820-469e-8961-0f66286cd1b3)

ACCURACY

![image](https://github.com/user-attachments/assets/f287fc69-be5c-42e7-90a8-39a55f9e226a)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
