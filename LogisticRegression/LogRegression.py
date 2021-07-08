```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

advert = pd.read_csv('advertising.csv')
advert.head(10)
#advert.info()
advert.describe().transpose()
sns.jointplot(x='Age', y = 'Area Income', data = advert) #output_5_1.png
sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = advert) #output_6_1.png
sns.pairplot(data = advert, hue = 'Clicked on Ad') #output_7_1.png

from sklearn.model_selection import train_test_split
X = advert[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = advert['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))
'''
    [[146   9]
     [ 21 124]]
    
    
                  precision    recall  f1-score   support
    
               0       0.87      0.94      0.91       155
               1       0.93      0.86      0.89       145
    
        accuracy                           0.90       300
       macro avg       0.90      0.90      0.90       300
    weighted avg       0.90      0.90      0.90       300 
'''


'''
With this model, I have a overall precision of 0.9 and a f1-score of 0.9 with a test size of 300. However since 
identifying class 1 has a precision of 0.93, this model is slightly better at identifying class 1 (those who
clicked on ad) than class 0.
'''
