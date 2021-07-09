import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
#output_0_0.jpg

import seaborn as sns
iris = sns.load_dataset('iris')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

sns.pairplot(iris, hue = 'species') #output_4_1.png
iris.info()
'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   species       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
'''

from sklearn.model_selection import train_test_split
X=iris.drop('species', axis = 1)
y=iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
'''
    [[19  0  0]
     [ 0 15  1]
     [ 0  0 10]]
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        19
      versicolor       1.00      0.94      0.97        16
       virginica       0.91      1.00      0.95        10
    
        accuracy                           0.98        45
       macro avg       0.97      0.98      0.97        45
    weighted avg       0.98      0.98      0.98        45
'''

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 3)
#grid.fit(X_train, y_train)
grid.best_params_
'''
    {'C': 10, 'gamma': 0.1}
'''
pred = grid.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
'''
    [[19  0  0]
     [ 0 15  1]
     [ 0  0 10]]
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        19
      versicolor       1.00      0.94      0.97        16
       virginica       0.91      1.00      0.95        10
    
        accuracy                           0.98        45
       macro avg       0.97      0.98      0.97        45
    weighted avg       0.98      0.98      0.98        45
'''

'''
Overall, there isn't an increase in accuracy of the model as first model was already quite accurate. 
The reason for not being able to increase accuracy may be due to 1 data point which is classified as virginica 
but is actually versicolor.
'''
