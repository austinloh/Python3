import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('loan_data.csv')
#df.head()
df.info()
'''
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9578 entries, 0 to 9577
    Data columns (total 14 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   credit.policy      9578 non-null   int64  
     1   purpose            9578 non-null   object 
     2   int.rate           9578 non-null   float64
     3   installment        9578 non-null   float64
     4   log.annual.inc     9578 non-null   float64
     5   dti                9578 non-null   float64
     6   fico               9578 non-null   int64  
     7   days.with.cr.line  9578 non-null   float64
     8   revol.bal          9578 non-null   int64  
     9   revol.util         9578 non-null   float64
     10  inq.last.6mths     9578 non-null   int64  
     11  delinq.2yrs        9578 non-null   int64  
     12  pub.rec            9578 non-null   int64  
     13  not.fully.paid     9578 non-null   int64  
    dtypes: float64(6), int64(7), object(1)
    memory usage: 1.0+ MB
'''

df['not.fully.paid'].value_counts()
'''
    0    8045
    1    1533
    Name: not.fully.paid, dtype: int64
'''

plt.figure(figsize=(12,6))
sns.countplot(x = 'purpose', data = df, hue = 'not.fully.paid') #output_5_1.png
sns.jointplot(x='fico', y = 'int.rate', data = df) #output_6_1.png

plt.figure(figsize=(16,8))
sns.lmplot(x = 'fico', y = 'int.rate', data = df, col = 'not.fully.paid', hue 
        ='credit.policy') #output_7_2.png
    
name_of_col = ['purpose']
final_df = pd.get_dummies(df, columns = name_of_col, drop_first = True)
#final_df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_df.drop(columns = 'not.fully.paid'),
                                                    final_df['not.fully.paid'], test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
'''
    [[2027  392]
     [ 339  116]]
                  precision    recall  f1-score   support
    
               0       0.86      0.84      0.85      2419
               1       0.23      0.25      0.24       455
    
        accuracy                           0.75      2874
       macro avg       0.54      0.55      0.54      2874
    weighted avg       0.76      0.75      0.75      2874
'''

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
'''
    [[2407   12]
     [ 448    7]]
                  precision    recall  f1-score   support
    
               0       0.84      1.00      0.91      2419
               1       0.37      0.02      0.03       455
    
        accuracy                           0.84      2874
       macro avg       0.61      0.51      0.47      2874
    weighted avg       0.77      0.84      0.77      2874
'''

'''
Overall, using the random forest classifier did not significantly improve the results as compard to using the 
decision tree classifier. This may in part be due to the skewed data where there was around 5 times more not 
fully paid classes than fully paid.
'''
