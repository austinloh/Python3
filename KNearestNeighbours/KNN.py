import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('KNN_Project_Data')
df.head()
sns.pairplot(data = df, hue = 'TARGET CLASS') #output_3_1.png

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df.drop('TARGET CLASS', axis = 1))
df_scaled = pd.DataFrame(scaled, columns = df.columns[:-1])

from sklearn.model_selection import train_test_split
X = df_scaled
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

sns.set_style('whitegrid')
plt.figure(figsize=(10,7))
plt.plot(range(1,40),error_rate, marker = 'o', markerfacecolor = 'red', markersize = 10) #output_13_1.png

knn = KNeighborsClassifier(n_neighbors= 35)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print('With K = 35')
print('\n Confusion Matrix:')
print(confusion_matrix(y_test, pred))
print('\n Classification Report:')
print(classification_report(y_test, pred))
'''
    With K = 35
    
     Confusion Matrix:
    [[105  26]
     [ 22 147]]
    
     Classification Report:
                  precision    recall  f1-score   support
    
               0       0.83      0.80      0.81       131
               1       0.85      0.87      0.86       169
    
        accuracy                           0.84       300
       macro avg       0.84      0.84      0.84       300
    weighted avg       0.84      0.84      0.84       300
'''


#This model can predict the target class to a precision of 0.84.
