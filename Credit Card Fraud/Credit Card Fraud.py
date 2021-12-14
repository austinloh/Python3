#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('creditcard.csv')


# In[2]:


df.shape


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


#0: Not Fraud; 1: Fraud
df.Class.value_counts()


# Although decision tree, random forest, and naives bayes do not require scaling of data, we shall do so for ease of reusing data points for training and testing of other models – knn, logistic regression and support vector classification.    
# Also making assumption data is follows normal distribution by using StandardScaler

# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[8]:


scaled = scaler.fit_transform(df.drop(['Time', 'Amount', 'Class'], axis = 1))


# In[9]:


df_scaled = pd.DataFrame(scaled, columns = df.columns[1:-2])


# In[10]:


X = pd.concat([df['Time'], df_scaled, df['Amount']], axis = 1)


# In[11]:


X


# In[12]:


X.describe()


# In[28]:


X[['V28']].boxplot(figsize = (12,10))


# Remove outliers in data

# In[27]:


X.drop(X[X['V28'] > 60].index, inplace = True) #remove v3, v8, v19, v27, v28


# In[29]:


y = df['Class'][X.index]


# In[30]:


X.shape


# In[31]:


y.shape


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# In[35]:


X_train.shape


# In[38]:


logr = LogisticRegression(max_iter = 300)
logr.fit(X_train, y_train)


# In[39]:


pred = logr.predict(X_test)


# In[40]:


print('Confusion Matrix:')
print(confusion_matrix(y_test, pred))
print('\n Classification Report:')
print(classification_report(y_test, pred))


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
error_rate = []


# In[42]:


for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[43]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,7))
plt.plot(range(1,10),error_rate, marker = 'o', markerfacecolor = 'red', markersize = 10)


# In[44]:


knn = KNeighborsClassifier(n_neighbors= 2)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)


# In[45]:


print('Confusion Matrix:')
print(confusion_matrix(y_test, pred))
print('\n Classification Report:')
print(classification_report(y_test, pred))


# In[46]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)


# In[47]:


print('Confusion Matrix:')
print(confusion_matrix(y_test, pred))
print('\n Classification Report:')
print(classification_report(y_test, pred))


# In[48]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)


# In[49]:


print('Confusion Matrix:')
print(confusion_matrix(y_test, rf_pred))
print('\n Classification Report:')
print(classification_report(y_test, rf_pred))


# In[50]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)


# In[51]:


print('Confusion Matrix:')
print(confusion_matrix(y_test, pred))
print('\n Classification Report:')
print(classification_report(y_test, pred))


# In[52]:


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 3)


# In[53]:


grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# In[54]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)


# In[55]:


print('Confusion Matrix:')
print(confusion_matrix(y_test, pred))
print('\n Classification Report:')
print(classification_report(y_test, pred))


# As this is a credit card fraud dataset where 0 is not fraud and 1 is fraud, it is assumed that the recall for class 1 and f1-score would be more relevant. Recall (TP/TP+FN) as we want to be able to predict more fraud cases and f1-score which places equal importance on both precision and recall.
# 
# logistic regression: 0.64 0.73  
# knn:                           0.08  0.15                      
# decision tree:                 0.77  0.79    
# random forest:                 0.78  0.85  
# support vector classification: 0.00  0.00 (not enough computation power to perform grid search for hyper-parameter tuning for svm)  
# naives bayes:                  0.62  0.28
# 
# Overall, best performing model would be random forest classifier
