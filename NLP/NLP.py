import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('yelp.csv')
#df.head()
#df.describe()
df['text length'] = df['text'].apply(len)
#df

g = sns.FacetGrid(data = df, col = 'stars')
g.map(plt.hist, 'text length') #output_7_1.png
sns.set_style('whitegrid')
sns.countplot(x='stars', data=df) #output_8_1.png

stars = df.groupby('stars').mean()
#stars
plt.figure(figsize=(10,6))
sns.heatmap(data = stars.corr(), annot=True, cmap = 'coolwarm') #output_11_1.png

df_15 = df[(df['stars'] == 1) | (df['stars'] == 5)]
#df_15
X = df_15['text']
y = df_15['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred = mnb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
'''
    [[151  69]
     [ 27 979]]
                  precision    recall  f1-score   support
    
               1       0.85      0.69      0.76       220
               5       0.93      0.97      0.95      1006
    
        accuracy                           0.92      1226
       macro avg       0.89      0.83      0.86      1226
    weighted avg       0.92      0.92      0.92      1226
'''

'''
The model has already attain quite a high precision, recall and f1-score of 0.92
'''

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

X = df_15['text']
y = df_15['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
'''
    [[ 31 167]
     [147 881]]
                  precision    recall  f1-score   support
    
               1       0.17      0.16      0.16       198
               5       0.84      0.86      0.85      1028
    
        accuracy                           0.74      1226
       macro avg       0.51      0.51      0.51      1226
    weighted avg       0.73      0.74      0.74      1226
'''

'''
It seems that with tfidf, the metrics of the model worsened instead.
'''
