import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('College_Data', index_col = 0)
#df
df.info()
'''
    <class 'pandas.core.frame.DataFrame'>
    Index: 777 entries, Abilene Christian University to York College of Pennsylvania
    Data columns (total 18 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Private      777 non-null    object 
     1   Apps         777 non-null    int64  
     2   Accept       777 non-null    int64  
     3   Enroll       777 non-null    int64  
     4   Top10perc    777 non-null    int64  
     5   Top25perc    777 non-null    int64  
     6   F.Undergrad  777 non-null    int64  
     7   P.Undergrad  777 non-null    int64  
     8   Outstate     777 non-null    int64  
     9   Room.Board   777 non-null    int64  
     10  Books        777 non-null    int64  
     11  Personal     777 non-null    int64  
     12  PhD          777 non-null    int64  
     13  Terminal     777 non-null    int64  
     14  S.F.Ratio    777 non-null    float64
     15  perc.alumni  777 non-null    int64  
     16  Expend       777 non-null    int64  
     17  Grad.Rate    777 non-null    int64  
    dtypes: float64(1), int64(16), object(1)
    memory usage: 115.3+ KB
'''
#df.describe()
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.scatterplot(x='Grad.Rate', y='Room.Board', hue='Private', data = df) #output_5_1.png

sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.scatterplot(x='Apps', y='Grad.Rate', hue='Private', data = df) #output_6_1.png

sns.set_style('whitegrid')
g = sns.FacetGrid(data=df, hue='Private', height=6, aspect=3, legend_out=True)
g = g.map(sns.histplot, 'Outstate', alpha=0.7, bins=20) #output_7_0.png

sns.set_style('whitegrid')
g = sns.FacetGrid(data=df, hue='Private', height=6, aspect=3, legend_out=True)
g = g.map(sns.histplot, 'Grad.Rate', alpha=0.7, bins=20) #output_8_0.png

#df[df['Grad.Rate']>100]
#df['Grad.Rate']['Cazenovia College']=100

sns.set_style('whitegrid')
g = sns.FacetGrid(data=df, hue='Private', height=6, aspect=3, legend_out=True)
g = g.map(sns.histplot, 'Grad.Rate', alpha=0.7, bins=20) #output_11_0.png

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(df.drop('Private', axis=1))
km.cluster_centers_
'''
    array([[1.81323468e+03, 1.28716592e+03, 4.91044843e+02, 2.53094170e+01,
            5.34708520e+01, 2.18854858e+03, 5.95458894e+02, 1.03957085e+04,
            4.31136472e+03, 5.41982063e+02, 1.28033632e+03, 7.04424514e+01,
            7.78251121e+01, 1.40997010e+01, 2.31748879e+01, 8.93204634e+03,
            6.50926756e+01],
           [1.03631389e+04, 6.55089815e+03, 2.56972222e+03, 4.14907407e+01,
            7.02037037e+01, 1.30619352e+04, 2.46486111e+03, 1.07191759e+04,
            4.64347222e+03, 5.95212963e+02, 1.71420370e+03, 8.63981481e+01,
            9.13333333e+01, 1.40277778e+01, 2.00740741e+01, 1.41705000e+04,
            6.75925926e+01]])
'''
#km.labels_

def converter(cluster):
    if cluster == 'Yes':
        return 0
    else:
        return 1

df['cluster'] = df['Private'].apply(converter)
#df

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['cluster'], km.labels_))
print(classification_report(df['cluster'], km.labels_))
'''
    [[531  34]
     [138  74]]
                  precision    recall  f1-score   support
    
               0       0.79      0.94      0.86       565
               1       0.69      0.35      0.46       212
    
        accuracy                           0.78       777
       macro avg       0.74      0.64      0.66       777
    weighted avg       0.76      0.78      0.75       777
'''

'''
Overall, as this is an unsupervised learning, the accuracy of the model is not that desirable. 
I also had to flip the classes in my converter as the model might have been classifying Private Uni as 0 
instead of 1.
'''
