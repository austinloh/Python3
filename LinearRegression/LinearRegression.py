
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

customers =pd.read_csv('Ecommerce Customers')
customers = customers.drop(['Avatar', 'Email', 'Address'], axis=1)
#customers
customers.info()
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers) #output_5_1.png
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers) #output_6_1.png
sns.pairplot(customers) #output_7_1.png

X=customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y=customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train, y_train)
predictions=lm.predict(X_test)
plt.figure(figsize=(12,8))
sns.displot((y_test - predictions), bins=40) #output_16_2.png

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
'''
    MAE: 7.750073430457381
    MSE: 99.35425484098381
    RMSE: 9.967660449723587
'''

coefficients = pd.DataFrame(lm.coef_, index=X.columns)
coefficients.columns= ['Coefficients']
#coefficients

'''
It seems that a 1 unit increase in time on app will increase 39.2 increase in yearly amount spent which is much
greater than the 0.40 increase on website. Thus, it may be beneficial to continue their efforts on developing the 
app. However, do also consider that length of membership seems to be a much more significant factor in contributing
to the amount spent. By giving more benefits to retain customers membership and encouraging them to shop through the app should increase amount spent. 
'''
