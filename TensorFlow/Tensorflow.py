import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df_info = pd.read_csv('/Users/austinloh/Desktop/lending_club_info.csv', index_col = 'LoanStatNew')

def get_info(col_name):
    print(df_info.loc[col_name]['Description'])

get_info('loan_status') #Current status of the loan
df = pd.read_csv('/Users/austinloh/Desktop/lending_club_loan_two.csv')
#df.head()
#df.info()
#df.describe().transpose()

sns.countplot(x='loan_status', data = df) #output_8_1.png
plt.figure(figsize = (10,6))
sns.set_style('whitegrid')
df['loan_amnt'].plot(kind='hist', bins=40) #output_9_1.png

#df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='viridis') #output_11_1.png

get_info('open_acc') #The number of open credit lines in the borrower's credit file.
get_info('pub_rec') #Number of derogatory public records
get_info('pub_rec_bankruptcies') #Number of public record bankruptcies

sns.scatterplot(x='installment', y='loan_amnt', data=df) #output_15_1.png
#df.groupby('loan_status')['loan_amnt'].describe()
#df['grade'].unique()
#sorted(df['sub_grade'].unique())

plt.figure(figsize=(12,5))
sns.countplot(x='sub_grade', data=df, palette='coolwarm', hue='loan_status', 
  order=sorted(df['sub_grade'].unique())) #output_19_1.png

f_g = df[(df['grade']== 'F') | (df['grade'] == 'G')]
plt.figure(figsize=(12,5))
sns.countplot(x='sub_grade', data=df, palette='coolwarm', hue='loan_status',
            order=sorted(f_g['sub_grade'].unique())) #output_21_1.png

df = df.drop('grade', axis=1)
#df.info()
df['repaid']=pd.get_dummies(df['loan_status'], drop_first=True)
#df[['repaid', 'loan_status']]
df.corr()['repaid'].sort_values().drop('repaid').plot(kind='bar') #output_26_1.png

df = df.drop('loan_status', axis=1)
#df.info()
df.isnull().sum()
'''
    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    sub_grade                   0
    emp_title               22927
    emp_length              18301
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    purpose                     0
    title                    1755
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    repaid                      0
    dtype: int64
'''

get_info('emp_title')
get_info('emp_length')
get_info('title')
'''
    The job title supplied by the Borrower when applying for the loan.*
    Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 
    The loan title provided by the borrower
'''

df['emp_title'].nunique() #173105
df['emp_length'].nunique() #11
df['emp_title'].value_counts()
'''
    Teacher                           4389
    Manager                           4250
    Registered Nurse                  1856
    RN                                1846
    Supervisor                        1830
                                      ... 
    Region Human Resources Manager       1
    bowers ambulance                     1
    Director of Veteran Services         1
    Business Outreach Coordinator        1
    Dr. John C. Perry M.D.               1
    Name: emp_title, Length: 173105, dtype: int64
'''

df = df.drop('emp_title', axis=1)
sorted(df['emp_length'].dropna().unique())
'''
    ['1 year',
     '10+ years',
     '2 years',
     '3 years',
     '4 years',
     '5 years',
     '6 years',
     '7 years',
     '8 years',
     '9 years',
     '< 1 year']
'''
len_order = [ '< 1 year',
'1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years'
]

plt.figure(figsize=(10,6))
sns.countplot(x='emp_length', data=df, order=len_order) #output_37_1.png
plt.figure(figsize=(10,6))
sns.countplot(x='emp_length', data=df, order=len_order, hue='repaid') #output_38_1.png

'''
There doesn't seem to be a huge correlation between employment length and repaying loan.
'''

len_repaid = df[df['repaid'] == 1]['emp_length'].value_counts()
len_unpaid = df[df['repaid'] == 0]['emp_length'].value_counts()
len_precentage = len_repaid / (len_repaid + len_unpaid)
len_precentage
'''
    1 year       0.800865
    10+ years    0.815814
    2 years      0.806738
    3 years      0.804769
    4 years      0.807615
    5 years      0.807813
    6 years      0.810806
    7 years      0.805226
    8 years      0.800240
    9 years      0.799530
    < 1 year     0.793128
    Name: emp_length, dtype: float64
'''

len_precentage.plot(kind='bar') #output_44_1.png
    
'''
Since repayment rate is so similar across all employment length, we can decide to ignore this feature.
'''

df = df.drop('emp_length', axis=1)
df.isnull().sum()
'''
    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    sub_grade                   0
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    purpose                     0
    title                    1755
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    repaid                      0
    dtype: int64
'''

get_info('purpose')
get_info('title')
'''
    A category provided by the borrower for the loan request. 
    The loan title provided by the borrower
'''
df['purpose'].value_counts()
'''
    debt_consolidation    234507
    credit_card            83019
    home_improvement       24030
    other                  21185
    major_purchase          8790
    small_business          5701
    car                     4697
    medical                 4196
    moving                  2854
    vacation                2452
    house                   2201
    wedding                 1812
    renewable_energy         329
    educational              257
    Name: purpose, dtype: int64
'''

df['title'].value_counts()
'''
    Debt consolidation                     152472
    Credit card refinancing                 51487
    Home improvement                        15264
    Other                                   12930
    Debt Consolidation                      11608
                                            ...  
    Living on the Edge                          1
    Major Card Consolidation                    1
    Generater Loan                              1
    Ricks Bad dream                             1
    Vacation/ Dental/ DebtConsolidation         1
    Name: title, Length: 48817, dtype: int64
'''

'''
It seems that title is just a subcategory of purpose, so we will drop this column
'''

df = df.drop('title', axis=1)
df.isnull().sum()
'''
    loan_amnt                   0
    term                        0
    int_rate                    0
    installment                 0
    sub_grade                   0
    home_ownership              0
    annual_inc                  0
    verification_status         0
    issue_d                     0
    purpose                     0
    dti                         0
    earliest_cr_line            0
    open_acc                    0
    pub_rec                     0
    revol_bal                   0
    revol_util                276
    total_acc                   0
    initial_list_status         0
    application_type            0
    mort_acc                37795
    pub_rec_bankruptcies      535
    address                     0
    repaid                      0
    dtype: int64
'''

get_info('mort_acc') #Number of mortgage accounts.
df['mort_acc'].value_counts()
'''
    0.0     139777
    1.0      60416
    2.0      49948
    3.0      38049
    4.0      27887
    5.0      18194
    6.0      11069
    7.0       6052
    8.0       3121
    9.0       1656
    10.0       865
    11.0       479
    12.0       264
    13.0       146
    14.0       107
    15.0        61
    16.0        37
    17.0        22
    18.0        18
    19.0        15
    20.0        13
    24.0        10
    22.0         7
    21.0         4
    25.0         4
    27.0         3
    23.0         2
    32.0         2
    26.0         2
    31.0         2
    30.0         1
    28.0         1
    34.0         1
    Name: mort_acc, dtype: int64
'''

df.corr()['mort_acc'].sort_values()
'''
    int_rate               -0.082583
    dti                    -0.025439
    revol_util              0.007514
    pub_rec                 0.011552
    pub_rec_bankruptcies    0.027239
    repaid                  0.073111
    open_acc                0.109205
    installment             0.193694
    revol_bal               0.194925
    loan_amnt               0.222315
    annual_inc              0.236320
    total_acc               0.381072
    mort_acc                1.000000
    Name: mort_acc, dtype: float64
'''

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
def fill(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill(x['total_acc'], x['mort_acc']), axis=1)
df.isnull().sum()
'''
    loan_amnt                 0
    term                      0
    int_rate                  0
    installment               0
    sub_grade                 0
    home_ownership            0
    annual_inc                0
    verification_status       0
    issue_d                   0
    purpose                   0
    dti                       0
    earliest_cr_line          0
    open_acc                  0
    pub_rec                   0
    revol_bal                 0
    revol_util              276
    total_acc                 0
    initial_list_status       0
    application_type          0
    mort_acc                  0
    pub_rec_bankruptcies    535
    address                   0
    repaid                    0
    dtype: int64
'''

df.corr()['pub_rec_bankruptcies'].sort_values()
'''
    revol_bal              -0.124532
    loan_amnt              -0.106539
    installment            -0.098628
    revol_util             -0.086751
    annual_inc             -0.050162
    open_acc               -0.027732
    dti                    -0.014558
    repaid                 -0.009383
    mort_acc                0.029276
    total_acc               0.042035
    int_rate                0.057450
    pub_rec                 0.699408
    pub_rec_bankruptcies    1.000000
    Name: pub_rec_bankruptcies, dtype: float64
'''

pub_rec_avg = df.groupby('pub_rec').mean()['pub_rec_bankruptcies']
def fillpublic(pub_rec, pub_rec_bankruptcies):
    if np.isnan(pub_rec_bankruptcies):
        return pub_rec_avg[pub_rec]
    else:
        return pub_rec_bankruptcies

df['pub_rec_bankruptcies'] = df.apply(lambda x: fillpublic(x['pub_rec'], x['pub_rec_bankruptcies']), axis=1)
df.isnull().sum()
'''
    loan_amnt                 0
    term                      0
    int_rate                  0
    installment               0
    sub_grade                 0
    home_ownership            0
    annual_inc                0
    verification_status       0
    issue_d                   0
    purpose                   0
    dti                       0
    earliest_cr_line          0
    open_acc                  0
    pub_rec                   0
    revol_bal                 0
    revol_util              276
    total_acc                 0
    initial_list_status       0
    application_type          0
    mort_acc                  0
    pub_rec_bankruptcies      0
    address                   0
    repaid                    0
    dtype: int64
'''

df.corr()['revol_util'].sort_values()
'''
    open_acc               -0.131420
    total_acc              -0.104273
    pub_rec_bankruptcies   -0.086387
    repaid                 -0.082373
    pub_rec                -0.075910
    mort_acc                0.005821
    annual_inc              0.027871
    dti                     0.088375
    loan_amnt               0.099911
    installment             0.123915
    revol_bal               0.226346
    int_rate                0.293659
    revol_util              1.000000
    Name: revol_util, dtype: float64
'''

100* df.isnull().sum()['revol_util']/len(df) #0.06969169002348306


'''
Since revol_util does not have a significant relationship with other features, and the number of 
missing data in this column account for only 0.069% of data, we will drop the rows with missing data
'''

df = df.dropna()
df.isnull().sum()
'''
    loan_amnt               0
    term                    0
    int_rate                0
    installment             0
    sub_grade               0
    home_ownership          0
    annual_inc              0
    verification_status     0
    issue_d                 0
    purpose                 0
    dti                     0
    earliest_cr_line        0
    open_acc                0
    pub_rec                 0
    revol_bal               0
    revol_util              0
    total_acc               0
    initial_list_status     0
    application_type        0
    mort_acc                0
    pub_rec_bankruptcies    0
    address                 0
    repaid                  0
    dtype: int64
'''

df.select_dtypes('object').columns
'''
    Index(['term', 'sub_grade', 'home_ownership', 'verification_status', 'issue_d',
           'purpose', 'earliest_cr_line', 'initial_list_status',
           'application_type', 'address'],
          dtype='object')
'''

df['term'].value_counts()
'''
     36 months    301782
     60 months     93972
    Name: term, dtype: int64
'''
df['term'] = df['term'].apply(lambda x: int(x[:3]))
df['term']
'''
    0         36
    1         36
    2         36
    3         36
    4         60
              ..
    396025    60
    396026    36
    396027    36
    396028    60
    396029    36
    Name: term, Length: 395754, dtype: int64
'''
#df.transpose()
sub_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
df = pd.concat([df, sub_dummies],axis=1 )
df.drop('sub_grade', axis=1, inplace=True)
#df

df.select_dtypes('object').columns
'''
    Index(['term', 'home_ownership', 'verification_status', 'issue_d', 'purpose',
           'earliest_cr_line', 'initial_list_status', 'application_type',
           'address'],
          dtype='object')
'''

dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)
df.select_dtypes('object').columns
'''
    Index(['home_ownership', 'issue_d', 'earliest_cr_line', 'address'], dtype='object')
'''

df['home_ownership'].value_counts()
'''
    MORTGAGE    198219
    RENT        159677
    OWN          37714
    OTHER          110
    NONE            31
    ANY              3
    Name: home_ownership, dtype: int64
'''

df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
df['home_ownership'].value_counts()
'''
    MORTGAGE    198219
    RENT        159677
    OWN          37714
    OTHER          144
    Name: home_ownership, dtype: int64
'''

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)
df.select_dtypes('object').columns
'''
    Index(['issue_d', 'earliest_cr_line', 'address'], dtype='object')
'''

df['address']
'''
    0              0174 Michelle Gateway\nMendozaberg, OK 22690
    1           1076 Carney Fort Apt. 347\nLoganmouth, SD 05113
    2           87025 Mark Dale Apt. 269\nNew Sabrina, WV 05113
    3                     823 Reid Ford\nDelacruzside, MA 00813
    4                      679 Luna Roads\nGreggshire, VA 11650
                                    ...                        
    396025       12951 Williams Crossing\nJohnnyville, DC 30723
    396026    0114 Fowler Field Suite 028\nRachelborough, LA...
    396027     953 Matthew Points Suite 414\nReedfort, NY 70466
    396028    7843 Blake Freeway Apt. 229\nNew Michael, FL 2...
    396029          787 Michelle Causeway\nBriannaton, AR 48052
    Name: address, Length: 395754, dtype: object
'''

df['zip_code'] = df['address'].apply(lambda x: x[-5:])
df['zip_code'].value_counts()
'''
    70466    56943
    30723    56496
    22690    56494
    48052    55884
    00813    45793
    29597    45455
    05113    45363
    11650    11217
    93700    11137
    86630    10972
    Name: zip_code, dtype: int64
'''

dumb = pd.get_dummies('zip_code', drop_first=True)
df = pd.concat([df.drop(['zip_code', 'address'], axis=1), dumb], axis=1)
df.columns
'''
    Index(['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'issue_d',
           'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
           'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'repaid',
           'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3',
           'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5',
           'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5',
           'verification_status_Source Verified', 'verification_status_Verified',
           'application_type_INDIVIDUAL', 'application_type_JOINT',
           'initial_list_status_w', 'purpose_credit_card',
           'purpose_debt_consolidation', 'purpose_educational',
           'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
           'purpose_medical', 'purpose_moving', 'purpose_other',
           'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding', 'OTHER', 'OWN', 'RENT'],
          dtype='object')
'''
df = df.drop('issue_d', axis=1)
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x: int(x[-4:]))
df.drop('earliest_cr_line', axis=1, inplace=True)
df.select_dtypes('object').columns
'''
    Index([], dtype='object')
'''

from sklearn.model_selection import train_test_split
X = df.drop('repaid',axis=1).values
y = df['repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation
model = Sequential()

model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(19, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )
'''
    Epoch 1/25
    1237/1237 [==============================] - 3s 2ms/step - loss: 0.4760 - val_loss: 0.4580
    Epoch 2/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4633 - val_loss: 0.4575
    Epoch 3/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4603 - val_loss: 0.4574
    Epoch 4/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4593 - val_loss: 0.4557
    Epoch 5/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4579 - val_loss: 0.4558
    Epoch 6/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4569 - val_loss: 0.4551
    Epoch 7/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4563 - val_loss: 0.4548
    Epoch 8/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4563 - val_loss: 0.4550
    Epoch 9/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4558 - val_loss: 0.4541
    Epoch 10/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4556 - val_loss: 0.4549
    Epoch 11/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4551 - val_loss: 0.4540
    Epoch 12/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4547 - val_loss: 0.4544
    Epoch 13/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4546 - val_loss: 0.4543
    Epoch 14/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4544 - val_loss: 0.4539
    Epoch 15/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4543 - val_loss: 0.4535
    Epoch 16/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4539 - val_loss: 0.4535
    Epoch 17/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4537 - val_loss: 0.4533
    Epoch 18/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4534 - val_loss: 0.4537
    Epoch 19/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4535 - val_loss: 0.4532
    Epoch 20/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4531 - val_loss: 0.4529
    Epoch 21/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4525 - val_loss: 0.4527
    Epoch 22/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4525 - val_loss: 0.4532
    Epoch 23/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4524 - val_loss: 0.4526
    Epoch 24/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4521 - val_loss: 0.4524
    Epoch 25/25
    1237/1237 [==============================] - 2s 2ms/step - loss: 0.4521 - val_loss: 0.4521
'''

from tensorflow.keras.models import load_model
model.save('tensorflow.h5')
losses = pd.DataFrame(model.history.history)
losses.plot() #output_108_1.png

from sklearn.metrics import classification_report,confusion_matrix
pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
'''
    [[  306 15191]
     [  226 63428]]
                  precision    recall  f1-score   support
    
               0       0.58      0.02      0.04     15497
               1       0.81      1.00      0.89     63654
    
        accuracy                           0.81     79151
       macro avg       0.69      0.51      0.46     79151
    weighted avg       0.76      0.81      0.72     79151
'''

'''
From the statistics, we can see that the model is better in predicting whether a loan is repaid. 
This may be due to the data being skewed towards loan being repaid, as there is around 4 times more loan 
being repaid than unpaid. However, the model achieve a perfect score of 1.00 for recall for class 1, 
meaning it can perfectly predict when a loan is repaid and will not classify wrongly as loan would not be 
repaid.
'''
