#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import xgboost as xgb

# get titanic & test csv files as a DataFrame
train_df = pd.read_csv("./train.csv")
test_df  = pd.read_csv("./test.csv")

train = train_df.drop('PassengerId',axis=1)
test = test_df.drop('PassengerId',axis=1)

# Feature engineering

full_data = [train, test]

#print train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

#print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
#print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
#print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4, labels=['low','medium','high','very high'])
#print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

for dataset in full_data:
    age_avg        = dataset['Age'].mean()
    age_std        = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5,labels=['very low','low','medium','high','very high'])

#print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    '''
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    '''
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']  = 4

# Feature Selection
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp',\
                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
#print train.head(10)

test  = test.drop(drop_elements, axis = 1)

train = train.values
Y_train, X_train = train[:,0],train[:,1:]
test = test.values
X_test = test

dtrain = xgb.DMatrix(X_train,label=Y_train)

# Modeling and Paramater Tuning
# cross-validation
param = {'max_depth':5, 'eta':1, 'silent':1, 'objective':'binary:logistic','n_estimators':300}
num_boost_round = 10
n_fold = 5
res = xgb.cv(param,dtrain,num_boost_round,n_fold,metrics={'error'},seed=0,
            callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                      xgb.callback.early_stop(3)])
print res

# sklearn-style xgboost
'''
xgbClassifier = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05,silent=False).fit(X_train,Y_train)
predictions = xgbClassifier.predict(X_test)
print predictions
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions })
submission.to_csv("submission.csv", index=False)
'''


