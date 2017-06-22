#!/usr/bin/env python
# coding=utf-8


# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
#%matplotlib inline

# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;

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

def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        #print title_search.group(0), title_search.group(1)
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
print type(train['Title'])

#print(pd.crosstab(train['Title'], train['Sex']))


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


#print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,"Master":4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
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
test  = test.drop(drop_elements, axis = 1)

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    #pdb.set_trace()
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train0 = train.values # Creates an array of the train data
x_test0 = test.values # Creats an array of the test data


# The 1st Stacking
# Create 5 objects that represent our 4 models
rf0 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et0 = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada0 = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb0 = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc0 = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et0, x_train0, y_train, x_test0) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf0,x_train0, y_train, x_test0) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada0, x_train0, y_train, x_test0) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb0,x_train0, y_train, x_test0) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc0,x_train0, y_train, x_test0) # Support Vector Classifier
#pdb.set_trace()
print("The 1st Stacking training is complete")

x_train1 = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test1 = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

# The 2nd Stacking
rf1 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et1 = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada1 = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb1 = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc1 = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train1, et_oof_test1 = get_oof(et1, x_train1, y_train, x_test1) # Extra Trees
rf_oof_train1, rf_oof_test1 = get_oof(rf1,x_train1, y_train, x_test1) # Random Forest
ada_oof_train1, ada_oof_test1 = get_oof(ada1, x_train1, y_train, x_test1) # AdaBoost 
gb_oof_train1, gb_oof_test1 = get_oof(gb1,x_train1, y_train, x_test1) # Gradient Boost
svc_oof_train1, svc_oof_test1 = get_oof(svc1,x_train1, y_train, x_test1) # Support Vector Classifier
#pdb.set_trace()
print("The 2nd Stacking training is complete")

x_train2 = np.concatenate(( et_oof_train1, rf_oof_train1, ada_oof_train1, gb_oof_train1, svc_oof_train1), axis=1)
x_test2 = np.concatenate(( et_oof_test1, rf_oof_test1, ada_oof_test1, gb_oof_test1, svc_oof_test1), axis=1)

# The 3rd Stacking

rf2 = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et2 = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada2 = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb2 = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc2 = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train2, et_oof_test2 = get_oof(et2, x_train2, y_train, x_test2) # Extra Trees
rf_oof_train2, rf_oof_test2 = get_oof(rf2,x_train2, y_train, x_test2) # Random Forest
ada_oof_train2, ada_oof_test2 = get_oof(ada2, x_train2, y_train, x_test2) # AdaBoost 
gb_oof_train2, gb_oof_test2 = get_oof(gb2,x_train2, y_train, x_test2) # Gradient Boost
svc_oof_train2, svc_oof_test2 = get_oof(svc2,x_train2, y_train, x_test2) # Support Vector Classifier
#pdb.set_trace()
print("The 3rd Stacking training is complete")

x_train3 = np.concatenate(( et_oof_train2, rf_oof_train2, ada_oof_train2, gb_oof_train2, svc_oof_train2), axis=1)
x_test3 = np.concatenate(( et_oof_test2, rf_oof_test2, ada_oof_test2, gb_oof_test2, svc_oof_test2), axis=1)



# The second level xgboost classifier
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train3, y_train)
predictions = gbm.predict(x_test3)

# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
print 'Stacking results saved to .csv successfully!'


#print train.head(10)

'''
# Modeling and Paramater Tuning

#train = train.values
X_train, Y_train = x_train2, y_train
#test = test.values
X_test = x_test2

dtrain = xgb.DMatrix(X_train,label=Y_train)

maxDepth = 5
eta = 1
nEstimators = 300

# cross-validation
param = {'max_depth':maxDepth, 'eta':eta, 'silent':1, 'objective':'binary:logistic','n_estimators':nEstimators}
num_boost_round = 10
n_fold = 5
res = xgb.cv(param,dtrain,num_boost_round,n_fold,metrics={'error'},seed=0,
            callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                      xgb.callback.early_stop(3)])
print res
'''
'''
# sklearn-style xgboost

xgbClassifier = xgb.XGBClassifier(max_depth=maxDepth,n_estimators=nEstimators,learning_rate=eta,silent=True).fit(X_train,Y_train)
predictions = xgbClassifier.predict(X_test)
#print predictions
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions })
submission.to_csv("submission.csv", index=False)
'''


