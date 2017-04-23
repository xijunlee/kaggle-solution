#!/usr/bin/env python
# coding=utf-8

# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("./train.csv")
test_df    = pd.read_csv("./test.csv")

# preview the data
#print titanic_df.head()
#print titanic_df.info()
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'],axis=1)
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).sum()

print titanic_df.index

#print embark_perc.info()

#print embark_perc.head()


#print embark_perc.tail()

# Fare

# only for test_df, since there is a missing "Fare" values
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# get fare for survived & didn't survive passengers 
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]

# get average and std for fare of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

#print fare_not_survived

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

print average_age_titanic,std_age_titanic,count_nan_age_titanic

