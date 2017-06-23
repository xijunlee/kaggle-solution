import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# The competition datafiles are in the directory ../input
# Read competition data files:
train_df = pd.read_csv("../../train.csv")
test_df  = pd.read_csv("../../test.csv")

train = train_df.values
test = test_df.values

train_x, train_y = train[:,1:].astype(np.float32), train[:,0]
test_x = test.astype(np.float32)

def standardize(x):
    mean = x.mean().astype(np.float32)
    std = x.std().astype(np.float32)
    return (x-mean)/std

# Standardizing the original data
train_x = standardize(train_x)
test_x = standardize(test_x)

nb_components = 15*15
# Principle component analysis
pca = decomposition.PCA(n_components = nb_components)
pca.fit(train_x)
train_x = pca.transform(train_x)

clf = svm.SVC(kernel='rbf', C=1)
scores = cross_val_score(clf,train_x,train_y,cv=5)
print scores