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

print ('Standardizing the original data ...')
# Standardizing the original data
train_x = standardize(train_x)
test_x = standardize(test_x)

print ('PCA ...')
nb_components = 15*15
# Principle component analysis
pca = decomposition.PCA(n_components = nb_components)
pca.fit(train_x)
train_x = pca.transform(train_x)
print (train_x.shape)

print ('Model fitting ...')
clf = SVC(kernel='rbf', C=1)
#scores = cross_val_score(clf,train_x,train_y,cv=5)
clf.fit(train_x,train_y)
print ('PCA fit on test_x ... ')
pca.fit(test_x)
test_x = pca.transform(test_x)
print ('Predicting ...')
yPreds = clf.predict(test_x)
imageId = [i for i in range(1,len(yPreds)+1)]
submission = pd.DataFrame({'ImageId':imageId,'Label':yPreds})
submission.to_csv('submission_svm.csv',index = False)
print ('Result saved to .csv successfully!')

