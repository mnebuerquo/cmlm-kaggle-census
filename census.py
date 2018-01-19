import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
from load import load
from fixes import *
from sklearn.tree import DecisionTreeRegressor

train_data = load("data/adult.data")
X_test = feature_xform(load("data/adult.test.X", test=True))

y = get_y(train_data)
X_train = feature_xform(drop_income(train_data))

# this fails in various ways if the columns aren't the same between the two
# data sets. We're going to drop any columns from either which are not in both.
intrain = set(X_train.columns) - set(X_test.columns)
intest  = set(X_test.columns) - set(X_train.columns)
dropcols = list( intrain.union(intest) )
for c in dropcols:
    if c in X_train.columns:
        X_train = X_train.drop(c, axis=1)
    if c in X_test.columns:
        X_test = X_test.drop(c, axis=1)

# normalize all the columns by scaling
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

pca = decomposition.PCA(n_components=3)
pca.fit(X_train_std)
X = pca.transform(X_train_std)
Xt = pca.transform(X_test_std)

dtr = DecisionTreeRegressor()
dtr.fit(X, y)

print(train_data[:10])
print(X[:10])
print(dtr.predict(X[:10]))

Yt = dtr.predict(Xt)

cutoff = .5
with open('solution.csv', 'w') as of:
    of.write("id,label\n")
    for i in range(16281):
        outcome = int(Yt[i] + 1 - cutoff)
        of.write(','.join([str(i),str(outcome)])+"\n")
