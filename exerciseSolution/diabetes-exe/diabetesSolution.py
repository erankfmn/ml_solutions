import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = read_csv('diabetes.csv')
print(dataset.describe())


######  mark missing data

#replace missing data with NaN
zero_not_accepted=['plasma','blood pressure','Triceps thickness','serum insulin','BMI']
for column in zero_not_accepted:
    print(column,(dataset[column] == 0).sum())
    dataset[column] = dataset[column].replace(0, np.NaN)


print(dataset.describe())


#ToDo:
#normilize values
# use Imputer to deal with missing values
# use OneHotEncode for categorial data
# balance the data

# fill missing values with mean column values
values = dataset.values
names=dataset.columns._data[:-1]
data=values[:,:-1]

X=dataset.iloc[:,:9]
y=dataset.iloc[:,9]

X["ethnicity"] = X["ethnicity"].astype('category')


#
# enc = preprocessing.OneHotEncoder(categorical_features=[-1])
# data = enc.fit_transform(data).toarray()

# one_hot_encoder = OneHotEncoder(sparse=False)
# one_hot_encoder = one_hot_encoder.fit_transform(np.array(diabetes['ethnicity']).reshape(-1, 1))
#
# diabetes['ethnicity'] = one_hot_encoder


X_train,X_test,y_train,y_test=train_test_split(X,y)



imputer = preprocessing.Imputer()
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
#
sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

from sklearn import tree
depth=10
clf = tree.DecisionTreeClassifier(max_depth=depth ,class_weight={1:2})


clf = clf.fit(X_train, y_train)
print("score for",depth, clf.score(X_test, y_test))


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=names, filled=True, rounded=True, class_names=['sick','healthy'] )
graph = graphviz.Source(dot_data)
graph.render("tree")