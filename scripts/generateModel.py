import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# from matplotlib import style
from sklearn import svm, preprocessing, cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
# style.use('ggplot')

NUMERICAL_COLUMNS = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_COLUMNS = ['workclass','education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

scaler = MinMaxScaler()

df = pd.read_csv('../data/adultData.csv')
dfX = pd.get_dummies(df.drop('earning', axis=1), prefix=CATEGORICAL_COLUMNS)
dfX[NUMERICAL_COLUMNS] = scaler.fit_transform(dfX[NUMERICAL_COLUMNS])
with open('inputRows.csv', 'wb') as csvFile:
    writer = csv.writer(csvFile);
    writer.writerow(dfX.columns)

X = np.array(dfX)
y = np.array(df['earning'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel='linear', verbose=1)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

pickle.dump(clf, open('classifier.pickle', 'wb'))
print ('end of program')

