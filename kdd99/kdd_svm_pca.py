import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, zero_one_loss
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from sklearn import metrics

df = pd.read_csv('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\kdd\\kdd99_classification_sids.csv')
print(df['result'].unique())
df.drop(['label'], inplace = True, axis = 1)
df.logged_in = df.logged_in.astype('category')
df.root_shell = df.root_shell.astype('category')
df.su_attempted = df.su_attempted.astype('category')
df.is_host_login = df.is_host_login.astype('category')
df.is_guest_login = df.is_guest_login.astype('category')
df.land = df.land.astype('category')
# drop and re-organize categorical data
dummy = pd.get_dummies(df[['logged_in', 'root_shell', 'su_attempted', 'is_host_login' ,'land', 'is_guest_login', 'protocol_type', 'service', 'flag']])
df = pd.concat([df, dummy], axis = 1)
df.drop(['logged_in', 'root_shell', 'su_attempted', 'is_host_login' ,'land', 'is_guest_login', 'protocol_type', 'service', 'flag'], inplace = True, axis = 1)
predictors = df.columns[1:]
print(predictors)
x_train, x_test, y_train, y_test = model_selection.train_test_split(df[predictors], df.result, test_size = 0.2, random_state = 1234)

# Standardize data
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled_data = scaler.transform(x_train)
x_test_scaled_data = scaler.transform(x_test)
# Create PCA for df
pca = PCA(n_components=4)
pca.fit(x_train_scaled_data)
X_train_pca = pca.transform(x_train_scaled_data)
X_test_pca = pca.transform(x_test_scaled_data)
print('pca done')

# SVM
st = time.time()
# clf = svm.SVC(C = 100, gamma = 0.1, max_iter = 1000000, kernel = 'linear')
clf = svm.SVC(C = 200, gamma = 0.1, max_iter=500, kernel = 'rbf')
clf.fit(X_train_pca, y_train)
print("Time taken to generate best param:{}".format(time.time()-st))
st = time.time()
pred_svc = clf.predict(X_test_pca)
print("Time taken to generate best param:{}".format(time.time()-st))
print(metrics.accuracy_score(y_test,pred_svc))
result = y_test.tolist()
pred = pred_svc.tolist()

tp = 0
tn = 0
fp = 0
fn = 0
for idx in range(len(result)):
    if result[idx] == 1 and pred[idx] == 1:
        tp += 1
    elif result[idx] != 1 and pred[idx] != 1:
        tn += 1
    elif result[idx] != 1 and pred[idx] == 1:
        fp += 1
    elif result[idx] == 1 and pred[idx] != 1:
        fn += 1
print(1)
print((tp)/(tp+fp)) #precision
print(tp/(tp+fn)) #recall
print((tp+fn)/(tp+fp+fn+tn))

tp = 0
tn = 0
fp = 0
fn = 0
for idx in range(len(result)):
    if result[idx] == 2 and pred[idx] == 2:
        tp += 1
    elif result[idx] != 2 and pred[idx] != 2:
        tn += 1
    elif result[idx] != 2 and pred[idx] == 2:
        fp += 1
    elif result[idx] == 2 and pred[idx] != 2:
        fn += 1
print(2)
print((tp)/(tp+fp)) #the percentage of correct predictions
print(tp/(tp+fn))
print((tp+fn)/(tp+fp+fn+tn))

tp = 0
tn = 0
fp = 0
fn = 0
for idx in range(len(result)):
    if result[idx] == 3 and pred[idx] == 3:
        tp += 1
    elif result[idx] != 3 and pred[idx] != 3:
        tn += 1
    elif result[idx] != 3 and pred[idx] == 3:
        fp += 1
    elif result[idx] == 3 and pred[idx] != 3:
        fn += 1
print(3)
print((tp)/(tp+fp)) #the percentage of correct predictions
print(tp/(tp+fn))
print((tp+fn)/(tp+fp+fn+tn))

tp = 0
tn = 0
fp = 0
fn = 0
for idx in range(len(result)):
    if result[idx] == 4 and pred[idx] == 4:
        tp += 1
    elif result[idx] != 4 and pred[idx] != 4:
        tn += 1
    elif result[idx] != 4 and pred[idx] == 4:
        fp += 1
    elif result[idx] == 4 and pred[idx] != 4:
        fn += 1
print(4)
print((tp)/(tp+fp)) #the percentage of correct predictions
print(tp/(tp+fn))
print((tp+fn)/(tp+fp+fn+tn))

tp = 0
tn = 0
fp = 0
fn = 0
for idx in range(len(result)):
    if result[idx] == 5 and pred[idx] == 5:
        tp += 1
    elif result[idx] != 5 and pred[idx] != 5:
        tn += 1
    elif result[idx] != 5 and pred[idx] == 5:
        fp += 1
    elif result[idx] == 5 and pred[idx] != 5:
        fn += 1
print(5)
print((tp)/(tp+fp)) #the percentage of correct predictions
print(tp/(tp+fn))
print((tp+fn)/(tp+fp+fn+tn))
