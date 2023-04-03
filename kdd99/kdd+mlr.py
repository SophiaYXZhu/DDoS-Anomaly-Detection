import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, zero_one_loss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from sklearn import metrics

df = pd.read_csv('kdd99_classification_sids.csv')
# drop and re-organize categorical data
df.drop(['label'], inplace = True, axis = 1)
df.logged_in = df.logged_in.astype('category')
df.root_shell = df.root_shell.astype('category')
df.su_attempted = df.su_attempted.astype('category')
df.is_host_login = df.is_host_login.astype('category')
df.is_guest_login = df.is_guest_login.astype('category')
df.land = df.land.astype('category')
dummy = pd.get_dummies(df[['logged_in', 'root_shell', 'su_attempted', 'is_host_login' ,'land', 'is_guest_login', 'protocol_type', 'service', 'flag']])
df = pd.concat([df, dummy], axis = 1)
df.drop(['logged_in', 'root_shell', 'su_attempted', 'is_host_login' ,'land', 'is_guest_login', 'protocol_type', 'service', 'flag'], inplace = True, axis = 1)
predictors = df.columns[1:]
x_train, x_test, y_train, y_test = model_selection.train_test_split(df[predictors], df.result, test_size = 0.2, random_state = 42)
# train, test = model_selection.train_test_split(df, test_size = 0.2, random_state = 42)

###########

# Standardize data
scaler = StandardScaler()
scaler.fit(x_train)
train_scaled_data = scaler.transform(x_train)
test_scaled_data = scaler.transform(x_test)

# Create PCA for df
pca = PCA(n_components=6)
pca.fit(train_scaled_data)
train_pca = pca.transform(train_scaled_data)
test_pca = pca.transform(test_scaled_data)
train_pca = pd.DataFrame(train_pca)
train_pca.columns =['0', '1', '2', '3', '4', '5']
train_pca=train_pca.assign(result=y_train.values)

##############################
st=time.time()
model = sm.formula.ols('result ~ Q("0")+Q("1")+Q("2")+Q("3")+Q("4")+Q("5")', data=train_pca).fit()
print(model.rsquared) # r-squared of the prediction model

test_pca = pd.DataFrame(test_pca)
# test_pca.columns = ['A', 'B','C','D','E','F','G','H', 'I', 'J','K','L','M','N']
test_pca.columns = ['0', '1', '2', '3', '4', '5']
test_pca=test_pca.assign(result=y_test.values)
pred = model.predict(exog = test_pca)
print("Time taken to generate best param:{}".format(time.time()-st))

# pred_list = pred.tolist()
# result = test_pca.result.tolist()
# pred = []
# for i in pred_list:
#     if i >= 0.5:
#         pred.append(1)
#     else:
#         pred.append(0)
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# for idx in range(len(result)):
#     if result[idx] == 1 and pred[idx] == 1:
#         tp += 1
#     elif result[idx] == 0 and pred[idx] == 0:
#         tn += 1
#     elif result[idx] == 0 and pred[idx] == 1:
#         fp += 1
#     elif result[idx] == 1 and pred[idx] == 0:
#         fn += 1
# print((tp)/(tp+fp)) #the percentage of correct predictions

# # pd.DataFrame({'Prediction': pred, 'Real': test.result}).to_csv('kdd99_mlr_binary.csv', index = False) #write to csv

# print(model.summary())

# Graph ROC
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# fig = plt.figure(figsize= (12,8))
# sm.graphics.plot_ccpr_grid(model, fig=fig)
# plt.show()

# figure = plt.figure()
# ax1 = figure.add_subplot(321)
# ax2 = figure.add_subplot(322)
# ax3 = figure.add_subplot(323)
# ax4 = figure.add_subplot(324)
# ax5 = figure.add_subplot(325)
# ax6 = figure.add_subplot(326)
# ax1.scatter(model.model.exog[:,1], model.resid)
# ax2.scatter(model.model.exog[:,2], model.resid)
# ax3.scatter(model.model.exog[:,3], model.resid)
# ax4.scatter(model.model.exog[:,4], model.resid)
# ax5.scatter(model.model.exog[:,5], model.resid)
# ax6.scatter(model.model.exog[:,6], model.resid)
# ax1.set_ylabel('Residual')
# ax1.set_xlabel('PC 0')
# ax2.set_ylabel('Residual')
# ax2.set_xlabel('PC 1')
# ax3.set_ylabel('Residual')
# ax3.set_xlabel('PC 2')
# ax4.set_ylabel('Residual')
# ax4.set_xlabel('PC 3')
# ax5.set_ylabel('Residual')
# ax5.set_xlabel('PC 4')
# ax6.set_ylabel('Residual')
# ax6.set_xlabel('PC 5')

# plt.show()