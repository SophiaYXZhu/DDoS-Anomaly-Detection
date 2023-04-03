from datetime import time
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import graphviz
import collections
import pydotplus
from pandas.plotting import scatter_matrix
import pickle

df = pd.read_csv('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\cic17_binary.csv')

# drop and re-organize categorical data
dummy = pd.get_dummies(df[['Destination Port']])
df = pd.concat([df, dummy], axis = 1)
df.drop(['Destination Port', 'Label'], inplace = True, axis = 1)
predictors = df.columns[1:]
x_train, x_test, y_train, y_test = model_selection.train_test_split(df[predictors], df.result, test_size = 0.2, random_state = 42)

# Standardize data
scaler = StandardScaler()
scaler.fit(x_train)
train_scaled = scaler.transform(x_train)
test_scaled = scaler.transform(x_test)

# Create PCA for df
pca = PCA(n_components=5)
pca.fit(train_scaled)
x_train_pca = pca.transform(train_scaled)
x_test_pca = pca.transform(test_scaled)

# # grid optimaization
# max_depth = [2,3,4,5,6]
# min_samples_split = [2,4,6,8]
# min_samples_leaf = [2,4,6,8,10]
# params = {'max_depth':max_depth,'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
# grid_dtcateg = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = params, cv=10)
# grid_dtcateg.fit(x_train_pca,y_train)
# print(grid_dtcateg.best_params_)

# Construct and decision decision tree
st = time.time()
CART_class = tree.DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 10, min_samples_split = 2)
decision_tree = CART_class.fit(x_train_pca, y_train)
print("Time taken to predict:{}".format(time.time()-st))
with open('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\tree_model.pkl','wb') as f:
    pickle.dump(decision_tree,f)
st = time.time()
pred = CART_class.predict(x_test_pca)
print("Time taken to predict:{}".format(time.time()-st))
print(metrics.accuracy_score(y_test,pred))

# # ROC curve
# fpr,tpr,threshold = metrics.roc_curve(y_test,pred)
# plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
# plt.plot(fpr,tpr,color='black')
# plt.plot(0.5,0.3)
# plt.xlabel('TPR')
# plt.ylabel('FPR')
# plt.show()

# pred_list = pred.tolist()
# result = y_test.tolist()
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

# # Graph ROC
# fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
# roc_auc = metrics.auc(fpr, tpr)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# # Graph the tree
# dot_data=export_graphviz(decision_tree, out_file=None,
#                                 feature_names=['PC0','PC1','PC2','PC3','PC4'],
#                                 class_names = ["0","1",'2','3','4','5','6'],
#                                 filled=True,
#                                 rounded = True,
#                                 special_characters=True)
# graph=pydotplus.graph_from_dot_data(dot_data)
# colors =  ('lightblue', 'lightyellow', 'forestgreen', 'lightred', 'white')
# nodes = graph.get_node_list()
# print(len(nodes))
# print(len(graph.get_edge_list()))
# # for node in nodes:
# #     if node.get_name() not in ('node', 'edge'):
# #         values = CART_class.tree_.value[int(node.get_name())][0]
# #         #color only nodes where only one class is present
# #         if max(values) == sum(values):    
# #             node.set_fillcolor(colors[np.argmax(values)])
# #         #mixed nodes get the default color
# #         else:
# #             node.set_fillcolor(colors[-1])
# graph.write_png('cic_class_tree.png')

