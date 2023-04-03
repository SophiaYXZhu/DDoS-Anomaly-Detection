import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
import graphviz
import collections
import pydotplus
from sklearn.tree import export_graphviz

df = pd.read_csv('kdd99_binary_complete.csv')
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
x_train, x_test, y_train, y_test = model_selection.train_test_split(df[predictors], df.result, test_size = 0.2, random_state = 42)
predictors = x_train.columns.to_list()

# # grid optimaization
# max_depth = [2,3,4,5,6]
# min_samples_split = [2,4,6,8]
# min_samples_leaf = [2,4,6,8,10]
# params = {'max_depth':max_depth,'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
# grid_dtcateg = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = params, cv=10)
# grid_dtcateg.fit(x_train,y_train)
# print(grid_dtcateg.best_params_)

st=time.time()
CART_class = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf = 2, min_samples_split=8)
decision_tree = CART_class.fit(x_train, y_train)
print("Time taken to generate best param:{}".format(time.time()-st))
st=time.time()
pred = CART_class.predict(x_test)
print("Time taken to generate best param:{}".format(time.time()-st))
print(metrics.accuracy_score(y_test,pred))

# pred = pred.tolist()
# result = y_test.tolist()
# tp = 0
# tn = 0
# fp = 0
# fn = 0
# for idx in range(len(pred)):
#     if result[idx] == 1 and pred[idx] == 1:
#         tp += 1
#     elif result[idx] == 0 and pred[idx] == 0:
#         tn += 1
#     elif result[idx] == 0 and pred[idx] == 1:
#         fp += 1
#     elif result[idx] == 1 and pred[idx] == 0:
#         fn += 1
# print((tp+tn)/(tp+tn+fn+fp)) #the percentage of correct predictions

# Graph
dot_data=export_graphviz(decision_tree, out_file=None,
                                feature_names=predictors,
                                class_names = ["0","1"],
                                filled=True,
                                rounded = True,
                                special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data)
colors =  ('lightblue', 'lightyellow', 'forestgreen', 'lightred', 'white')
nodes = graph.get_node_list()
nodes = graph.get_node_list()
# print(len(nodes))
# print(len(graph.get_edge_list()))
for node in nodes:
    if node.get_name() not in ('node', 'edge'):
        values = CART_class.tree_.value[int(node.get_name())][0]
        #color only nodes where only one class is present
        if max(values) == sum(values):    
            node.set_fillcolor(colors[np.argmax(values)])
        #mixed nodes get the default color
        else:
            node.set_fillcolor(colors[-1])
graph.write_png('tree_colored.png')