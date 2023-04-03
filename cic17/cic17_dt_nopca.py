from datetime import time
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
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
from sklearn.tree import export_graphviz

df = pd.read_csv('cic17_binary_sample.csv')

# drop and re-organize categorical data
dummy = pd.get_dummies(df[['Destination Port']])
df = pd.concat([df, dummy], axis = 1)
df.drop(['Destination Port', 'Label'], inplace = True, axis = 1)
predictors = df.columns[1:]
x_train, x_test, y_train, y_test = model_selection.train_test_split(df[predictors], df.result, test_size = 0.2, random_state = 42)
predictors = x_train.columns.to_list()

# # grid optimaization
# max_depth = [2,3,4,5,6]
# min_samples_split = [2,4,6,8]
# min_samples_leaf = [2,4,6,8,10]
# params = {'max_depth':max_depth,'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
# grid_dtcateg = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = params, cv=10)
# grid_dtcateg.fit(x_train_pca,y_train)
# print(grid_dtcateg.best_params_)

st = time.time()
CART_class = tree.DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 10, min_samples_split = 2)
decision_tree = CART_class.fit(x_train, y_train)
print("Time taken to generate best param:{}".format(time.time()-st))
st = time.time()
pred = CART_class.predict(x_test)
print("Time taken to generate best param:{}".format(time.time()-st))
print(metrics.accuracy_score(y_test,pred))

dot_data=export_graphviz(decision_tree, out_file=None,
                                feature_names=predictors,
                                class_names = ["0","1"],
                                filled=True,
                                rounded = True,
                                special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data)
colors =  ('lightblue', 'lightyellow', 'forestgreen', 'lightred', 'white')
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