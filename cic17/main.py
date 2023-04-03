import numpy as np
import pandas as pd
import pickle
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# read PKL file of decision tree (AIDS)
with open("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\tree_model.pkl", 'rb') as f:
    tree_model = pickle.load(f)

# read PKL file of SVM (SIDS)
with open("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\svm_model.pkl", 'rb') as f:
    svm_model = pickle.load(f)

df = pd.read_csv('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\cic17_classification_main.csv')
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
with open("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\scaler.pkl", "wb") as fw:
    pickle.dump(scaler, fw)
    
# Create PCA for df
pca = PCA(n_components=5)
pca.fit(train_scaled)
x_train_pca = pca.transform(train_scaled)
x_test_pca = pca.transform(test_scaled)
with open("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\pca.pkl", "wb") as fw:
    pickle.dump(pca, fw)

pred = tree_model.predict(x_test_pca)
print(metrics.accuracy_score(y_test, pred))

# filter only network anomalies for SIDS
x_test["tree_predict"] = pred
# x_test.to_csv("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\cic17_binary_main_tree_test.csv", index=False)
x_test_anomaly = x_test[x_test.tree_predict == 1]
x_test_anomaly.drop('tree_predict', inplace = True, axis=1)
# x_test_anomaly_pca = pca.transform(x_test_anomaly)
# x_test_anomaly.to_csv("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\cic17_binary_main_tree_test_anomaly.csv", index=False)
# x_test_anomaly = pd.read_csv("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\cic17_binary_main_tree_test_anomaly.csv")
############## Need preprocessing it by scaler
x_test_anomaly_scaled = scaler.transform(x_test_anomaly)
x_test_anomaly_pca = pca.transform(x_test_anomaly_scaled)



pred_svc = svm_model.predict(x_test_anomaly_pca)
# print(x_test_anomaly_pca['class'])
print(metrics.accuracy_score(x_test_anomaly['class'], pred_svc))