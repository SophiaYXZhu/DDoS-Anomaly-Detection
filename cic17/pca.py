import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv('cic17_binary.csv')
result = df.result
df = df.drop(['Destination Port', 'result', 'Label'], axis = 1)
# Standardize data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)


# Create PCA for df
pca = PCA(n_components=7)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
print(pca.components_)
print(pca.explained_variance_ratio_)


# # Graphs
# plt.figure(figsize = (8,6))
# plt.scatter(x_pca[:,0], x_pca[:,1], x_pca[:,2], c=result, cmap = 'plasma')
# plt.savefig('scattered_pca.png')

df_comp = pd.DataFrame(pca.components_, columns = df.columns)
plt.figure(figsize = (20,20))
# print(pca.components_)
# print(df.columns)

sns.heatmap(df_comp, cmap = 'plasma', xticklabels=True, yticklabels=True)
plt.show(sns)


# # Estimate feature importance
# model = ExtraTreesClassifier()
# model.fit(df, result)
# print(model.feature_importances_)