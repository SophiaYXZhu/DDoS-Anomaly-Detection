import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv('kdd99_binary_complete.csv')
result = df.result
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

# Standardize data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# Create PCA for df
pca = PCA(n_components=6)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
print(pca.components_)
print(pca.explained_variance_ratio_)

# Graphs
plt.figure(figsize = (8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], x_pca[:,2], c=result, cmap = 'plasma')
plt.savefig('scattered_pca.png')

df_comp = pd.DataFrame(pca.components_, columns = df.columns)
plt.figure(figsize = (20,20))
print(pca.components_)
print(df.columns)

sns.heatmap(df_comp, cmap = 'plasma', xticklabels=True, yticklabels=True)
plt.show(sns)