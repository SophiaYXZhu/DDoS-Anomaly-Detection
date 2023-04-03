import pandas as pd
import numpy as np

cic = pd.read_csv('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\cic\\cic17_binary.csv')
print(cic['result'].unique())

# # # df = df[df[' Label'] != 'Heartbleed']
# cic.loc[cic['Label']=="BENIGN", 'detection'] = 0
# cic.loc[cic['Label']=="Bot", 'detection']=1
# cic.loc[cic['Label']=="DoS slowloris", 'detection']=2
# cic.loc[cic['Label']=="DoS Slowhttptest", 'detection']=3
# cic.loc[cic['Label']=="DoS Hulk", 'detection']=4
# cic.loc[cic['Label']=="DoS GoldenEye", 'detection']=5
# cic.loc[cic['Label']=="DDoS", 'detection']=6

cic.loc[cic['Label']=="BENIGN", 'class'] = 0
cic.loc[cic['Label']=="Bot", 'class']=1
cic.loc[cic['Label']=="DoS slowloris", 'class']=2
cic.loc[cic['Label']=="DoS Slowhttptest", 'class']=3
cic.loc[cic['Label']=="DoS Hulk", 'class']=4
cic.loc[cic['Label']=="DoS GoldenEye", 'class']=5
cic.loc[cic['Label']=="DDoS", 'class']=6

# cols= cic.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# cic = cic[cols]

# # # cic=pd.read_csv("cic17_binary.csv")
# # # cic_sample=cic.groupby("Label").head(20000)
# # cic.to_csv("cic17_binary_classification.csv", index=False)

# df = df[~df['Label'].isin(['BENIGN'])]
# print(df['result'].unique())
# print(df['Label'].unique())

cic.to_csv('cic17_classification_main.csv')