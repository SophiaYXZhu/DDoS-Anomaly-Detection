import pandas as pd
import numpy as np

cic=pd.read_csv("C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\kdd\\kdd99_classification_sids.csv")
cic = cic[~cic['label'].isin(['portsweep.'])]
cic = cic[~cic['label'].isin(['satan.'])]
cic = cic[~cic['label'].isin(['ftp_write.'])]
cic = cic[~cic['label'].isin(['guess_password.'])]
cic = cic[~cic['label'].isin(['imap.'])]
cic = cic[~cic['label'].isin(['multihop.'])]
cic = cic[~cic['label'].isin(['phf.'])]
cic = cic[~cic['label'].isin(['spy.'])]
cic = cic[~cic['label'].isin(['warezclient.'])]
cic = cic[~cic['label'].isin(['warezmaster.'])]
cic = cic[~cic['label'].isin(['buffer_overflow.'])]
cic = cic[~cic['label'].isin(['loadmodule.'])]
cic = cic[~cic['label'].isin(['perl.'])]
cic = cic[~cic['label'].isin(['sub_total.'])]
cic = cic[~cic['label'].isin(['rookit.'])]
cic.loc[cic['label']=="pod.", 'result']=1
cic.loc[cic['label']=="smurf.", 'result']=2
cic.loc[cic['label']=="teardrop.", 'result']=3
cic.loc[cic['label']=="ipsweep.", 'result']=4
cic.loc[cic['label']=="nmap.", 'result']=5
# print(cic['label'].unique())
# print(cic['result'].unique())
cic.to_csv('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\kdd\\kdd99_dos_classification.csv')

#### save up to 500K per label value. Total 1.545m rows
# df=df.groupby("label").head(500000)

# find unqiue label value, to set result (0,1) based on label value
# cic.loc[cic['label']=="normal", 'result'] = 0
# cic.loc[cic['label']=="back", 'result']=0
# cic.loc[cic['label']=="land", 'result']=0
# cic.loc[cic['label']=="neptune", 'result']=0
# cic.loc[cic['label']=="pod.", 'result']=1
# cic.loc[cic['label']=="smurf.", 'result']=1
# cic.loc[cic['label']=="teardrop.", 'result']=1
# cic.loc[cic['label']=="ipsweep.", 'result']=1
# cic.loc[cic['label']=="nmap.", 'result']=1
# cic.loc[cic['label']=="portsweep.", 'result']=2
# cic.loc[cic['label']=="satan.", 'result']=2
# cic.loc[cic['label']=="ftp_write.", 'result']=2
# cic.loc[cic['label']=="guess_password.", 'result']=2
# cic.loc[cic['label']=="imap.", 'result']=2
# cic.loc[cic['label']=="multihop.", 'result']=2
# cic.loc[cic['label']=="phf.", 'result']=3
# cic.loc[cic['label']=="spy.", 'result']=3
# cic.loc[cic['label']=="warezclient.", 'result']=3
# cic.loc[cic['label']=="warezmaster.", 'result']=3
# cic.loc[cic['label']=="buffer_overflow.", 'result']=3
# cic.loc[cic['label']=="loadmodule.", 'result']=3
# cic.loc[cic['label']=="perl.", 'result']=4
# cic.loc[cic['label']=="subtotal.", 'result']=4
# cic.loc[cic['label']=="rookit.", 'result']=4

# cic.to_csv("kdd99_binary_classification.csv", index=False)
# cic = cic[~cic['label'].isin(['normal.'])]
# cic = cic.dropna()
# print(cic['result'].unique())
# cic.to_csv('kdd99_classification_sids.csv')

# df=pd.read_csv('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\kdd\\kdd99_binary_classification.csv')
# df = df[~df['label'].isin(['normal.'])]
# print(df['result'].unique())
# df.to_csv('C:\\Users\\Duality\\Documents\\Sophia+Zhu\\Pioneer\\Research+Paper\\Code\\kdd\\kdd99_classification_sids.csv')