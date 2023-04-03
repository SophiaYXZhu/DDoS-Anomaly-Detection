import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn.decomposition import PCA
import matplotlib as plt
import statsmodels.api as sm
import time

df = pd.read_csv('cic17_binary_sample.csv')
df = df.drop('Destination Port', axis = 1)

train, test = model_selection.train_test_split(df, test_size = 0.2, random_state = 0)
st = time.time()
# model = sm.formula.ols('result ~ Q("Total Length of Fwd Packets")+Q("Total Length of Bwd Packets")+Q("Fwd Packet Length Mean")+Q("Bwd Packet Length Mean")+Q("Fwd Header Length")+Q("Bwd Header Length")+Q("Average Packet Size")+Q("RST Flag Count")+Q("PSH Flag Count")+Q("ACK Flag Count")+Q("URG Flag Count")+Q("CWE Flag Count")+Q("ECE Flag Count")+Q("Idle Std")+Q("Down/Up Ratio")+Q("Init_Win_bytes_backward")+Q("Subflow Fwd Packets")+Q("Subflow Bwd Packets")', data=train).fit()
model = sm.formula.ols('result ~ Q("Flow Duration")+Q("Total Fwd Packets")+Q("Total Backward Packets")+Q("Total Length of Fwd Packets")+Q("Total Length of Bwd Packets")+Q("Fwd Packet Length Max")+Q("Fwd Packet Length Min")+Q("Fwd Packet Length Mean")+Q("Fwd Packet Length Std")+Q("Bwd Packet Length Max")+Q("Bwd Packet Length Min")+Q("Bwd Packet Length Mean")+Q("Bwd Packet Length Std")+Q("Flow IAT Mean")+Q("Flow IAT Std")+Q("Flow IAT Max")+Q("Flow IAT Min")+Q("Fwd IAT Total")+Q("Fwd IAT Mean")+Q("Fwd IAT Std")+Q("Fwd IAT Max")+Q("Fwd IAT Min")+Q("Bwd IAT Total")+Q("Bwd IAT Mean")+Q("Bwd IAT Std")+Q("Bwd IAT Max")+Q("Bwd IAT Min")+Q("Fwd PSH Flags")+Q("Bwd PSH Flags")+Q("Fwd URG Flags")+Q("Bwd URG Flags")+Q("Fwd Header Length")+Q("Bwd Header Length")+Q("Fwd Packets/s")+Q("Bwd Packets/s")+Q("Min Packet Length")+Q("Max Packet Length")+Q("Packet Length Mean")+Q("Packet Length Std")+Q("Packet Length Variance")+Q("FIN Flag Count")+Q("SYN Flag Count")+Q("RST Flag Count")+Q("PSH Flag Count")+Q("ACK Flag Count")+Q("URG Flag Count")+Q("CWE Flag Count")+Q("ECE Flag Count")+Q("Down/Up Ratio")+Q("Average Packet Size")+Q("Avg Fwd Segment Size")+Q("Avg Bwd Segment Size")+Q("Fwd Header Length.1")+Q("Fwd Avg Bytes/Bulk")+Q("Fwd Avg Packets/Bulk")+Q("Fwd Avg Bulk Rate")+Q("Bwd Avg Bytes/Bulk")+Q("Bwd Avg Packets/Bulk")+Q("Bwd Avg Bulk Rate")+Q("Subflow Fwd Packets")+Q("Subflow Fwd Bytes")+Q("Subflow Bwd Packets")+Q("Subflow Bwd Bytes")+Q("Init_Win_bytes_forward")+Q("Init_Win_bytes_backward")+Q("act_data_pkt_fwd")+Q("min_seg_size_forward")+Q("Active Mean")+Q("Active Std")+Q("Active Max")+Q("Active Min")+Q("Idle Mean")+Q("Idle Std")+Q("Idle Max")+Q("Idle Min")', data=train).fit()
test_X = test.drop(labels = 'result', axis = 1)
print("Time taken to generate best param:{}".format(time.time()-st))
print(model.rsquared)

pred = model.predict(exog = test_X)
# print('Predicted~Real:\n', pd.DataFrame({'Prediction': pred, 'Real': test.result}))
pd.DataFrame({'Prediction': pred, 'Real': test.result}).to_csv('cic17_mlr_binary.csv', index = False)

pred_list = []
for i in pred:
    if i>=0.5:
        pred_list.append(1)
    else:
        pred_list.append(0)
result = []
for i in test.result:
    result.append(i)
count = 0
for idx in range(len(result)):
    if result[idx] == pred_list[idx]:
        count += 1
print(count/len(result))