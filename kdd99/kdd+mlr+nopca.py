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

df = pd.read_csv('kdd99_binary_all.csv')

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

train, test = model_selection.train_test_split(df, test_size = 0.2, random_state = 42)

##############################
st=time.time()
model = sm.formula.ols('result ~ duration+src_bytes+dst_bytes+wrong_fragment+urgent+hot+num_failed_logins+num_compromised+num_root+num_file_creations+num_shells+num_access_files+num_outbound_cmds+count+srv_count+serror_rate+srv_serror_rate+rerror_rate+srv_rerror_rate+same_srv_rate+diff_srv_rate+srv_diff_host_rate+dst_host_count+dst_host_srv_count+dst_host_same_srv_rate+dst_host_diff_srv_rate+dst_host_same_src_port_rate+dst_host_srv_diff_host_rate+dst_host_serror_rate+dst_host_srv_serror_rate+dst_host_rerror_rate+dst_host_srv_rerror_rate+logged_in_0+logged_in_1+root_shell_0+root_shell_1+su_attempted_0+su_attempted_1+su_attempted_2+is_host_login_0+land_0+land_1+is_guest_login_0+is_guest_login_1+protocol_type_icmp+protocol_type_tcp+protocol_type_udp+service_IRC+service_X11+service_Z39_50+service_aol+service_auth+service_bgp+service_courier+service_csnet_ns+service_ctf+service_daytime+service_discard+service_domain+service_domain_u+service_echo+service_eco_i+service_ecr_i+service_efs+service_exec+service_finger+service_ftp+service_ftp_data+service_gopher+service_harvest+service_hostnames+service_http+service_http_2784+service_http_443+service_imap4+service_iso_tsap+service_klogin+service_kshell+service_ldap+service_link+service_login+service_mtp+service_name+service_netbios_dgm+service_netbios_ns+service_netbios_ssn+service_netstat+service_nnsp+service_nntp+service_ntp_u+service_other+service_pm_dump+service_pop_2+service_pop_3+service_printer+service_private+service_remote_job+service_rje+service_shell+service_smtp+service_sql_net+service_ssh+service_sunrpc+service_supdup+service_systat+service_telnet+service_tim_i+service_time+service_uucp+service_uucp_path+service_vmnet+service_whois+flag_OTH+flag_REJ+flag_RSTO+flag_RSTOS0+flag_RSTR+flag_S0+flag_S1+flag_S2+flag_S3+flag_SF+flag_SH', data=train).fit()
# print('Variable~Coefficient:\n', model.params)
# test_X = test.drop(labels = 'result', axis = 1)
print(model.rsquared) # r-squared of the prediction model

# test_pca.columns = ['A', 'B','C','D','E','F','G','H', 'I', 'J','K','L','M','N']
pred = model.predict(exog = test)
print("Time taken to generate best param:{}".format(time.time()-st))
# print('Predicted~Real:\n', pd.DataFrame({'Prediction': pred, 'Real': test.result}))
pred = pred.to_numpy()
test_result = test.result.to_numpy()
pred_list = []
for i in pred:
    if i >= 0.5:
        pred_list.append(1)
    else:
        pred_list.append(0)
result = []
for i in test_result:
    result.append(i)
count = 0
for idx in range(len(pred_list)):
    if pred_list[idx] == result[idx]:
        count += 1
print(count/len(pred_list)) #the percentage of correct predictions