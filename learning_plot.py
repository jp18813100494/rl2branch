# code:utf-8  	Ubuntu
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import os
import pandas as pd


def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")
 
            smooth_data.append(d)
 
    return smooth_data

env_name = "Cauctions"
iql_samples_per_eval = 5000
xls_path =  os.path.join('D://学术之路//Myproject//毕设论文//补充材料//Branch//results',env_name)
iql_file = os.path.join(xls_path,'IQL-branch.xlsx')
iql_nnods = pd.read_excel(iql_file,sheet_name='valid_nodes',index_col=[0,1])
rl_file = os.path.join(xls_path,'RL.xlsx')
rl_nnodes = pd.read_excel(iql_file,sheet_name='RL',index_col=[0,1,2,3])
rl_samples_file = os.path.join(xls_path,'RL_train_samples.xlsx')
rl_samples = pd.read_excel(iql_file,sheet_name='RL_train_samples',index_col=[0,4,5,6])

iql_samples = iql_nnods[:,0]*iql_samples_per_eval
iql_nnodes =  iql_nnods[:,1]

rl_steps = rl_nnodes[:,0]
rl_mdp_nnodes = rl_nnodes[:,1]
rl_tmdp_dfs_nnodes = rl_nnodes[:,2]
rl_tmdp_objlim_nnodes = rl_nnodes[:,3]

rl_mdp_samples = rl_samples[:,1]
rl_tmdp_dfs_samples = rl_samples[:,2]
rl_tmdp_objlim_samples = rl_samples[:,3]

plt.plot(iql_samples,iql_nnodes,linewidth='2', label = "IQL-branch", color=' red')
plt.legend(loc='upper left')
plt.show()
plt.savefig(xls_path+'//{}.png'.format(env_name))