import os
import collections
import pathlib
import itertools
import pickle
import time

os.environ['OPENBLAS_NUM_THREADS']='50'


import numpy as np
import pandas as pd
import scipy
import random 
import copy
import more_itertools
from tqdm import tqdm
import scipy.special
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import import_data_for_batch
import initialisation
import VEM
import generate_data_clusters


##------------- Parameters data -------------
data_generation = True
build_batchs = True
execute_VEM = True
init = True
n_batches = 100
n, m = int(5e4), 1000
K,L = 100,20
n_clusters_with_signal = int(0.20*K*L)
learning_rate = 1e-3
n_sim = 1
basefilename=f"K{K}_L{L}_sim{n_sim}"


### Folders containing the data and where to store the batches files
baseREPname = f"simulation"
REP_DATA_KER = pathlib.Path(f"/work_home/adewalsche/lbm_association/data/{baseREPname}/ker_{n_sim}/")
REP_PROV_BATCH = pathlib.Path(f"/work_home/adewalsche/lbm_association/data/{baseREPname}/prov/")
REP_BATCH = pathlib.Path(f"/work_home/adewalsche/lbm_association/data/{baseREPname}/batch_{n_sim}/")
REP_RES = pathlib.Path(f"/work_home/adewalsche/lbm_association/data/{baseREPname}/results/")
REP_TRUTH = pathlib.Path(f"/work_home/adewalsche/lbm_association/data/{baseREPname}/results/")


## Generate simulated data
if data_generation:
    generate_data_clusters.generate_data(n=n, m=m, K=K, L=L, 
                                        mean_under_H1 = 3,
                                        n_groups_with_signal = n_clusters_with_signal,
                                        REP_DATA_KER=REP_DATA_KER,
                                        REP_RES=REP_RES, basefilename = basefilename)



## Build the batches and import data
if build_batchs :
    import_data_for_batch.build_loadable_data(input_path=REP_DATA_KER,
                                            output_path=REP_BATCH, prov_path= REP_PROV_BATCH,
                                            n_batches=n_batches,input_format="parquet")

data = import_data_for_batch.load_loadable_data(REP_BATCH)


##------------- Initialisation -------------
## from spectral clustering
if init:
    print("Initialisation")
    taus_u, tau_v = initialisation.init_from_sc(data, K, L)
    
    with open(REP_DATA_KER / 'p0.pkl', "rb") as f:
        p0 = pickle.load(f)

    pi, nu, lambda_param = initialisation.init_stepM(taus_u, tau_v, p0)
    
    ##initialisation
    diff_taus_u = []
    diff_tau_v = []
    entropy_vec = []
    rows_group_splitting = []
    cols_group_splitting = []
    loglikelihood_crit = []
    ARI_tau_u = []
    ARI_tau_u_cluster = []
    ARI_tau_v = []

else: ##initialisation from previous VEM
    ### Load the results
    with open(REP_RES / f'taus_u_{basefilename}.pkl', "rb") as f:
        taus_u = pickle.load(f)

    with open(REP_RES / f'taus_v_{basefilename}.pkl', "rb") as f:
        tau_v = pickle.load(f)

    with open(REP_RES / f'lambda_{basefilename}.pkl', "rb") as f:
        lambda_param = pickle.load(f)

    with open(REP_RES / f'pi_{basefilename}.pkl', "rb") as f:
        pi = pickle.load(f)

    with open(REP_RES / f'nu_{basefilename}.pkl', "rb") as f:
        nu = pickle.load(f)

    with open(REP_RES / f'likelihood_{basefilename}.pkl', "rb") as f:
        loglikelihood_crit = pickle.load(f)

    with open(REP_RES / f'changement_tau_v_{basefilename}.pkl', "rb") as f:
        diff_tau_v = pickle.load(f)
    
    with open(REP_RES / f'changement_tau_u_{basefilename}.pkl', "rb") as f:
        diff_taus_u = pickle.load(f)
    
    with open(REP_RES / f'entropy_{basefilename}.pkl', "rb") as f:
        entropy_vec = pickle.load(f)

    with open(REP_RES / f'ARI_tau_u_{basefilename}.pkl', "rb") as f:
        ARI_tau_u = pickle.load(f)
        
    with open(REP_RES / f'ARI_tau_u_cluster_{basefilename}.pkl', "rb") as f:
        ARI_tau_u_cluster = pickle.load(f)
        
    with open(REP_RES / f'ARI_tau_v_{basefilename}.pkl', "rb") as f:
        ARI_tau_v = pickle.load(f)
        
    # with open(REP_RES / f'rows_group_splitting_{basefilename}.pkl', "rb") as f:
    #     rows_group_splitting = pickle.load(f)

    # with open(REP_RES / f'cols_group_splitting_{basefilename}.pkl', "rb") as f:
    #     cols_group_splitting = pickle.load(f)
    
    rows_group_splitting = []
    cols_group_splitting = []
    

if execute_VEM:
    print("VEM")
    VEM.perform_VEM(data=data, taus_u= taus_u, tau_v= tau_v, pi= pi, nu= nu, lambda_param=lambda_param,
    loglikelihood_crit = loglikelihood_crit, diff_taus_u = diff_taus_u, diff_tau_v = diff_tau_v, entropy_vec = entropy_vec, 
    rows_group_splitting = rows_group_splitting, cols_group_splitting = cols_group_splitting, 
    ARI_tau_u = ARI_tau_u, ARI_tau_u_cluster=ARI_tau_u_cluster, ARI_tau_v=ARI_tau_v,
    learning_rate=1e-3,optimizer="RMSProp",max_n_epochs=1000,
    basefilename=basefilename,results_path=REP_RES,simulated_data=True)

