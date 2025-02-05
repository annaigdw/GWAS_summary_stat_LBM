import itertools
import pathlib
import pickle

import numpy as np
import scipy
import scipy.special
import sklearn.cluster
import more_itertools

from import_data_for_batch import LoadableData, BatchData

def _X_batch(batch: BatchData):
    f0_copy = batch.f0.copy()
    f0_copy[f0_copy==0] = np.min(f0_copy[f0_copy> 0])
    f1_copy = batch.f1.copy()
    f1_copy[f1_copy==0] = np.min(f1_copy[f1_copy> 0])
      
    return scipy.special.expit(np.log(f1_copy)-np.log(f0_copy))

def _init_from_sc_rows(data: LoadableData, k: int, k_desc: int):
    _,m = data.batches[0].f0.shape
    
    X_sum_rows = np.zeros(m)
    for batch in data.batches:
        X_sum_rows += _X_batch(batch).sum(axis=0)

    X2TX2 = np.zeros((m,m))
    for batch in data.batches:
        X_batch = _X_batch(batch)
        X2_batch = ((X_batch@X_sum_rows)**-.5)[:,None]*X_batch
        X2TX2 += X2_batch.T@X2_batch

    _, v = scipy.linalg.eigh(X2TX2, subset_by_index=(m-k_desc, m-1))

    desc_rows = []
    for batch in data.batches:
        X_batch = _X_batch(batch)
        X2_batch = ((X_batch@X_sum_rows)**-.5)[:,None]*X_batch
        desc_rows.append(X2_batch@v)

    concatenated_all_rows = np.concatenate(desc_rows, axis=0)

    kmeans_model = sklearn.cluster.KMeans(k, n_init=10)
    kmeans_model.fit(concatenated_all_rows)

    return [kmeans_model.labels_[a:b] for a,b in more_itertools.pairwise(itertools.chain((None,),np.cumsum([desc.shape[0] for desc in desc_rows])))]

def _init_from_sc_cols(data: LoadableData, k: int, k_desc: int):
    _,m = data.batches[0].f0.shape

    M = np.zeros((m,m))
    for batch in data.batches:
        X_batch = _X_batch(batch)
        M += X_batch.T@X_batch
    
    norma = M.sum(axis=1)**-.5
    M = M*norma[:,None]*norma[None,:]

    values, vectors = scipy.linalg.eigh(M, subset_by_index=(m-k_desc, m-1))
    desc_cols = vectors*np.sqrt(values)[None,:]

    kmeans_model = sklearn.cluster.KMeans(k, n_init=10)
    kmeans_model.fit(desc_cols)
    return kmeans_model.labels_

def init_from_sc(data: LoadableData, k_rows: int, k_cols: int):
    n_batches = len(data.batches)
    m = data.batches[0].f0.shape[1]
    k_desc = min(k_cols, k_rows)

    labels_u_init, labels_v_init = (_init_from_sc_rows(data,k_rows,k_desc), _init_from_sc_cols(data, k_cols, k_desc))
    taus_u = [None] * n_batches
    tau_v = np.zeros(shape=(m,k_cols)) + 0.1/(k_cols-1)

    for i_b in range(n_batches):
        taus_u[i_b] = np.zeros(shape=(labels_u_init[i_b].shape[0],k_rows)) + 0.1/(k_rows-1)
        for i,k in enumerate(labels_u_init[i_b]) :
            taus_u[i_b][i,k]=0.9

    for i,k in enumerate(labels_v_init) :
        tau_v[i,k]=0.9

    return (taus_u, tau_v)


def init_stepM(taus_u, tau_v, p0: np.array):
    n = sum(tau_u.shape[0] for tau_u in taus_u)
    m = tau_v.shape[0]

    K = taus_u[0].shape[1]
    L = tau_v.shape[1]

    pi = np.array([tau_u.sum(axis=0) for tau_u in taus_u]).sum(axis=0)/n
    nu = tau_v.mean(axis=0)

    lambda_param_init = np.array((p0 @ tau_v) / (nu*m))* np.ones((K,1))
    logit_lambda_init = np.ones(shape=(K,L)) * scipy.special.logit(lambda_param_init) 
    lambda_param = scipy.special.expit(logit_lambda_init)
    return(pi, nu, lambda_param)