import pathlib
import pickle

import numpy as np
import pandas as pd
import scipy

from import_data_for_batch import LoadableData, BatchData


def generate_data(n: int, m: int, K: int, L: int, mean_under_H1: float, n_groups_with_signal: int, REP_DATA_KER: pathlib.Path, REP_RES: pathlib.Path, basefilename: str ):
    
    ### Parameters for data generation
    pi_true = np.ones(K)*1/K
    nu_true = np.ones(L)*1/L
    mu = np.ones(m)*mean_under_H1

    lambda_true = np.random.uniform(size=(K,L),low=0.85,high=1)
    index_clusters = np.random.choice(K * L, size=n_groups_with_signal, replace=False)
    rows_index_clusters, cols_index_clusters = np.unravel_index(index_clusters, lambda_true.shape)
    lambda_true[rows_index_clusters,cols_index_clusters] = np.random.uniform(size=n_groups_with_signal,low=0.3,high=0.6)

    ##Latent variables generation
    U = np.array(np.random.choice(a= np.arange(K),p=pi_true, size=n))
    V = np.array(np.random.choice(a= np.arange(L),p=nu_true, size=m))

    ##Z-scores profile generation
    X = np.random.normal(size=(n,m))
    f1 = np.random.normal(loc =mu[None,:], size=(n,m))
    lambda_mat = lambda_true[U[:,None],V[None,:]]
    latent_mat = np.random.uniform(size=(n,m)) < (1-lambda_mat)
    X[latent_mat] = f1[latent_mat]

    tau_u_true = np.zeros(shape=(n,K))
    tau_v_true = np.zeros(shape=(m,L))

    for i,k in enumerate(U) :
        tau_u_true[i,k]=1

    for i,k in enumerate(V) :
        tau_v_true[i,k]=1

    ## build the ker estimate dataframes
    phi_x = scipy.stats.norm.pdf(x=X)
    f1_x = scipy.stats.norm(loc=mu[None,:]).pdf(x=X)
    Marker_Name_x =  [f"SNP_{i}" for i in range(1, n+1)]
    Chromosome_x = np.ones(n)
    Position_x = range(1,n+1)

    for i in range(m):
        df_col = pd.DataFrame({'Marker_Name': Marker_Name_x, 'Chromosome': Chromosome_x, 'Position': Position_x, 'f0': phi_x[:,i],'f1':f1_x[:,i]})
        df_col.to_parquet(REP_DATA_KER / f'Ker_estimate_{i+1:08d}.parquet')

    p0 = np.minimum(2 * np.mean(X < 0, axis=0), 1 - 1 / X.shape[0])
    with open(REP_DATA_KER / "p0.pkl", "wb") as f:
            pickle.dump(p0, f)

    ### Save the true parameters
    with open(REP_RES / f'tau_u_true_{basefilename}.pkl', "wb") as f:
        pickle.dump(tau_u_true, f)

    with open(REP_RES / f'tau_v_true_{basefilename}.pkl', "wb") as f:
        pickle.dump(tau_v_true, f)

    with open(REP_RES / f'lambda_true_{basefilename}.pkl', "wb") as f:
        pickle.dump(lambda_true, f)

    with open(REP_RES / f'pi_true_{basefilename}.pkl', "wb") as f:
        pickle.dump(pi_true, f)

    with open(REP_RES / f'nu_true_{basefilename}.pkl', "wb") as f:
        pickle.dump(nu_true, f)

    with open(REP_RES / f'rows_index_clusters_{basefilename}.pkl', "wb") as f:
        pickle.dump(rows_index_clusters, f)

    with open(REP_RES / f'cols_index_clusters_{basefilename}.pkl', "wb") as f:
        pickle.dump(cols_index_clusters, f)


def compute_loglikelihood_with_truth(data: LoadableData,K: int, L:int,REP_TRUTH: pathlib.Path,REP_RES: pathlib.Path, REP_DATA_KER: pathlib.Path):

    n_batches = len(data.batches)
    n = sum(b.f0.shape[0] for b in data.batches)
    m = data.batches[0].f0.shape[1]

    ## Load the true parameters
    with open(REP_TRUTH / f'{"lambda_true"}{K}{".pkl"}', "rb") as f:
        lambda_true = pickle.load(f)

    with open(REP_TRUTH / f'{"pi_true"}{K}{".pkl"}', "rb") as f:
        pi_true = pickle.load(f)

    with open(REP_TRUTH / f'{"nu_true"}{K}{".pkl"}', "rb") as f:
        nu_true = pickle.load(f)
    with open(REP_TRUTH / f'tau_u_true{K}.pkl', "rb") as f:
        tau_u_true = pickle.load(f)

    with open(REP_TRUTH / f'tau_v_true{K}.pkl', "rb") as f:
        tau_v_true = pickle.load(f)

    logit_lambda_true = scipy.special.logit(lambda_true) 
    one_minus_lambda_param_true = scipy.special.expit(-logit_lambda_true)

    ## With true lambda
    loglikelihood_true = 0
    taus_u_true=[tau_u_true[i_b::n_batches] for i_b in range(n_batches)]
    for i_b, batch in enumerate(data.batches):
        loglikelihood_true += (taus_u_true[i_b] @ np.log(pi_true)[:,None]).sum()
        for k in range(K):
            for l in range(L):
                loglikelihood_true += taus_u_true[i_b][:,k] @ np.log(one_minus_lambda_param_true[k,l]*batch.f1+lambda_true[k,l]*batch.f0) @ tau_v_true[:,l]
    loglikelihood_true += (tau_v_true @ np.log(nu_true)[:,None]).sum()           

    with open(REP_RES / f'{"likelihood_true_"}{K}{".pkl"}', "wb") as f:
            pickle.dump(loglikelihood_true, f)

    print(f"    loglikelihood true ={loglikelihood_true:.1f}")

    ## With lambda optimisation
    n_epochs = 1000
    beta1 = 0.9
    beta2 = 0.999
    m_adam = 0
    v_adam = 0
    e_RMSProp = 0
    epsilon = 1e-8

    optimizer ="adam"
    t = 0

    learning_rate = 1e-3

    pi = np.array([tau_u.sum(axis=0) for tau_u in taus_u_true]).sum(axis=0)/n
    nu = tau_v_true.mean(axis=0)

    with open(REP_DATA_KER / "p0.pkl", 'rb') as file:
        p0 = pickle.load(file)

    lambda_param = ((p0 @ tau_v_true) / (nu*m))[None,:] * np.ones((K,1))
    logit_lambda = np.ones(shape=(K,L)) * scipy.special.logit(lambda_param) 
    lambda_param = scipy.special.expit(logit_lambda)
    one_minus_lambda_param = scipy.special.expit(-logit_lambda)
    loglikelihood_old = 0

    for e in range(n_epochs):
        for i_b, batch in enumerate(data.batches):
            ##Step M
            ### Optimisation for lambda 
            t += 1
            grad_lambda = np.zeros_like(logit_lambda)
            for k in range(K):
                for l in range(L):
                    M =((batch.f0_minus_f1)/(one_minus_lambda_param[k,l]*batch.f1+lambda_param[k,l]*batch.f0))
                    grad_lambda[k,l] = taus_u_true[i_b][:,k]@M@tau_v_true[:,l]

            grad_logit_lambda = grad_lambda * lambda_param * scipy.special.expit(-logit_lambda)

            if optimizer=="adam":
                m_adam = beta1 * m_adam + (1-beta1) * grad_logit_lambda
                v_adam = beta2 * v_adam + (1-beta2) * grad_logit_lambda**2 
                logit_lambda += learning_rate * (m_adam/(1-beta1**t)) / ((v_adam/(1-beta2**t))**.5 + epsilon)
            
            if optimizer=="RMSProp":
                e_RMSProp = beta1 * e_RMSProp + (1 - beta1) * grad_logit_lambda**2 
                logit_lambda += learning_rate * grad_logit_lambda / (e_RMSProp + epsilon)**.5
            
            lambda_param = scipy.special.expit(logit_lambda)
            one_minus_lambda_param = scipy.special.expit(-logit_lambda)

        ### Convergence criterion : (variational-) log-likelihood approximations
        print(f"it: {e:4d}")
        loglikelihood_new = 0
        for i_b, batch in enumerate(data.batches):
            loglikelihood_new += (taus_u_true[i_b] @ np.log(pi_true)[:,None]).sum()
            for k in range(K):
                for l in range(L):
                    loglikelihood_new += taus_u_true[i_b][:,k] @ np.log(one_minus_lambda_param[k,l]*batch.f1+lambda_param[k,l]*batch.f0) @ tau_v_true[:,l]
        loglikelihood_new += (tau_v_true @ np.log(nu_true)[:,None]).sum()           
        print(f"    loglikelihood current={loglikelihood_new:.1f} diff={loglikelihood_new - loglikelihood_old:.1f}")

        if abs(loglikelihood_new-loglikelihood_old) < 0.1:
            print("Convergence !! ")
            break
        loglikelihood_old = loglikelihood_new.copy()

    with open(REP_RES / f'{"likelihood_true_lambda_opt_"}{K}{".pkl"}', "wb") as f:
            pickle.dump(loglikelihood_new, f)
    with open(REP_RES / f'{"lambda_opt_with_true_"}{K}{".pkl"}', "wb") as f:
            pickle.dump(lambda_param, f)
