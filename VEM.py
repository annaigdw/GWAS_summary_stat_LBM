import pathlib
import pickle
import numpy as np
import warnings

import sklearn.metrics
import scipy.special
from tqdm import tqdm
import copy

from import_data_for_batch import LoadableData, BatchData


def _compute_loglikelihood(data:LoadableData, taus_u:list, tau_v: np.ndarray, pi: np.ndarray, nu: np.ndarray,lambda_param: np.ndarray, one_minus_lambda_param: np.ndarray):
    K = taus_u[0].shape[1]
    L = tau_v.shape[1]
    n_batches = len(data.batches)

    loglikelihood = 0
    for i_b, batch in tqdm(enumerate(data.batches),desc="Likelihood ",total=n_batches):
        loglikelihood += (taus_u[i_b] @ np.log(pi)[:,None]).sum()
        for k in range(K):
            for l in range(L):
                loglikelihood += taus_u[i_b][:,k] @ np.log(one_minus_lambda_param[k,l]*batch.f1+lambda_param[k,l]*batch.f0) @ tau_v[:,l]
    loglikelihood += (tau_v @ np.log(nu)[:,None]).sum()           
    return (loglikelihood)


def _compute_ARI(taus_u: list, tau_v:np.ndarray, tau_u_true:np.ndarray, tau_v_true:np.ndarray, rows_index_clusters:np.ndarray):
    rows_cluster = np.where((tau_u_true[:, rows_index_clusters] == 1).any(axis=1))[0]
    reordered_labels_tau_u = [
        label 
        for _,_,label in sorted(
        (
            (i_batch, i_in_batch, label) 
            for i_batch, t in enumerate(taus_u)
            for i_in_batch,label in enumerate(t.argmax(axis=1))
        ),
        key= lambda x: (x[1], x[0])
    )
    ]
    ARI_tau_u = sklearn.metrics.adjusted_rand_score(reordered_labels_tau_u, np.argmax(tau_u_true,axis=1))
    ARI_tau_u_cluster = sklearn.metrics.adjusted_rand_score(np.array(reordered_labels_tau_u)[rows_cluster], np.argmax(tau_u_true[rows_cluster,:],axis=1))
    ARI_tau_v = sklearn.metrics.adjusted_rand_score(np.argmax(tau_v,axis=1), np.argmax(tau_v_true,axis=1))

    print(f"    Score ARI tau_u/u_cluster/v: {ARI_tau_u:.3f} {ARI_tau_u_cluster:.3f} {ARI_tau_v:.3f}")
    return (ARI_tau_u,ARI_tau_u_cluster,ARI_tau_v)

def perform_VEM(data:LoadableData,
                taus_u: list, tau_v: np.ndarray,
                pi: np.ndarray, nu: np.ndarray, lambda_param: np.ndarray,
                loglikelihood_crit: list, diff_taus_u: list, diff_tau_v: list, entropy_vec: list, 
                rows_group_splitting: list, cols_group_splitting: list, 
                ARI_tau_u: list, ARI_tau_u_cluster:list, ARI_tau_v: list,
                learning_rate: float, optimizer: str, max_n_epochs: int,
                basefilename: str, results_path:pathlib.Path,
                simulated_data: bool):
    
    n = sum(b.f0.shape[0] for b in data.batches)
    m = data.batches[0].f0.shape[1]
    K = taus_u[0].shape[1]
    L = tau_v.shape[1]
    n_batches = len(data.batches)

    logit_lambda = scipy.special.logit(lambda_param) 
    one_minus_lambda_param = scipy.special.expit(-logit_lambda)
    
    if simulated_data:
        ## Load the true classification for ARI computation 
        basefilename_data=f"K{K}_L{L}_simadam2"
        with open(results_path / f'tau_u_true_{basefilename_data}.pkl', "rb") as f:
            tau_u_true = pickle.load(f)

        with open(results_path / f'tau_v_true_{basefilename_data}.pkl', "rb") as f:
            tau_v_true = pickle.load(f)

        with open(results_path / f'rows_index_clusters_{basefilename_data}.pkl', "rb") as f:
            rows_index_clusters =pickle.load(f)

        ARI = _compute_ARI(taus_u, tau_v, tau_u_true, tau_v_true,rows_index_clusters)
        ARI_tau_u.append(ARI[0])
        ARI_tau_u_cluster.append(ARI[1])
        ARI_tau_v.append(ARI[2])

    if len(loglikelihood_crit)==0:
        ### Convergence criterion : (variational-) log-likelihood approximation
        loglikelihood_crit.append(_compute_loglikelihood(data,taus_u,tau_v,pi, nu, lambda_param, one_minus_lambda_param))
        last_epoch_group_split = 15
        
    else:
        last_epoch_group_split = -5

    
    ##Hyper parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    ## Initialisation
    t=0
    m_adam = 0
    v_adam = 0
    e_RMSProp = 0

   
    loglikelihood_old = loglikelihood_crit[-1].copy()
    old_log_taus_u =[np.log(taus_u[i_b]) for i_b in range(n_batches)]

    stop = False
    # Convert RuntimeWarnings to errors
    warnings.simplefilter("error", RuntimeWarning)
    try:
        ## Epoch
        first_E_is_done = False
        for e in range(max_n_epochs):
            only_M = (e <= last_epoch_group_split+4)
            old_taus_u = copy.deepcopy(taus_u)
            old_tau_v = tau_v.copy()
            print(f"epoch: {e:4d}")
            if only_M:
                print("    only M")
 
            for i_b, batch in tqdm(enumerate(data.batches),desc=f'{"Epoch "}{e}',total=n_batches):
                n_batch = batch.f0.shape[0]
            
                ##Step (V-)E
                if not only_M:
                    log_tau_u_b = np.ones((n_batch,1))*np.log(pi[None,:])

                    for k in range(K):
                        for l in range(L):
                            log_tau_u_b[:,k] += np.log(one_minus_lambda_param[k,l]*batch.f1 +lambda_param[k,l]*batch.f0) @ tau_v[:,l]
        
                    log_tau_u_b -= log_tau_u_b.max(axis=1)[:,None]
                    log_tau_u_b = np.maximum(-10,log_tau_u_b)
                    old_log_taus_u[i_b] = log_tau_u_b.copy()
        
                    tau_u_b = np.exp(log_tau_u_b )
                    tau_u_b/= tau_u_b.sum(axis=1)[:,None]
                    old_tau_u_b = taus_u[i_b].copy()
        
                    taus_u[i_b] = tau_u_b.copy()
        
                    if not first_E_is_done:
                        if i_b==0:
                            log_tau_v = np.zeros_like(tau_v)
                        for l in range(L):
                            for k in range(K):
                                log_tau_v[:,l] += np.log(one_minus_lambda_param[k,l]*batch.f1+lambda_param[k,l]*batch.f0).T @ taus_u[i_b][:,k]
                    else:
                        log_tau_v_new = log_tau_v_old + (np.ones((m,1))*(np.log(nu[None,:])-np.log(nu_old[None,:])))
                        for l in range(L):
                            for k in range(K):
                                log_tau_v_new[:,l] += (np.log(one_minus_lambda_param[k,l]*batch.f1+lambda_param[k,l]*batch.f0).T @ (taus_u[i_b][:,k] - old_tau_u_b[:,k]))
        
                        log_tau_v_new -= log_tau_v_new.max(axis=1)[:,None]
                        log_tau_v_old = log_tau_v_new.copy()
                        log_tau_v_new = np.maximum(-10,log_tau_v_new)
                        tau_v_new = np.exp(log_tau_v_new)
                        tau_v_new /= tau_v_new.sum(axis=1)[:,None]

                        tau_v = tau_v_new.copy()
        
                ##Step M
                if not only_M and first_E_is_done:
                    pi += (taus_u[i_b] - old_tau_u_b).sum(axis=0)/n
                    pi = np.maximum(1e-8,pi)
                    pi /= sum(pi)
                    nu_old = nu.copy()
                    nu = tau_v.mean(axis=0)
                    nu= np.maximum(1e-8,nu)
                    nu /= sum(nu) 
        
                ### Optimisation for lambda 
                t += 1
                grad_lambda = np.zeros_like(lambda_param)
                for k in range(K):
                    for l in range(L):
                        M =((batch.f0_minus_f1)/(one_minus_lambda_param[k,l]*batch.f1+lambda_param[k,l]*batch.f0))
                        grad_lambda[k,l] = taus_u[i_b][:,k]@M@tau_v[:,l]
        
                grad_logit_lambda = grad_lambda * lambda_param * one_minus_lambda_param
        
                if optimizer=="adam":
                    m_adam = beta1 * m_adam + (1-beta1) * grad_logit_lambda
                    v_adam = beta2 * v_adam + (1-beta2) * grad_logit_lambda**2 
                    logit_lambda += learning_rate  * (m_adam/(1-beta1**t)) / ((v_adam/(1-beta2**t))**.5 + epsilon)
                
                if optimizer=="RMSProp":
                    e_RMSProp = beta1 * e_RMSProp + (1 - beta1) * grad_logit_lambda**2 
                    logit_lambda += learning_rate * grad_logit_lambda / (e_RMSProp + epsilon)**.5
            
                lambda_param = scipy.special.expit(logit_lambda)
                one_minus_lambda_param = scipy.special.expit(-logit_lambda)
        
            if not only_M and not first_E_is_done:
                log_tau_v += np.ones((m,1))*np.log(nu[None,:])
                log_tau_v -= log_tau_v.max(axis=1)[:,None]
                log_tau_v_old = log_tau_v.copy()
                log_tau_v = np.maximum(-10,log_tau_v)
                tau_v = np.exp(log_tau_v)
                tau_v/= tau_v.sum(axis=1)[:,None]
            
                nu_old = nu.copy()
                nu = tau_v.mean(axis=0)
                pi = np.array([tau_u.sum(axis=0) for tau_u in taus_u]).sum(axis=0)/n
            
                first_E_is_done = True
                
            if not first_E_is_done:
                with open(results_path / f'lambda_{basefilename}_init.pkl', "wb") as f:
                    pickle.dump(lambda_param, f)

            if not only_M :
                ### Convergence criterion : log-likelihood and ELBO
                loglikelihood_new = _compute_loglikelihood(data, taus_u, tau_v, pi, nu, lambda_param, one_minus_lambda_param)          
                loglikelihood_crit.append(loglikelihood_new)
                print(f"    loglikelihood current={loglikelihood_new:.1f} diff={loglikelihood_new - loglikelihood_old:.1f}")
        
                entropy = -sum(np.log(tau_u).reshape(-1)@tau_u.reshape(-1) for tau_u in taus_u)
                entropy += -(np.log(tau_v).reshape(-1)@tau_v.reshape(-1))
                entropy_vec.append(entropy)
                print(f"    ELBO current= {loglikelihood_new+entropy_vec[-1]:.1f} diff={loglikelihood_new+entropy_vec[-1] - (loglikelihood_old+entropy_vec[-2]) if len(entropy_vec)>=2 else 0:.1f}")

                ### ARI computation
                if simulated_data:
                    ARI = _compute_ARI(taus_u, tau_v, tau_u_true, tau_v_true,rows_index_clusters)
                    ARI_tau_u.append(ARI[0])
                    ARI_tau_u_cluster.append(ARI[1])
                    ARI_tau_v.append(ARI[2])

                ### Cluster changes
                diff_taus_u.append(sum((np.argmax(np.vstack(taus_u),axis=1)- np.argmax(np.vstack(old_taus_u),axis=1))!=0))
                diff_tau_v.append(sum((np.argmax(tau_v,axis=1)- np.argmax(old_tau_v,axis=1))!=0))
                print("    Changement de grps tau_u/v: ", diff_taus_u[-1], diff_tau_v[-1])
                
                ### Save the results
                with open(results_path / f'taus_u_{basefilename}.pkl', "wb") as f:
                    pickle.dump(taus_u, f)
        
                with open(results_path / f'taus_v_{basefilename}.pkl', "wb") as f:
                    pickle.dump(tau_v, f)
        
                with open(results_path / f'lambda_{basefilename}.pkl', "wb") as f:
                    pickle.dump(lambda_param, f)
        
                with open(results_path / f'pi_{basefilename}.pkl', "wb") as f:
                    pickle.dump(pi, f)
        
                with open(results_path / f'nu_{basefilename}.pkl', "wb") as f:
                    pickle.dump(nu, f)
        
                with open(results_path / f'likelihood_{basefilename}.pkl', "wb") as f:
                    pickle.dump(loglikelihood_crit, f)
        
                if simulated_data:
                    with open(results_path / f'ARI_tau_u_{basefilename}.pkl', "wb") as f:
                        pickle.dump(ARI_tau_u, f)
                    with open(results_path / f'ARI_tau_u_cluster_{basefilename}.pkl', "wb") as f:
                        pickle.dump(ARI_tau_u_cluster, f)
                    with open(results_path / f'ARI_tau_v_{basefilename}.pkl', "wb") as f:
                        pickle.dump(ARI_tau_v, f)
                
                with open(results_path / f'changement_tau_u_{basefilename}.pkl', "wb") as f:
                    pickle.dump(diff_taus_u, f)
        
                with open(results_path/ f'changement_tau_v_{basefilename}.pkl', "wb") as f:
                    pickle.dump(diff_tau_v, f)
        
                with open(results_path / f'entropy_{basefilename}.pkl', "wb") as f:
                    pickle.dump(entropy_vec, f)
                
                ### did the algorithm converge ? 
                if abs(loglikelihood_new+entropy_vec[-1] - (loglikelihood_old+entropy_vec[-2]) if len(entropy_vec)>=2 else 1) < 0.1: 
                    print("Convergence !! ")
                    break
                if np.isnan(loglikelihood_new):
                    print("The algorithm do not converge... ")
                    break
                loglikelihood_old = loglikelihood_new.copy()

                ### is there empty groups ? 
                while pi.min()<1/n:
                    row_class_to_replace = np.argmin(pi)
                    row_class_to_split = np.argmax(pi)
                    print(f"    row groups splitting: {row_class_to_replace, row_class_to_split}")
                    for i_b in range(n_batches):
                        nelem,_ = taus_u[i_b].shape
                        split = np.random.beta(0.25,0.25,size=nelem)
                        taus_u[i_b][:,row_class_to_replace]+=split*taus_u[i_b][:,row_class_to_split]
                        taus_u[i_b][:,row_class_to_split]*=(1-split)
                        taus_u[i_b] /= taus_u[i_b].sum(axis=1, keepdims=True)
                    lambda_param[row_class_to_replace,:] = lambda_param[row_class_to_split,:]
                    one_minus_lambda_param[row_class_to_replace,:] = one_minus_lambda_param[row_class_to_split,:]
                    logit_lambda[row_class_to_replace,:] = logit_lambda[row_class_to_split,:] 
                    pi[row_class_to_replace] += pi[row_class_to_split]/2
                    pi[row_class_to_split] /= 2
                    last_epoch_group_split = e
                    rows_group_splitting.append(e)
                    with open(results_path / f'rows_group_splitting_{basefilename}.pkl', "wb") as f:
                        pickle.dump(rows_group_splitting, f)
                while nu.min()<1/m:
                    col_class_to_replace = np.argmin(nu)
                    col_class_to_split = np.argmax(nu)
                    cols_in_class_to_split = np.argmax(tau_v,axis=1)==col_class_to_split
                    n_cols_in_class_to_split = cols_in_class_to_split.sum()
                    gramm = np.zeros((n_cols_in_class_to_split,)*2)
                    for batch in data.batches:
                        f0_copy = batch.f0.copy()
                        f0_copy[f0_copy==0] = np.min(f0_copy[f0_copy> 0])
                        f1_copy = batch.f1.copy()
                        f1_copy[f1_copy==0] = np.min(f1_copy[f1_copy> 0])
                        log_ratio_f1_f0_batch = (np.log(f1_copy[:,cols_in_class_to_split]) - np.log(f0_copy[:,cols_in_class_to_split]))
                        log_ratio_f1_f0_batch -= log_ratio_f1_f0_batch.mean(axis=1)[:,None]
                        gramm += log_ratio_f1_f0_batch.T@log_ratio_f1_f0_batch
                    _, eigenvectors = scipy.linalg.eigh(gramm, subset_by_index = (n_cols_in_class_to_split-1,)*2)
                    eigenvector = eigenvectors[:,0]
                    threshold = np.median(eigenvector)
                    split = .5*np.ones(m)
                    for i,i_orig in enumerate(np.where(cols_in_class_to_split)[0]):
                        split[i_orig] = 1-1e-2 if eigenvector[i]<threshold else 1e-2

                    print(f"    col groups splitting: {col_class_to_replace, col_class_to_split}")
                    tau_v[:,col_class_to_replace]+=split*tau_v[:,col_class_to_split]
                    tau_v[:,col_class_to_split]*=(1-split)
                    tau_v /= tau_v.sum(axis=1, keepdims=True)
                    log_tau_v_old[:,col_class_to_replace]= np.log(tau_v[:,col_class_to_replace])
                    log_tau_v_old[:,col_class_to_split] = np.log(tau_v[:,col_class_to_split])
                    lambda_param[:, col_class_to_replace] = lambda_param[:, col_class_to_split]
                    one_minus_lambda_param[:, col_class_to_replace] = one_minus_lambda_param[:, col_class_to_split]
                    logit_lambda[:, col_class_to_replace] = logit_lambda[:, col_class_to_split]
                    nu[col_class_to_replace] += nu[col_class_to_split]/2
                    nu[col_class_to_split] /= 2
                    last_epoch_group_split = e
                    cols_group_splitting.append(e)
                    with open(results_path / f'cols_group_splitting_{basefilename}.pkl', "wb") as f:
                        pickle.dump(cols_group_splitting, f)

    except RuntimeError as e:
        print(f"RuntimeWarning encountered: {e}")
        # Stop the code explicitly if necessary
        raise SystemExit("Exiting due to divide by zero encountered in log.") 
    

    if e == max_n_epochs-1:
        print("Warning: The algorithm does not converge within the maximum epochs.")





