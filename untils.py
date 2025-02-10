import numpy as np
from sklearn.gaussian_process.kernels import Matern, RBF
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.linalg import expm, cholesky
from numpy.linalg import inv, det
import scipy
import itertools
import os
from scipy.stats import multivariate_normal
from scipy.special import logsumexp as lse

# Compute Covariance Matrix 
def compute_mean_explicit(A,mu_0,t):
    return expm(t*A).dot(mu_0)
    

def compute_var_explicit(A, L, q, t, Sigma_0):
    d = A.shape[0]
    B = np.block([
        [A, q * L @ L.T],
        [np.zeros((d, d)), -A.T]
    ])
    D = expm(B * t) @ np.block([
        [Sigma_0],
        [np.eye(d)]
    ])
    var = D[0:d] @ inv(D[d:])
    return var


def compute_cov_explicit(A, L, q, s, t,Sigma_0):
    # return the cov matrix of (z(s), z(t))^T
    # assume that s < t
    if s >= t:
        print("We assumed s < t. Swap the fourth and the last parameter.")
        return 
    var_s = compute_var_explicit(A, L, q, s, Sigma_0)
    var_t = compute_var_explicit(A, L, q, t, Sigma_0)
    cov_st = var_s @ expm((t - s) * A).T
    cov =  np.block([
        [var_s, cov_st],
        [cov_st.T, var_t]
    ])
    return cov

# Sampling from the Gaussian Process

def solve_stationary(A,q):
    d = A.shape[0]
    B = np.zeros((d**2,d**2))
    count = 0
    for i in range(d):
        for j in range(d):
            B[count,j*d+i] = A[i,j]
        count += 1

    for i in range(0,d):
        for j in range(i+1,d):
            for k in range(d):
                B[count,k*d+j] = A[i,k]
                B[count,k*d+i] = A[j,k]
            count += 1

    for i in range(0,d):
        for j in range(i+1,d):
            B[count,i*d+j] = 1
            B[count,j*d+i] = -1
            count += 1
    b = np.zeros(d**2)
    b[d-1] = -q/2
    return np.linalg.inv(B)@b


def index_perm(d):
    res = [i for i in range(2*d)]
    res.remove(0)
    res.remove(d)
    return [0, d] + res

# compute conditional covariances for each time step and each pair of (x_t, x_{t+1})(T-1, N, N, d-1, d-1)

def compute_conditional_cov(COVs_X, COVs_Y, COVs_XY, epsilon):
    ## return the conditional covariance matrix for (y_t, y_{t+1}) | (x_t, x_{t+1}) for t = 1, ..., T-1
    ## the shape of cond_COVs is (T-1, d-1, d-1)
    cond_COVs = epsilon * (COVs_Y - COVs_XY.transpose((0, 2, 1)) @ np.linalg.pinv(COVs_X) @ COVs_XY)
    
    return (cond_COVs + cond_COVs.transpose(0, 2, 1)) / 2

def compute_conditional_mean(COVs_X, COVs_XY, D, d):
    ## return the conditional mean for (y_t, y_{t+1}) | (x_t, x_{t+1}) for t = 1, ..., T-1
    ## the shape of cond_means is ( T-1, N, N, 2*(d-1) )
    all_matchings = X_all_matchings(D)
    ## reshape all_matchings to (T-1, 2, N*N)
    T,N = D.shape[0],D.shape[1]
    all_matchings = all_matchings.transpose((0, 3, 1, 2)).reshape(T-1, 2, N*N)
    cond_means = (COVs_XY.transpose((0, 2, 1)) @ inv(COVs_X)) @ all_matchings  # .shape = (T-1, 2*(d-1), N*N)
    cond_means = cond_means.transpose((0, 2, 1)).reshape(T-1, N, N, 2*(d-1))
    
    return cond_means

def X_all_matchings(D):
    ## D is a (T, N) array
    ## returns a (T-1, N, N, 2) tensor 
    T,N = D.shape[0],D.shape[1]
    res = np.zeros((T-1, N, N, 2))
    for i in range(D.shape[0]-1):
        for j in range(N):
            for k in range(N):
                res[i, j, k, :] = np.array([D[i, j], D[i+1, k]])
    return res



def data_generate(d_data,mu_0,sigma_0,dt,ddt,T,N,A_data,L_data,q_gen,n):
    Z = np.zeros((T+1,d_data,N))
    Z_i = np.random.multivariate_normal(mu_0,sigma_0,n).T
    #Z_i[0,:] = np.random.normal(0,1,n)
    Z[0,:,:] = Z_i
    Z_c = np.zeros((T*(int(dt/ddt))+1,d_data,N))
    Z_c[0,:,:] = Z_i
    for i in range(1,T*(int(dt/ddt))+1):
        bw_incre = L_data.reshape(-1,1).dot((np.sqrt(q_gen)*np.random.normal(0,1,n)).reshape(1,-1))*np.sqrt(ddt)
        drift_incre = A_data.dot(Z_i)*ddt
        Z_i += (bw_incre+drift_incre)
        Z_c[i,:,:] = Z_i
        if i%(int(dt//ddt)) == 0:
            Z[i//(int(dt//ddt)),:,:] = Z_i
    D_ob = Z[:,0,:].squeeze()
    return Z_c,D_ob



def solve_stationary(A,q):
    d = A.shape[0]
    B = np.zeros((d**2,d**2))
    count = 0
    for i in range(d):
        for j in range(d):
            B[count,j*d+i] = A[i,j]
        count += 1

    for i in range(0,d):
        for j in range(i+1,d):
            for k in range(d):
                B[count,k*d+j] = A[i,k]
                B[count,k*d+i] = A[j,k]
            count += 1

    for i in range(0,d):
        for j in range(i+1,d):
            B[count,i*d+j] = 1
            B[count,j*d+i] = -1
            count += 1
    b = np.zeros(d**2)
    b[d-1] = -q/2
    return np.linalg.inv(B)@b      
        

def wave_pdf(T,N,total_indices,d,cov,cond_cov,mean_list,cond_cov_margin,mean_margin_list,save_dir,s):
    vars_y = np.diag(cov[0][1:d,1:d]) # (T-1, 2*(d-1))
    scale = 3 * np.sqrt(vars_y) # (T-1, 2*(d-1))
    volume = np.prod(np.sqrt(scale*2))**2
    ## we have y_over_s.shape = Y_pairs_sample.shape = (T-1, N, N, 2*(d-1))
    Kp = np.prod(total_indices)
    
    phi_phi_pdf = np.zeros((T,N,N,Kp,Kp))
    phi_phi_pdf = phi_phi_pdf.astype('float')
    psi_psi = np.zeros((T,N,Kp))
    
    v_indices = np.array([p for p in itertools.product(*[[i for i in range(total_indices[j])] for j in range(len(total_indices))])])
    t_map_v_indices = (v_indices/total_indices - 1/2)*(2*scale)
    t1_map_v_indices = (v_indices/total_indices - 1/2)*(2*scale)
    eva_array_back = np.array([i for i in itertools.product(t_map_v_indices, t_map_v_indices)]).reshape((len(v_indices)**2,-1))
    eva_array_back_m = t_map_v_indices.reshape((len(v_indices),-1))
    t_map_v_indices = ((v_indices+1)/total_indices - 1/2)*(2*scale)
    t1_map_v_indices = ((v_indices+1)/total_indices - 1/2)*(2*scale)
    eva_array_for = np.array([i for i in itertools.product(t_map_v_indices, t_map_v_indices)]).reshape((len(v_indices)**2,-1))
    eva_array_for_m = t_map_v_indices.reshape((len(v_indices),-1))
    for i in range(T):
        #if os.path.exists(save_dir+'/_{}.npz'.format(i)):
           # print('c,skip')
            #continue
        j_eva_array_for = (eva_array_for.reshape((1,Kp**2,2*d-2)) - mean_list[i].reshape((N**2,1,2*d-2))).reshape((-1,2*d-2))
        j_eva_array_back = (eva_array_back.reshape((1,Kp**2,2*d-2)) - mean_list[i].reshape((N**2,1,2*d-2))).reshape((-1,2*d-2))
        j_eva_array_for_m = (eva_array_for_m.reshape((1,Kp,d-1)) - mean_margin_list[i].reshape((N,1,d-1))).reshape((-1,d-1))
        j_eva_array_back_m = (eva_array_back_m.reshape((1,Kp,d-1)) - mean_margin_list[i].reshape((N,1,d-1))).reshape((-1,d-1))
        #grid_volume = np.prod(j_eva_array_for[0,:] - j_eva_array_back[0,:])
        phi_part_list =[]
        for r in np.linspace(0,1,s+2)[1:-1]:
            phi_part = wave_phi_eva_ex_pdf(scale,j_eva_array_for,j_eva_array_back,cov = cond_cov[i], mean = np.zeros(2*d-2), total_indices = total_indices, len_v=len(v_indices),N=N,r=r) - wave_phi_eva_ex_margin_pdf(j_eva_array_for_m,j_eva_array_back_m,cov = cond_cov_margin[i], mean = np.zeros(d-1),  len_v=len(v_indices),N=N,r=r).reshape((N,1,Kp,1))
            phi_part_list.append(phi_part)
        phi_phi_pdf[i,:,:,:,:] = lse(np.stack(phi_part_list,axis=0),axis=0)
    return phi_phi_pdf - np.log(s)

def condition_xx(D,Cov,d):
    T = D.shape[0]-1
    N = D.shape[1]
    Cond_xx = np.zeros((T,N,N))
    for i in range(T):
        Cov_xx = Cov[i][d,d] - Cov[i][0,d]**2/Cov[i][0,0]
        for j in range(N):
            x_t = D[i,j]
            cond_mean = Cov[i][0,d]/Cov[i][0,0]*x_t
            Cond_xx[i,j,:] = multivariate_normal.logpdf(D[i+1,:],mean=cond_mean,cov=Cov_xx)
    return Cond_xx

def wave_phi_eva_ex_margin_pdf(eva_array_for,eva_array_back,cov,mean,len_v,N,r):
    eva_array = multivariate_normal.logpdf(x=(r*eva_array_for+(1-r)*eva_array_back),mean = mean, cov = cov, allow_singular=True)
    return eva_array.reshape((N,len_v))



def wave_phi_eva_ex_pdf(scale_t,eva_array_for,eva_array_back,cov,mean,total_indices,len_v,N,r):
        
    eva_array = multivariate_normal.logpdf(x=(r*eva_array_for+(1-r)*eva_array_back),mean = mean, cov = cov)
    #eva_array1 = multivariate_normal.logpdf(x=eva_array_for,mean = mean, cov = cov)
    #eva_array2 = multivariate_normal.logpdf(x=eva_array_back,mean = mean, cov = cov)
    #eva_array = np.minimum.reduce([eva_array3,eva_array1,eva_array2])
    return eva_array.reshape((N,N,len_v,len_v))


def base_convert(i, b, data_D):
    result = []
    while i > 0:
            result.insert(0, i % b)
            i = i // b
    if len(result) < data_D:
        result = [0]*(data_D-len(result)) + result
    return result

def sample_y(lower,upper,trial,cond_mean,cond_cov):
    d = cond_cov.shape[0]
    samples = np.random.multivariate_normal(mean=cond_mean,cov=cond_cov,size=trial)
    test = (samples[:,0] >= lower[0]) & (samples[:,0] <= upper[0])
    for i in range(1,d):
        test = (test & (samples[:,i] >= lower[i]) & (samples[:,i] <= upper[i]))
    if np.max(test) == 1:
        return samples[np.argmax(test)]
    else:
        print('not found')
        return (lower + upper) /2

def draw_tra(z_list,dt,npts,sigma_0_list,d,A,L,q_list,data_D,T):
    total_trajectory_list = []
    ddt = dt/(npts+1)
    for dim in range(data_D):
        cov_m = np.zeros((3*d,3*d))
        cov_m[0:2*d,0:2*d] = compute_cov_explicit(A, L, q_list[dim], 0, ddt,sigma_0_list[dim])
        cov_m[2*d:3*d,2*d:3*d] = sigma_0_list[dim]
        trajectory_list = []
        for i in range(T):
            z_s = z_list[i,:,dim]
            z_e = z_list[i+1,:,dim]
            t_0 = 0
            trajectory_list.append(z_s+0)
            for n in range(npts):
                z_known = np.concatenate([z_s,z_e],axis=0) + 0
                cov_block_13 = compute_cov_explicit(A, L, q_list[dim], t_0, dt,sigma_0_list[dim])[0:d,d:2*d]
                cov_block_23 = compute_cov_explicit(A, L, q_list[dim], t_0+ddt, dt,sigma_0_list[dim])[0:d,d:2*d]
                cov_m[0:d,2*d:3*d] = cov_block_13
                cov_m[2*d:3*d,0:d] = cov_block_13.T
                cov_m[d:2*d,2*d:3*d] = cov_block_23
                cov_m[2*d:3*d,d:2*d] = cov_block_23.T
                cov11 = cov_m[d:2*d,d:2*d]
                cov12 = np.block([[cov_m[d:2*d,0:d],cov_m[d:2*d,2*d:3*d]]])
                cov22 = np.block([[cov_m[0:d,0:d],cov_m[0:d,2*d:3*d]],[cov_m[2*d:3*d,0:d],cov_m[2*d:3*d,2*d:3*d]]])
                cond_mean = cov12 @ np.linalg.pinv(cov22) @ z_known
                cond_cov = cov11 - cov12 @ np.linalg.pinv(cov22) @ cov12.T
                z_s = np.random.multivariate_normal(mean=cond_mean,cov=cond_cov).reshape(-1)
                trajectory_list.append(z_s+0)
                t_0 += ddt
        trajectory_list.append(z_e+0)
        total_trajectory_list.append(trajectory_list)
    return total_trajectory_list
    
    
    