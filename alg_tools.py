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

def sample_yt1_ini_sample_log_md_sample(joint_d_t,n,s):
    Kp = joint_d_t.shape[-2]
    y_t0_index = np.zeros(n*s)
    condition_matrix = lse(joint_d_t,axis=(1,3)) + 0
    count = 0
    for i in range(n):
        m_ti = condition_matrix[i,:] + 0
        for j in range(s):
            #yti_index = np.random.choice([i for i in range(Kp)],p=np.exp(m_ti-lse(m_ti)))
            yti_index = np.random.choice([i for i in range(Kp)],p=np.exp(m_ti-lse(m_ti)))
            y_t0_index[count] = yti_index
            count += 1
        
            

    return y_t0_index
    
def message_passing(phi_phi_list,data_D,T,N,total_indices,max_iter,save_dir,save_freq,error_thre):
    
    #Initialization
    Kp_shape = [np.prod(i) for i in total_indices]
    Kp = [np.prod(total_indices[dim]) for dim in range(data_D)]
    Kp = Kp[0]
    shape = [T,N] + Kp_shape
    C_alpha_left = np.log(1/np.sqrt(Kp**data_D)) * np.ones(shape)
    C_alpha_right = np.log(1/np.sqrt(Kp**data_D)) * np.ones(shape)
    
    best_error = np.infty
    C_alpha_left_new = np.log(1/np.sqrt(Kp**data_D)) * np.ones(shape)
    C_alpha_right_new = np.log(1/np.sqrt(Kp**data_D)) * np.ones(shape)
    
    # Gamma_t(xt, xt+1), it has shape (T-1, N, N, 1)
    C_z_left = np.log(1/np.sqrt(Kp**data_D)) * np.ones(shape)
    C_z_right = np.log(1/np.sqrt(Kp**data_D)) * np.ones(shape)
    # Gamma times A_t, it is of shape (T-1, N, N, K**(d-1), K**(d-1))
    Kp_tuples = tuple([1+i for i in range(data_D)])    
    t = 0
    errors = []
    xn_list = ['i','j']
    yn_list = ['k','l','a','b','c']
    ein_sum_n = 'ij,j'    
    for dim in range(data_D):
        ein_sum_n += yn_list[dim]
    ein_sum_n += '->i'
    for dim in range(data_D):
        ein_sum_n += yn_list[dim]
    
    print('Starting Message Passing')
    print(' ')
    t = 0
    errors = []
    while True:
        #n message update
        for i in range(T-1,-1,-1):
            if i == T-1:
                C_z_left[T-1] = np.einsum(ein_sum_n,np.log(1/N)-lse(C_alpha_right[T-1],axis=Kp_tuples).reshape(-1,1) , np.ones([1]+[p for p in Kp_shape]))
            else:
                C_z_left[i] = C_alpha_left_new[i+1]-lse(C_alpha_left_new[i+1]+C_alpha_right[i],axis=Kp_tuples).reshape([-1]+[1 for i in range(data_D)])
            for dim in range(data_D):
                phi_phi = phi_phi_list[dim][i].transpose(0,2,1,3)
                if dim == 0:
                    phi_phi_part =  lse(np.expand_dims(phi_phi,axis=[-j-1 for j in range(data_D-1)])+np.expand_dims(C_z_left[i],axis=(0,1)),axis=3)
                    #shape (N,Kp,N,Kp...)
                else:
                    phi_phi_part = lse(np.expand_dims(phi_phi,axis=[j+1 for j in range(dim)]+[-j-1 for j in range(data_D-dim-1)])+np.expand_dims(phi_phi_part,axis=[1+dim]),axis=3+dim)
            C_alpha_left_new[i] = lse(phi_phi_part,axis=-1)
                    
            
            C_alpha_left_new[i] -= lse(C_alpha_left_new[i]*2)/2    
        
        for i in range(0,T):
            phi_phi = phi_phi_list[dim]
            if i == 0:
                C_z_right[0] =  np.einsum(ein_sum_n,(np.log(1/N)-(lse(C_alpha_left_new[0],axis=Kp_tuples))).reshape(-1,1) ,np.ones([1]+[p for p in Kp_shape]))
            else:
                C_z_right[i] = C_alpha_right_new[i-1]-lse(C_alpha_left_new[i]+C_alpha_right_new[i-1],axis=Kp_tuples).reshape([-1]+[1 for i in range(data_D)])
            for dim in range(data_D):
                phi_phi = phi_phi_list[dim][i].transpose(1,3,0,2)
                if dim == 0:
                    phi_phi_part =  lse(np.expand_dims(phi_phi,axis=[-j-1 for j in range(data_D-1)])+np.expand_dims(C_z_right[i],axis=(0,1)),axis=3)
                else:
                    phi_phi_part = lse(np.expand_dims(phi_phi,axis=[j+1 for j in range(dim)]+[-j-1 for j in range(data_D-dim-1)])+np.expand_dims(phi_phi_part,axis=[1+dim]),axis=3+dim)
                    
            C_alpha_right_new[i] = lse(phi_phi_part,axis=-1)
                
            C_alpha_right_new[i] -= lse(C_alpha_right_new[i]*2)/2
            
        error = np.sum(np.abs(np.exp(C_alpha_right) - np.exp(C_alpha_right_new)))
        errors.append(error)
        print('Norm difference between two consecutive messages at iteration t={}: '.format(t+1),error)
        if error <= best_error:
            print('Smallest change between consecutive messages, saving to '+save_dir+'/message_sample_best.npz')
            np.savez(save_dir+'/message_sample_best.npz',C_alpha_left,C_alpha_right,C_z_left,C_z_right)
            best_error = error+0
            #print(np.sum(np.abs(C_alpha_left - C_alpha_left_new)))
        if error <= error_thre or t==max_iter-1:
            if error <= error_thre:
                print('Stopping criterion met: minimum changes in message!')
            else:
                print('Stopping criterion met: maximum iterations reached!')
            break
        if t%save_freq == 0:
            print('Saving the latest messages to '+save_dir+'/message_sample_new.npz')
            np.savez(save_dir+'/message_sample_new.npz',C_alpha_left,C_alpha_right,C_z_left,C_z_right)
        print(' ')
        C_alpha_left = C_alpha_left_new +0 
        C_alpha_right = C_alpha_right_new + 0
    
        t += 1
    return errors


def generate_trajectories_sample(C_z_right,C_z_left,phi_phi_list,data_D,T,N,Kp_shape,sample,save_traj_dir):
    n = N 
    t=0
    C_z_right_t = C_z_right[t].reshape(N,-1)
    C_z_left_t = C_z_left[t].reshape(N,-1)
    C_z_right_left_t = C_z_right_t.reshape((N,1,-1,1)) + C_z_left_t.reshape((1,N,1,-1))
    Kpp = Kp_shape[0]
    Kp = np.prod(Kp_shape)
    if data_D > 1:
        for dim in range(data_D-1):
            Kpp *= Kp_shape[dim+1]
            if dim == 0:
                phi_phi_expand_t = (np.expand_dims(phi_phi_list[dim][t],axis=(-3,-1))+np.expand_dims(phi_phi_list[dim+1][t],axis=(-4,-2))).reshape(N,N,Kpp,Kpp)
            else:
                phi_phi_expand_t = (np.expand_dims(phi_phi_expand_t,axis=(-3,-1))+np.expand_dims(phi_phi_list[dim+1][t],axis=(-4,-2))).reshape(N,N,Kpp,Kpp)
        joint_d_t = phi_phi_expand_t+C_z_right_left_t
    else:
        joint_d_t = phi_phi_list[0][t]+C_z_right_left_t
    total_yt_index_recorder = np.zeros((sample*n,T+1)).astype('int')
    total_trajectory = np.zeros((sample*n,T+1)).astype('int')
    for tt in range(sample*n):
        total_trajectory[tt,0] = tt//sample
    xt_index = total_trajectory[:,0].astype('int')
    print('Generating velocity for initial observations')
    yt_index = sample_yt1_ini_sample_log_md_sample(joint_d_t,n,sample).astype('int')
    for t in range(T):
        print('Matching observations for time step {}'.format(t))
        phi_phi_expand_t = 0
        joint_d = 0
        C_z_right_left_t = 0
        C_z_right_t = C_z_right[t].reshape(N,-1)
        C_z_left_t = C_z_left[t].reshape(N,-1)
        C_z_right_left_t = C_z_right_t.reshape((N,1,-1,1)) + C_z_left_t.reshape((1,N,1,-1))
        Kpp = Kp_shape[0]
        if data_D > 1:
            for dim in range(data_D-1):
                Kpp *= Kp_shape[dim+1]
                if dim == 0:
                    phi_phi_expand_t = (np.expand_dims(phi_phi_list[dim][t],axis=(-3,-1))+np.expand_dims(phi_phi_list[dim+1][t],axis=(-4,-2))).reshape(N,N,Kpp,Kpp)
                else:
                    phi_phi_expand_t = (np.expand_dims(phi_phi_expand_t,axis=(-3,-1))+np.expand_dims(phi_phi_list[dim+1][t],axis=(-4,-2))).reshape(N,N,Kpp,Kpp)
            joint_d_t = phi_phi_expand_t+C_z_right_left_t
        else:
            joint_d_t = phi_phi_list[0][t]+C_z_right_left_t
        m_t1 = lse(joint_d_t,axis=-1) + 0
        yt1_index = np.zeros(sample*n).astype('int')
        xt1_index = np.zeros(sample*n).astype('int')
        for i in range(sample*n):
            B = m_t1[xt_index[i],:,yt_index[i]] + 0
            B -= lse(B)
            xt1_index[i] = np.random.choice([i for i in range(N)], p = np.exp(B))
            yt1_p = joint_d_t[xt_index[i],xt1_index[i],yt_index[i],:] + 0
            yt1_p -= lse(yt1_p)
            yt1_index[i] = np.random.choice([i for i in range(Kp)], p = np.exp(yt1_p))
        total_yt_index_recorder[:,t+1] = yt1_index + 0
        total_trajectory[:,t+1] = xt1_index + 0
        yt_index = (yt1_index+0).astype('int')
        xt_index = (xt1_index+0).astype('int')
    print('Saving results to {}'.format(save_traj_dir))
    np.savez(save_traj_dir+'/trajectories',np.array(total_trajectory))
    np.savez(save_traj_dir+'/y_trajectories',np.array(total_yt_index_recorder))
    return np.array(total_trajectory),np.array(total_yt_index_recorder)