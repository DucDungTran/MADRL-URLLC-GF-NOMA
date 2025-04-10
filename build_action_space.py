# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:20:01 2022

@author: duc.tran
"""

import itertools as itt
from scipy import special as sp
import numpy as np
# L, K, Kmax = 5,2,2
# P = np.linspace(0.1, 0.9, L)
# n_actions = 1
# for i in range(Kmax): n_actions += L**(i+1)*sp.comb(K, i+1, exact=True)
class action_space():
    def function(L, K, Kmax, P):
        n_actions = 0
        for i in range(Kmax): n_actions += L**(i+1)*sp.comb(K, i+1, exact=True)
        
        A = np.zeros((n_actions, 2*K))
        t = 0
        for k in range(Kmax):
            sc_idx = np.array(list(itt.combinations(np.arange(K),k+1)))
            power_idx = np.array(list(itt.product(np.arange(L), repeat=k+1)))
            len_k = L**(k+1)*len(sc_idx)
            
            sc_k = np.zeros((int(len_k),k+1), dtype=int)
            power_k = np.zeros((len_k,k+1), dtype=int)
            for i in range(len(sc_idx)):
                sc_k[(i*len_k//len(sc_idx)):(i+1)*len_k//len(sc_idx),:] = sc_idx[i]
                power_k[(i*len_k//len(sc_idx)):(i+1)*len_k//len(sc_idx),:] = power_idx

            A[np.arange(t, t+len_k).reshape([len_k,1]),sc_k] = 1
            A[np.arange(t, t+len_k).reshape([len_k,1]),K+sc_k] = P[power_k]
            
            t += len_k
        
        A1 = np.zeros((n_actions, 2))
        if Kmax == 1:
            sc = np.argwhere(A[:, 0:K] == 1)
            A1[:,0] = sc[:,1]
            A1[:,1] = A[sc[:,0], sc[:,1]+K]
            
        return A, A1
    
    def action_space_M_users(ats_one_user, Mm):
        at_size_1u = ats_one_user.shape[1]#action size for one user
        at_size = at_size_1u*Mm#action size for Mm users
        ats_size = len(ats_one_user)**Mm #action space size for Mm users
        ats_idx = np.array(list(itt.product(np.arange(len(ats_one_user)), repeat=Mm)))#selected action indices from ation space of each user
        
        ats = np.zeros((ats_size,at_size))#action space for Mm users
        for i in range(ats_size):
            for j in range(Mm):
                ats[i,j*at_size_1u:(j+1)*at_size_1u] = ats_one_user[ats_idx[i,j],:]
        
        return ats
        
#         A = np.zeros((n_actions, 2*K))
#         A[1:L+1,0] = 1
#         A[1:L+1,2] = P
        
#         A[L+1:2*L+1,1] = 1
#         A[L+1:2*L+1,3] = P
        
#         A[2*L+1:n_actions,0:2] = 1
        
#         A[2*L+1:3*L+1,2] = P[0]
#         A[2*L+1:3*L+1,3] = P
        
#         A[3*L+1:4*L+1,2] = P[1]
#         A[3*L+1:4*L+1,3] = P
        
#         A[4*L+1:5*L+1,2] = P[2]
#         A[4*L+1:5*L+1,3] = P
        
#         A[5*L+1:6*L+1,2] = P[3]
#         A[5*L+1:6*L+1,3] = P
                
#         A[6*L+1:7*L+1,2] = P[4]
#         A[6*L+1:7*L+1,3] = P
        
#         return A       

