# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from scipy import special as sp #used to calculate qfuncinv

class ENVIRONMENT():
    def __init__(self, Gm, Mm, Pc, Pmax, n_subchannels, sub_bandwidth, Dmax,\
                 bler_th, Rth, sigma2, state_size, action_space_binary, action_space_scidx, MA):
        """
        self.Gm = Hm2/dm**theta
        Hm2: channel gain
        theta: path-loss exponent
        self.Mm: number of users
        self.Pc: circuit power consumption
        self.Pmax: maximum transmission power
        self.K: number of sub-channels
        self.W: sub-channel bandwidth
        self.Dmax: latency threshold
        self.bler_th = decoding error probability
        self.Rth: Rate threshold
        self.sigma2: Noise power
        
        """
        
        self.Gm = Gm
        self.Mm = Mm
        self.Pc = Pc
        self.Pmax = Pmax
        self.K = n_subchannels
        self.W = sub_bandwidth
        self.Dmax = Dmax
        self.bler_th = bler_th
        self.Rth = Rth
        self.sigma2 = sigma2
        self.cs_index = 0#index of channel sample
        self.state_size = state_size
        self.AS_binary = action_space_binary
        self.AS = action_space_scidx
        self.n_actions = len(self.AS)
        self.MA = MA#multiple access methods (NOMA or OMA)

    def qfuncinv(self, x):
        return np.sqrt(2)*sp.erfinv(1-2*x)
    
    def reset(self):
        states_agents = []
        actions_idx = np.random.randint(0, self.n_actions, size=self.Mm)
        
        #Proposed state
        if (self.state_size == 2):
            for i in range(self.Mm):
                actioni = self.AS[actions_idx[i]]
                statesi = actioni#np.concatenate((actioni, [0]))
                states_agents.append(statesi)      
        #State 2: Achievable rate of all agents over all SCs
        elif (self.state_size == self.Mm*self.K):
            for i in range(self.Mm):
                statesi = np.zeros(self.Mm*self.K)
                states_agents.append(statesi)
                
        return states_agents
        
    def step(self, actions_idx):
        
        actions_sc_power = self.AS[actions_idx]
        actions_sc = actions_sc_power[:,0]
        actions_power = actions_sc_power[:,1]
        
        cGm = np.reshape(self.Gm[self.cs_index,:], [self.Mm, self.K])
        
        Rm = np.zeros(self.K*self.Mm)
        R = np.zeros(self.Mm)
        
        for k in range(self.K):
            check = 0
            users_k = np.argwhere(actions_sc == k)[:,0]
            if len(users_k) != 0:            
                ## Decoding order based on received powers
                cG_k = cGm[users_k,k]
                power_k = actions_power[users_k]
                Pk = cG_k*power_k#received power
                
                if self.MA == 'OMA':
                    gk = Pk/self.sigma2
                    vuk = np.log2(np.exp(1))**2*(1 -  1/(1 + gk)**2)
                    for i in range(len(users_k)):
                        R[users_k[i]] = self.W*(np.log2(1 + gk[i]) - np.sqrt(vuk[i]/(self.Dmax*self.W))*self.qfuncinv(self.bler_th))/len(users_k)
                        if R[users_k[i]] < self.Rth:
                            check += 1
                elif self.MA == 'NOMA':
                    Pk_sort = np.array(sorted(Pk, reverse=True))
                    Pk_sort_idx = np.argsort(-Pk)
                    for i in range(len(users_k)):
                        useri_idx = users_k[Pk_sort_idx[i]]
                        gki = Pk_sort[i]/(np.sum(Pk_sort[(i+1):]) + self.sigma2)
                        vuki = np.log2(np.exp(1))**2*(1 -  1/(1 + gki)**2)
                        R[useri_idx] = self.W*(np.log2(1 + gki) - np.sqrt(vuki/(self.Dmax*self.W))*self.qfuncinv(self.bler_th))
                        Rm[useri_idx*self.K + k] = R[useri_idx]
                        if R[useri_idx] < self.Rth:
                            check += 1
            if (check != 0):
                break
        
        self.cs_index += 1
        
        rewards = np.zeros(self.Mm)
        EE = np.sum(R)/(np.sum(actions_power) + self.Mm*self.Pc)
        R_check = R >= self.Rth
        if (len(np.argwhere(R_check == True)) == self.Mm):
            rewards[:] = EE
        
        nstates = []
        actions_idx = np.array(actions_idx)
        
        # Next states
        # Proposed state
        if (self.state_size == 2):
            for i in range(self.Mm):
                actioni = self.AS[actions_idx[i]]
                statesi = actioni
                nstates.append(statesi)
        # State 2: Achievable rate of all agents over all SCs
        elif (self.state_size == self.Mm*self.K):
            for i in range(self.Mm):
                statesi = Rm
                nstates.append(statesi)
            
        return rewards, nstates, actions_power, R