"""
Parameters for URLLC-enabled grant-free NOMA systems:

Mm: number of users
K: number of sub-channels
nu: 5G numerology index
W: sub-channel bandwidth
sigma2: noise power
Pmax: maximum transmission power
theta: path-loss exponent
Pl: number of power levels
Dmax: latency threshold
bler_th: decoding error threshold
nb: packet size
Ruth: target rate
Pc: circuit power consumption
dm: distance from users to base station
Hm: wireless channel coefficient following Rayleigh distribution

"""

import numpy as np
from tqdm import tqdm
from scipy import special as sp #used to calculate qfuncinv
import matplotlib.pyplot as plt
import time

from D3QN_model import D3QN

from NOMA_env import ENVIRONMENT
from build_action_space import action_space as ats


#---------- System parameters ------------
Mm = 4
K = 2
nu = 2
W = 2**nu*180*10**3

FdB, N0dBm = 6, -174
F, N0 = 10**(FdB/10), 10**((N0dBm - 30)/10)
sigma2 = F*N0*W

PmaxdBm = 23
Pmax = 10**((PmaxdBm-30)/10)
theta = 3
Pl = 7
Pm = np.linspace(Pmax/Pl, Pmax, Pl)

Dmax = 1*10**(-3)
bler_th = 10**(-5)
nb = 32*8
guth = 2**(nb/(Dmax*W) + np.sqrt(2)*sp.erfinv(1-2*bler_th)/(np.log(2)*np.sqrt(Dmax*W))) - 1
vuth = np.log2(np.exp(1))**2*(1 -  1/(1 + guth)**2)
Ruth = W*(np.log2(1 + guth) - np.sqrt(vuth/(Dmax*W))*np.sqrt(2)*sp.erfinv(1-2*bler_th))#rate threshold for URLLC user over one SC
Pc = 0.05

dm = np.loadtxt('Channel_Distance\distance_Radius_500_Mm_4.dat')
dmk = np.zeros(Mm*K)
for k in range(Mm): dmk[k*K:(k+1)*K] = dm[k]

#---------- Channel ---------
Hm = np.loadtxt('Channel_Distance\Rayleigh_Channel_URLLC_60000samples_Mm_4_K_2.dat').view(complex)
Hm2 = abs(Hm[:,0:Mm*K])**2
Gm = Hm2/dmk**theta

#===========================================================
#---------- Learning parameters -------------

episodes = 500 #number of episode for training
max_steps = 100 #number of steps in each episode
gamma = 0.9 # Discount factor for future rewards
tau = 0.01 # Used to update target networks
alpha = 0.001#10**(-np.arange(2, 7, dtype=float))
epsilon = 1
epsilon_min = 0.001
epsilon_decay = 0.9997
target_update_freq = 50 #target updating frequency
memory_size = 500 # max size of memory
batch_size = 32

n_actions = K*Pl

action_space_binary, action_space_scidx = ats.function(Pl, K, 1, Pm)
# print(action_space_scidx)

id = 0
state_size = 2

env = ENVIRONMENT(Gm, Mm, Pc, Pmax, K, W, Dmax, bler_th, Ruth,\
                         sigma2, state_size, action_space_binary, action_space_scidx, 'NOMA')
agents = D3QN(Mm, state_size, n_actions, tau, alpha, gamma, epsilon, epsilon_min, epsilon_decay, target_update_freq,\
             memory_size, batch_size, 'Hard')

ep_reward_list = []
actions_ep = []
P_ep = []
t_ep_list = []
t1 = time.time()
for i in tqdm(range(episodes)):     
    states = env.reset()
    step_reward = []
    actions_step = []
    P_step = []
    for j in range(max_steps):
        actions = agents.choose_action(states)
        actions_step.append(actions)
        
        rewards, nstates, P, R = env.step(actions)
        P_step.append(P)
        
        agents.store_experience(states, actions, rewards, nstates)
        
        agents.learn() 

        step_reward.append(rewards[0])
        states = nstates
        if (i*max_steps + j + 1)%target_update_freq == 0:
            agents.update_target_parameters()
            
    actions_ep.append(np.array(actions_step))
    P_ep.append(np.array(P_step))
    ep_reward = np.mean(step_reward)
    ep_reward_list.append(ep_reward)

    print(f"Ep {i} => Reward: {ep_reward}")
    t_ep_list.append(time.time() - t1)
    
################ Testing ##################
env1 = ENVIRONMENT(Gm[49999:], Mm, Pc, Pmax, K, W, Dmax, bler_th, Ruth,\
                          sigma2, state_size, action_space_binary, action_space_scidx)
test_num = 100
ep_reward_t = []
for ii in range(test_num):
    states_t = env1.reset()
    step_reward_t = []
    for jj in range(max_steps):
        actions_t = agents.test_action(states_t)
        rewards_t, nstates_t, P_t, R_t = env1.step(actions_t)
        step_reward_t.append(rewards_t[0])
        states_t = nstates_t
    ep_reward_t1 = np.mean(step_reward_t)
    ep_reward_t.append(ep_reward_t1)
print(np.mean(ep_reward_t))

plt.plot(ep_reward_list)
plt.plot(ep_reward_t)
plt.legend(['Training', 'Testing'], loc = 'best')
plt.xlabel("Episode")
plt.ylabel("Average Rewards")
plt.show()