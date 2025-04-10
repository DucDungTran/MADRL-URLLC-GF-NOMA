# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:20:53 2021

@author: duc.tran
"""

import numpy as np
import random
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

class DDQN():
    def __init__(self, n_agents, state_size, n_actions, tau, alpha, gamma, epsilon, epsilon_min,
                 epsilon_decay, target_update_freq, memory_size, batch_size, technique, mode_update):
        self.n_agents = n_agents
        self.state_size = state_size
        self.n_actions = n_actions # number of actions
        self.tau = tau # target network soft update hyperparameter
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq #target network updating frequency
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learn_step = 0 #learning step counter
        self.memory_counter = 0
        self.technique = technique
        self.mode_update = mode_update
        
        self.model = []
        self.model_target = []
        self.mamemory = []
                
        #build model and memory
        for agent_idx in range(self.n_agents):
            self.model.append(self.build_model())
            self.model_target.append(self.build_model()) #Second (target) neural network
            self.mamemory.append(np.zeros((memory_size, self.state_size*2 + 2)))
    
    def build_model(self):
        
        inp = Input(shape=(self.state_size, ))
        h1 = Dense(256, activation='relu')(inp)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)

        outp = Dense(self.n_actions, activation='linear')(h3)
            
        model = Model(inputs = inp, outputs = outp)
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=self.alpha))
        return model
    
    def save_model(self):
        weights_model, weights_target = [], []
        for agent_idx in range(self.n_agents):
            wm = self.model[agent_idx].get_weights()
            wt = self.model_target[agent_idx].get_weights()
            weights_model.append(wm)
            weights_target.append(wt)
        return weights_model, weights_target
    
    def load_model(self, weights_models):
        for agent_idx in range(self.n_agents):
            self.model[agent_idx].set_weights(weights_models[agent_idx])
    
    # @tf.function
    def update_target_parameters(self):
        #Update the target model from the base model
        if (self.mode_update == 'hard') or (self.mode_update == 'Hard'):
            for agent_idx in range(self.n_agents):
                for (a1, b1) in zip(self.model_target[agent_idx].variables, self.model[agent_idx].variables):
                    a1.assign(b1)
        elif (self.mode_update == 'soft') or (self.mode_update == 'Soft'):
            for agent_idx in range(self.n_agents):
                for (a1, b1) in zip(self.model_target[agent_idx].variables, self.model[agent_idx].variables):
                    c = a1*(1-self.tau) + b1*self.tau
                    a1.assign(c)
    
    def choose_action(self, states):
        actions = []
        for agent_idx in range(self.n_agents):
            if np.random.rand() < self.epsilon:
                action = random.randrange(self.n_actions)
                actions.append(action) #Explore
            else:
                states_array = np.array(states[agent_idx]).reshape(1,self.state_size)
                action_vals = self.model[agent_idx].predict(states_array)
                actions.append(np.argmax(action_vals[0]))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return actions

    def test_action(self, states):
        actions = []
        for agent_idx in range(self.n_agents):
            states_array = np.array(states[agent_idx]).reshape(1,self.state_size)
            action_vals = self.model[agent_idx].predict(states_array)
            actions.append(np.argmax(action_vals[0]))
        return actions
    
    def store_experience(self, states, actions, rewards, nstates):
        idx = self.memory_counter % self.memory_size
        for agent_idx in range(self.n_agents):
            self.mamemory[agent_idx][idx,:] = np.concatenate((states[agent_idx], [actions[agent_idx], rewards[agent_idx]], nstates[agent_idx]))
        self.memory_counter += 1
    
    def learn(self):
        for agent_idx in range(self.n_agents):
            batch_indices = min(self.memory_counter, self.memory_size)
            minibatch = self.mamemory[agent_idx][np.random.choice(batch_indices, size = self.batch_size)]
                
            state_batch = minibatch[:, :self.state_size]
            action_batch = minibatch[:, self.state_size].astype(int)
            reward_batch = minibatch[:, self.state_size+1]
            nstate_batch = minibatch[:, -self.state_size:]
            
            q_model_st = self.model[agent_idx].predict(state_batch)
            q_target_nst = self.model_target[agent_idx].predict(nstate_batch)
            
            q_model_nst = self.model[agent_idx].predict(nstate_batch)
            q_model_st[range(self.batch_size), action_batch] = reward_batch + self.gamma*q_target_nst[range(self.batch_size), np.argmax(q_model_nst, axis=1)]
            
            self.model[agent_idx].fit(state_batch, q_model_st, self.batch_size, epochs=1, verbose=0)