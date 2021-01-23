
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit/RL Class

usage:

from bandit import CCTSB
bandit = CCTSB(N=[4,3,3,4,5], K=5, C=100, alpha_p=0.5, nabla_p=0.5)
bandit.observe(context)
actions = bandit.act()
bandit.update(reward,cost)

"""

import numpy as np

class Agent(object):
    def __init__(self):
        self.N = None
        self.K = None
        self.C = None

    def observe(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
            

class CCTSB(Agent):
    def __init__(self, N=None, K=None, C=None, alpha_p=None, nabla_p=None):
        
        # for example: 
        # four possible actions: school closure, diet, vaccine, travel control
        # they each have different levels of interventions: 0-3, 0-3, 0-4, and 0-2
        
        # needs a fix since there can be half values
        self.N = N # number of possible values in each action, e.g. [4,4,5,3]
        self.K = K # number of possible intervention actions, e.g. 4
        self.C = C # dimension of the context, e.g. 100
        
        self.alpha = alpha_p
        self.nabla = nabla_p
        
        self.B_i_k = [ n * [np.eye(C)] for n in self.N ]
        self.z_i_k = [ n * [np.zeros((C))] for n in self.N ]
        self.theta_i_k = [ n * [np.zeros((C))] for n in self.N ]
        
        self.c_t = None
        self.i_t = None
        
    def observe(self, c):
        self.c_t = c

    def act(self):
        sample_theta = [ n * [0] for n in self.N ]
        i_t = {}
        for k in range(self.K):
            
            for i in range(len(sample_theta[k])):
                sample_theta[k][i] = np.random.multivariate_normal(self.theta_i_k[k][i], self.alpha**2 * np.linalg.inv(self.B_i_k[k][i]))
            
            i_t[k] = np.argmax(self.c_t.T.dot(np.array(sample_theta[k]).T))
        
        self.i_t = i_t
        return i_t
    
    def update(self, r=None, s=None):
        r_star = r/s
        for k in range(self.K):
            i = self.i_t[k]
            self.B_i_k[k][i] = self.nabla * self.B_i_k[k][i] + self.c_t.dot(self.c_t.T)
            self.z_i_k[k][i] += self.c_t * r_star
            self.theta_i_k[k][i] = np.linalg.inv(self.B_i_k[k][i]).dot(self.z_i_k[k][i])
            
        
            
