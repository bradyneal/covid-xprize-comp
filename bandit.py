
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit/RL Class
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
    def __init__(self, N=None, K=None, C=None, alpha=None):
        self.N = N
        self.K = K
        self.C = C
        self.alpha = alpha
        
        self.A_k = self.K * [np.eye(N+1)]
        self.g_k = self.K * [np.zeros((N+1,N+1))]
        self.mu_k = self.K * [np.zeros((N+1,N+1))]
        self.B_i = self.N * [np.eye(N+1)]
        self.z_i = self.N * [np.zeros((N+1,N+1))]
        self.theta_i = self.N * [np.zeros((N+1,N+1))]
        
        self.c = None
        
    def observe(self,c):
        self.c = c

    def act(self):
        sample_mu = np.zeros((self.K,self.N))
        sample_theta = np.zeros((self.K,self.N))
        for k in range(self.K):
            for i in range(self.N):
                sample_mu[k,i] = np.random.multivariate_normal(self.mu_k[k], self.alpha**2 * np.inv(self.A_k[k]))
                sample_theta[k,i] = np.random.multivariate_normal(self.theta_i[i], self.alpha**2 * np.inv(self.B_i[i]))
    
    def update(self):
        raise NotImplementedError
            
