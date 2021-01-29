
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit/RL Class

usage:

from bandit import CCTSB
bandit = CCTSB(N=[4,3,3,4,5], K=5, C=100, alpha_p=0.5, nabla_p=0.5, w=0.5, obj_func=default_obj)
bandit.observe(context)
actions = bandit.act()
bandit.update(reward,cost)

"""

import numpy as np
import torch
from scipy.stats import lognorm

class Agent(object):
    def __init__(self):
        self.N = None
        self.K = None
        self.C = None
        self.geos = None
        self.weights = None

    def observe(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
            

def default_obj(r,s, weight=None):
    assert 0 < weight < 1, "objective weight should be in (0,1)"
    return weight * r + (1 - weight) / s
    
class CCTSB(Agent):
    def __init__(self,
                 N=None,
                 K=None,
                 C=None,
                 geos=None,
                 alpha_p=None,
                 nabla_p=None,
                 weights=None,
                 obj_func=default_obj):
        
        # for example: 
        # four possible actions: school closure, diet, vaccine, travel control
        # they each have different levels of interventions: 0-3, 0-3, 0-4, and 0-2
        
        # N needs a fix since there can be half values (to be verified)
        self.N = N # number of possible values in each action, e.g. [4,4,5,3]
        self.K = K # number of possible intervention actions, e.g. 4
        self.C = C # dimension of the context, e.g. 100
        self.geos = geos
        self.alpha = alpha_p
        self.nabla = nabla_p
        self.weights = weights
        self.obj_func = obj_func
        
        # fix: need to set useless N values to NaN or -inf
        self.B = torch.eye(C).repeat(1, np.max(N), K, len(geos), len(weights))
        self.z = torch.zeros(C, np.max(N), K, len(geos), len(weights))
        self.theta = torch.zeros(C, np.max(N), K, len(geos), len(weights))
        
        self.c_t = None
        self.i_t = None

        self.update_history = []
        self.update_range = 20
        self.update_coefficients = [lognorm.pdf(i, s= 0.4, scale=np.exp(2)) for i in range(self.update_range)]
        
    def observe(self, c):
        # C x geo x K x N x weights
        self.c_t = c.repeat(1, self.K, np.max(self.N), len(self.weights))

        # C x N x K x geo x weights
        self.c_t = torch.transpose(self.c_t, 1, 3)

    def act(self):        
        sample_theta = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=self.theta,
            covariance_matrix=self.B)
        
        dot_prod = torch.matmul(self.c_t, sample_theta, dim=0) # need to specify which dimension
        i_t = torch.argmax(dot_prod, dim=0)
        self.i_t = i_t
        self.update_history.append(self.i_t)
        return i_t
    
    def update(self, r=None, s=None, w=None):
        r_star = self.obj_func(r, s, w)
        for day in range(self.update_range):
            update_coefficient = self.update_coefficients[-(day+1)]
            for k in range(self.K):
                # print("Day : ", day)
                # print("update history : ", self.update_history)
                # print("update history specific : ", self.update_history[-(day+1)])
                try: 
                    i = self.update_history[-(day+1)][k] # at day 0, we want the most recent actions by the bandit (so -1)
                except:
                    # print('update history too short: ', len(self.update_history))
                    continue
                norm = np.linalg.norm(self.c_t, ord=1)
                norm_c_t = self.c_t/norm
                # print(norm, norm_c_t)
                # print(np.linalg.norm(self.B_i_k[k][i]))
                self.B_i_k[k][i] = self.nabla * self.B_i_k[k][i] + norm_c_t.dot(norm_c_t.T)
                # print(np.linalg.norm(self.B_i_k[k][i]))
                self.z_i_k[k][i] += norm_c_t * r_star * update_coefficient
                self.theta_i_k[k][i] = np.linalg.inv(self.B_i_k[k][i]).dot(self.z_i_k[k][i])
    
    def clear_update_hist(self):
        self.update_history = []
