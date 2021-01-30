
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
            

def default_obj(r,s, w):
    weights = torch.tensor(w)
    weights = weights.view(-1, 1)
    weights = weights.expand_as(r)

    s = s.transpose(0,1)

    reward = torch.mul(weights, r) + torch.div(1 - weights, s)
    # geo x weight
    return reward.transpose(0,1)
    
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
        # 5 x 12 x 200 x 10 x 12 x 12
        self.B = torch.eye(C).repeat(np.max(N), K, len(geos), len(weights), 1, 1)
        # 5 x 12 x 200 x 10 x 12
        self.z = torch.zeros(np.max(N), K, len(geos), len(weights), C)
        self.theta = torch.zeros(np.max(N), K, len(geos), len(weights), C)
        
        self.c_t = None
        self.i_t = None

        self.update_history = []
        self.update_range = 20
        self.update_coefficients = [lognorm.pdf(i, s= 0.4, scale=np.exp(2)) for i in range(self.update_range)]
        
    def observe(self, c):
        # N x K x weights x geos x C
        self.c_t = c.repeat(np.max(self.N), self.K,  len(self.weights), 1, 1)

        # N x K x geos x weights x C
        self.c_t = torch.transpose(self.c_t, 2, 3)

    def act(self):        
        sample_theta = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=self.theta,
            covariance_matrix=self.B).sample()
        
        dot_prod = torch.einsum("cnkgw,cnkgw->nkgw", self.c_t, sample_theta)
        i_t = torch.argmax(dot_prod, dim=0)

        # i_t is geo x w x k
        self.i_t = i_t
        self.update_history.append(self.i_t)
        return i_t
    
    def update(self, r=None, s=None, w=None):

        # geo x weight
        r_star = self.obj_func(r, s, w)

        for day in range(self.update_range):
            update_coefficient = self.update_coefficients[-(day+1)]
            try:  
                # geo x weight x k
                i = self.update_history[-(day+1)] # at day 0, we want the most recent actions by the bandit (so -1)
            except:
                # print('update history too short: ', len(self.update_history))
                continue
            norm = np.linalg.norm(self.c_t, ord=1)
            norm_c_t = self.c_t/norm
            self.B_i_k[k][i] = self.nabla * self.B_i_k[k][i] + norm_c_t.dot(norm_c_t.T)
            # print(np.linalg.norm(self.B_i_k[k][i]))
            self.z_i_k[k][i] += norm_c_t * r_star * update_coefficient
            self.theta_i_k[k][i] = np.linalg.inv(self.B_i_k[k][i]).dot(self.z_i_k[k][i])
    
    def clear_update_hist(self):
        self.update_history = []
