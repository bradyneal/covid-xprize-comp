
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
NORM_CONST_STRINGENCY = 1
import numpy as np
from scipy.stats import lognorm
import time

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
            

def default_obj(r_past, r_present,s, weight=None):
    # assert 0 < weight < 1, "objective weight should be in (0,1)"
    w = weight
    # print("reward : ", r)
    # print(" string :  ", s)
    # print(" r_star : ", weight * r + (1 - weight) / (s/NORM_CONST_STRING))
    # return weight * r + (1 - weight) / np.max((0.1, s / NORM_CONST_STRINGENCY))
    # return weight * r - (1 - weight) * s
    # return 1/ (1 + np.exp(-np.log(r)))
    # print('stringency : ', s)
    
    return w * (r_past/r_present) + (1-w)/(s/17)
    
class CCTSB(Agent):
    def __init__(self, N=None, K=None, C=None, alpha_p=None, nabla_p=None, w=0.5, obj_func=default_obj, verbose=False, choice=None):
        
        # for example: 
        # four possible actions: school closure, diet, vaccine, travel control
        # they each have different levels of interventions: 0-3, 0-3, 0-4, and 0-2
        
        # N needs a fix since there can be half values (to be verified)
        self.N = N # number of possible values in each action, e.g. [4,4,5,3]
        self.K = K # number of possible intervention actions, e.g. 4
        self.C = C # dimension of the context, e.g. 100
        self.verbose = verbose
        self.rewards = []
        self.choice = choice

        assert choice in ('fixed', 'sample', 'random'), 'choose a proper learning method'
        
        self.alpha = alpha_p
        self.nabla = nabla_p
        self.w = w
        self.obj_func = obj_func
        
        self.B_i_k = [ n * [np.eye(C)] for n in self.N ]
        self.z_i_k = [ n * [np.zeros((C))] for n in self.N ]
        self.theta_i_k = [ n * [np.zeros((C))] for n in self.N ]
        
        self.c_t = None
        self.i_t = None

        self.update_history = []
        self.update_range = 20
        self.update_coefficients = [lognorm.pdf(i, s= 0.4, scale=np.exp(2)) for i in range(self.update_range)]
        self.context_history = []
        
    def observe(self, c):
        # self.c_t = 1/ (1 + np.exp(-(c-1)))
        self.c_t = c

    def act(self):
        sample_theta = [ n * [0] for n in self.N ]
        i_t = {}
        for k in range(self.K):
            
            if self.choice == 'fixed':
            
                for i in range(len(sample_theta[k])):

                    # sample_theta[k][i] = np.random.multivariate_normal(self.theta_i_k[k][i], self.alpha**2 * np.linalg.pinv(self.B_i_k[k][i]))
                        sample_theta[k][i] = self.theta_i_k[k][i]
                # print('self.theta_i_k[k][i] : ', np.linalg.norm(self.theta_i_k[k][i]))
                i_t[k] = np.argmax(self.c_t.T.dot(np.array(sample_theta[k]).T))

            elif self.choice == 'sample':
                for i in range(len(sample_theta[k])):
                    sample_theta[k][i] = np.random.multivariate_normal(self.theta_i_k[k][i], self.alpha**2 * np.linalg.pinv(self.B_i_k[k][i]))
                i_t[k] = np.argmax(self.c_t.T.dot(np.array(sample_theta[k]).T))

            elif self.choice == 'random':
                i_t[k] = np.random.randint(0,len(sample_theta[k]))

            # print('dot product : ', self.c_t.T.dot(np.array(sample_theta[k]).T))
            # print('c_t : ', self.c_t)
        self.i_t = i_t
        self.update_history.append(self.i_t)
        self.context_history.append(self.c_t)
        return i_t
    
    def update(self, r_past=None, r_present=None, s=None, w=None, verbose=False):
        r_star = self.obj_func(r_past, r_present, s, w)
        inner_verbose=False
        for k in range(self.K):
            for day in list(range(self.update_range)): # {0 - 19}
                # if k == 0:
                #     if day == 0:
                #         inner_verbose=True
                #     else:
                #         inner_verbose=False
                # else:
                #     inner_verbose=False
                update_coefficient = self.update_coefficients[-(day+1)]
                try: 
                    i = self.update_history[-(day+1)][k] # at day 0, we want the most recent actions by the bandit (so -1)
                except:
                    continue

                norm = np.linalg.norm(self.c_t, ord=1)
                norm_c_t = self.context_history[-(day+1)][k]

                self.B_i_k[k][i] = self.nabla * self.B_i_k[k][i] + np.outer(norm_c_t, norm_c_t) * update_coefficient

                self.z_i_k[k][i] += norm_c_t * r_star * update_coefficient

                self.theta_i_k[k][i] = np.linalg.pinv(self.B_i_k[k][i]).dot(self.z_i_k[k][i])
                if verbose==True:
                    if inner_verbose == True:
                        print('Condition number : ', np.linalg.cond(self.B_i_k[k][i]))
        self.rewards.append((r_star, r_past, r_present, s, *self.i_t.values()))

    def clear_update_hist(self):
        self.update_history = []
