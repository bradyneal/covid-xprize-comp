
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
            
