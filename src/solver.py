import numpy as np
import time
import copy

# defined local bandit.py
from bandit import MultinomialBandit, BernoulliBandit


# abstract interface
class BaseSolver(object):
    def __init__(self, size, ell = 1):
        np.random.seed(int(time.time()))

        self.n = size

        self.A = [0]* self.n # == counts number of times each category presented(Served)
        self.Clicks = [0] * self.n # number of clicks, one entry per category?
        self.theta = [0.0] * self.n # number of clicks, one entry per category?
        self.served = []
        self.clicked = []
        self.ell = ell # e.g. ell = 20, present 20 categories (out of 100 per time_step) 

        # total reward (number of clicks)
        self.cum_reward = 0.0
        # cumulative rewards over time t
        self.cum_rewards = [] 
        # instantaneous rewards over time t
        self.inst_rewards = [] 

    @property
    def estimated_reward_prob(self):
        raise NotImplementedError

    def run_one_step(self):
        raise NotImplementedError

    def run(self, steps):
        for _ in range(steps):
            self.run_one_step()

# abstract interface
class SolverMultinomial(BaseSolver):
    def __init__(self, bandit, l = 1):
        assert isinstance(bandit, MultinomialBandit)
        assert bandit.size > l

        super(SolverMultinomial, self).__init__(bandit.size ,l)
        
        self.bandit = bandit


class SolverBernoulli(BaseSolver):
    def __init__(self, bandit, l = 1):
        assert isinstance(bandit, BernoulliBandit)
        assert bandit.size > l

        super(SolverBernoulli, self).__init__(bandit.size ,l)
        
        self.bandit = bandit
