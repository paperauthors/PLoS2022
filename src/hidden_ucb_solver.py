import numpy as np
import time
import copy
# A beta continuous random variable
from scipy.stats import beta

# defined local solver.py
# abstract interface
from solver import SolverBernoulli

class UCBHidden(SolverBernoulli):
    # hidden_frac = 0.5
    # the bandit instance should be initialised with hiddenMu
    def __init__(self, bandit, l, hidden_frac, th = 1.0):

        super(UCBHidden, self).__init__(bandit, l)
        self.t = 0
        self.theta = [th] * self.bandit.size

        # serverMu controls category presented.
        self.serverMu = copy.copy(self.bandit.Mu)

        # theta is the estimated reward probability
        self.hidden_fraction = [hidden_frac] * self.bandit.size

        # hiddenMu controls categories clicks,
        self.bandit.update_mu(self.hidden_fraction)
        

        self.this_step_served = []
        self.this_step_r = []
        self.this_step_clicked = []
        self.this_step_inst_reward = 0.0

    def generate_ell_served(self):
        # sort in reverse order
        ci = [0.0 - (self.theta[x] + np.sqrt(2 * np.log(self.t) / (1 + self.A[x]))) \
                for x in range(self.bandit.size)]
        # top l elements. this is the served list
        ell = np.array(ci).argsort(kind='stable')[:self.ell]

        self.this_step_served = ell
        self.served.append(ell)
        for a in ell:
            self.A[a] += 1
        # ell is a vector of self.ell categories (integers)
        return ell

    
    def generate_click_rewards(self):
        r = self.bandit.generate_rewards(self.this_step_served)
        clicked = [a for a in range(self.bandit.size) if r[a] == 1 ]
        self.this_step_clicked = clicked
        self.clicked.append(clicked)
        for a in self.this_step_clicked:
            self.Clicks[a] += 1 # clicked
        return r

    def update_rewards(self):
        self.this_step_inst_reward = sum(self.this_step_r)
        self.cum_reward += self.this_step_inst_reward
        self.cum_rewards.append(self.cum_reward)
        self.inst_rewards.append(self.this_step_inst_reward)


    def update_theta(self):
        for a in self.this_step_served:
            self.theta[a] = 1.0 * (self.Clicks[a]/self.A[a])


    def update_hidden_fraction(self):
        raise NotImplementedError

    def update_hidden_mu(self):
        self.bandit.update_mu(self.hidden_fraction)

    def run_one_step(self):
        self.t += 1

        self.generate_ell_served()
        self.this_step_r = self.generate_click_rewards()
        self.update_rewards()

        self.update_theta()
        self.update_hidden_fraction()
        self.update_hidden_mu()
        return self.this_step_inst_reward

    def print_theta(self):
        print('theta:- ' + str([round(a, 3) for a in self.theta]))

    def estimated_reward_prob(self):
        return self.theta

class UCBHiddenPartialR(UCBHidden):

    def __init__(self, bandit, l, k, hidden_frac, th = 1.0):
        super(UCBHiddenPartialR, self).__init__(bandit, l, hidden_frac, th)
        # k controls depleting rate of hidden categories. partial replenishment
        self.k = k

    def update_hidden_fraction(self):
        # partial replenishment
        for a in self.this_step_clicked:
            self.hidden_fraction[a] = max(0, self.hidden_fraction[a] - self.k * 1)



class UCBHiddenFullR(UCBHidden):

    def __init__(self, bandit, l, M, hidden_frac, th = 1.0):

        super(UCBHiddenFullR, self).__init__(bandit, l, hidden_frac, th)
        self.M = M 
        self.hidden_fraction_0 = [hidden_frac] * self.bandit.size

    def update_hidden_fraction(self):
        # full replenishment
        for a in self.this_step_served:
            xi = np.random.binomial(1, self.hidden_fraction_0[a])
            C_ta = 1 if a in self.this_step_clicked else 0
            self.hidden_fraction[a] = min(max(0, self.hidden_fraction[a] - C_ta/self.M + xi/self.M), 1.0)

class RandomBernoulli(UCBHidden):
    def __init__(self, bandit, l, hidden_frac, th = 1.0):

        super(RandomBernoulli, self).__init__(bandit, l, hidden_frac, th)

    def update_hidden_fraction(self):
        raise NotImplementedError

    def generate_ell_served(self):
        ell = np.random.choice(self.bandit.size, self.ell, replace=False)

        self.this_step_served = ell
        self.served.append(ell)
        for a in ell:
            self.A[a] += 1
        # ell is a vector of self.ell categories (integers)
        return ell

class RandomBernoulliHiddenPartialR(RandomBernoulli):

    def __init__(self, bandit, l, k, hidden_frac, th = 1.0):
        super(RandomBernoulliHiddenPartialR, self).__init__(bandit, l, hidden_frac, th)
        # k controls depleting rate of hidden categories. partial replenishment
        self.k = k

    def update_hidden_fraction(self):
        # partial replenishment
        for a in self.this_step_clicked:
            self.hidden_fraction[a] = max(0, self.hidden_fraction[a] - self.k)



class RandomBernoulliHiddenFullR(RandomBernoulli):

    def __init__(self, bandit, l, M, hidden_frac, th = 1.0):

        super(RandomBernoulliHiddenFullR, self).__init__(bandit, l, hidden_frac, th)
        self.M = M
        self.hidden_fraction_0 = [hidden_frac] * self.bandit.size

    def update_hidden_fraction(self):
        # full replenishment
        for a in self.this_step_served:
            xi = np.random.binomial(1, self.hidden_fraction_0[a])
            C_ta = 1 if a in self.this_step_clicked else 0
            self.hidden_fraction[a] = min(max(0, self.hidden_fraction[a] - C_ta/self.M + xi/self.M), 1.0)


