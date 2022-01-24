import numpy as np
import time
import copy

# defined local solver.py
# abstract interface
from solver import SolverMultinomial

class Hidden(SolverMultinomial):
    # hidden_frac = 0.5
    # the bandit instance should be initialised with hiddenMu
    def __init__(self, bandit, l, hidden_frac):

        super(Hidden, self).__init__(bandit, l)

        self.t = 0
        self.theta = [hidden_frac] * self.bandit.size

        # serverMu controls category presented.
        self.serverMu = copy.copy(self.bandit.Mu)

        # theta is the estimated reward probability
        self.hidden_fraction = [hidden_frac] * self.bandit.size

        # hiddenMu controls categories clicks,
        self.hiddenMu = [x * y  for (x, y) in zip(self.bandit.Mu, self.hidden_fraction)]
        self.bandit.update_mu(self.hidden_fraction)
        

        self.this_step_served = [] 
        self.this_step_r = []
        self.this_step_clicked = []
        self.this_step_inst_reward = 0.0

    def p_estimates_from_theta(self):
        theta_sum = sum(self.theta)
        return [self.theta[a]/theta_sum for a in range(self.bandit.size)]

    def generate_ell_served(self):

        ell = np.random.multinomial(self.ell, self.p_estimates_from_theta())
        #[2, 0, 1, ...] length ==  self.bandit.size , sum to self.ell
        self.this_step_served = ell
        self.served.append(ell)
        for a in range(self.bandit.size):
            self.A[a] += ell[a]
        return ell

    
    def generate_click_rewards(self):
        r = self.bandit.generate_rewards(self.this_step_served)
        clicked = [a for a in range(self.bandit.size) if r[a] > 0 ]
        self.this_step_clicked = clicked

        self.clicked.append(clicked)
        for a in self.this_step_clicked:
            self.Clicks[a] += r[a] # clicked
        # r is a binary vector of length self.bandit.size
        #self.this_step_r = r
        return r

    def update_rewards(self):
        #self.this_step_inst_reward = sum(r) ==> ucb reward, for comparison
        theta_sum = sum(self.theta)
        inst_reward = 0.0
        inst_reward = sum([self.ell*(self.theta[a]/theta_sum)*self.hidden_fraction[a] for a in range(self.bandit.size)])

        self.this_step_inst_reward = inst_reward
        self.cum_reward += self.this_step_inst_reward
        self.cum_rewards.append(self.cum_reward)
        self.inst_rewards.append(self.this_step_inst_reward)

    def update_theta(self):
        for a in self.this_step_clicked:
            # increment theta by number of clicks. unnormalised by /theta_sum
            self.theta[a] += self.this_step_r[a]
    

    def update_hidden_fraction(self):
        raise NotImplementedError

    def update_hidden_mu(self):
        theta_sum = sum(self.theta)
        p = [th/theta_sum for th in self.theta]
        mew_hidden_mu = [x * y  for (x, y) in zip(p, self.hidden_fraction)]

        self.bandit.update_mu(self.hidden_fraction)

    def run_one_step(self):
        self.t += 1

        self.generate_ell_served()
        self.this_step_r = self.generate_click_rewards()
        self.update_theta()
        self.update_rewards()
        self.update_hidden_fraction()
        self.update_hidden_mu()

        return self.this_step_inst_reward

    def print_theta(self):
        print('theta:- ' + str([round(a, 3) for a in self.theta]))

    def estimated_reward_prob(self):
        return self.theta

class HiddenPartialR(Hidden):

    def __init__(self, bandit, l, k, hidden_frac):
        super(HiddenPartialR, self).__init__(bandit, l, hidden_frac)
        # k controls depleting rate of hidden categories. partial replenishment
        self.k = k

    def update_hidden_fraction(self):
        # partial replenishment
        for a in self.this_step_clicked:
            self.hidden_fraction[a] = max(0, self.hidden_fraction[a] - self.k*self.this_step_r[a])


class HiddenFullR(Hidden):

    def __init__(self, bandit, l, M, hidden_frac):

        super(HiddenFullR, self).__init__(bandit, l, hidden_frac)
        # M controls depleting rate of hidden categories. full replenishment
        self.M = M 
        self.hidden_fraction_0 = [hidden_frac] * self.bandit.size

    def update_hidden_fraction(self):
        # full replenishment
        for a in range(self.bandit.size):
            xi = np.random.binomial(self.this_step_served[a], self.hidden_fraction_0[a])
            C_ta = self.this_step_r[a]
            self.hidden_fraction[a] = min(max(0, self.hidden_fraction[a] - C_ta/self.M + xi/self.M), 1.0)


class RandomMultinomial(Hidden):

    def __init__(self, bandit, l, hidden_frac):
        super(RandomMultinomial, self).__init__(bandit, l, hidden_frac)

    def update_hidden_fraction(self):
        raise NotImplementedError

    def generate_ell_served(self):
        served = np.random.choice(self.bandit.size, self.ell, replace=True)
        ell = [0] * self.bandit.size
        for a in served:
            ell[a] += 1

        self.this_step_served = ell
        self.served.append(ell)
        for a in range(self.bandit.size):
            self.A[a] += ell[a]
        # ell is a vector of self.ell categories (integers)
        return ell

class RandomMultinomialHiddenPartialR(RandomMultinomial):

    def __init__(self, bandit, l, k, hidden_frac):
        super(RandomMultinomialHiddenPartialR, self).__init__(bandit, l, hidden_frac)
        # k controls depleting rate of hidden categories. partial replenishment
        self.k = k

    def update_hidden_fraction(self):
        # partial replenishment
        for a in self.this_step_clicked:
            self.hidden_fraction[a] = max(0, self.hidden_fraction[a] - self.k*self.this_step_r[a])


class RandomMultinomialHiddenFullR(RandomMultinomial):

    def __init__(self, bandit, l, M, hidden_frac):

        super(RandomMultinomialHiddenFullR, self).__init__(bandit, l, hidden_frac)
        self.M = M
        self.hidden_fraction_0 = [hidden_frac] * self.bandit.size

    def update_hidden_fraction(self):
        # full replenishment
        for a in range(self.bandit.size):
            xi = np.random.binomial(self.this_step_served[a], self.hidden_fraction_0[a])
            C_ta = self.this_step_r[a]
            self.hidden_fraction[a] = min(max(0, self.hidden_fraction[a] - C_ta/self.M + xi/self.M), 1.0)


