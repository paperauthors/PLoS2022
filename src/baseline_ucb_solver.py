import numpy as np
import time
import copy

# abstract interface
from solver import BaseSolver

class BaselineUCB(BaseSolver):
    def __init__(self, size, Mu, ell ):

        super(BaselineUCB, self).__init__(size, ell)

        self.t = 0
        self.theta = [1.0] * self.n

        self.server_cum_reward = 0.0
        self.server_cum_rewards = []
        self.server_inst_rewards = []

        self.mu = copy.copy(Mu)
        self.max_mu = max(self.mu)

        self.this_step_served = [] 
        self.this_step_clicked = []
        self.this_step_r = []
        self.this_step_inst_reward = 0.0
        self.this_step_server_inst_reward = 0.0


    def p_estimates_from_theta(self):
        theta_sum = sum(self.theta)
        p = [1.0/self.n for a in range(self.n)]
        if theta_sum > 0: 
            p = [1.0 * self.theta[a]/theta_sum for a in range(self.n)]
        return p


    def _gen_ell(self):
        # sort in reverse order
        ci = [0.0 - (self.theta[x] + np.sqrt(2 * np.log(self.t) / (1 + self.A[x]))) \
                for x in range(self.n)]
        # top l elements. this is the served list
        ell = np.array(ci).argsort(kind='stable')[:self.ell]
        ell.sort()
        return ell

    def generate_ell_served(self):
        ell = self._gen_ell()

        self.this_step_served = ell
        self.served.append(ell)
        for a in ell:
            self.A[a] += 1
        return ell


    def generate_clicks(self):
        r = [0] * self.n
        like = 0
        # this_step_served is order for clicking.
        # click all items that is liked.
        for a in self.this_step_served: 
            like = np.random.binomial(1, self.mu[a]) # a is clicked based on prob. mu[a]
            if like:
                r[a] += 1

        clicked = [a for a in range(self.n) if r[a] > 0 ]
        self.this_step_clicked = clicked
        self.clicked.append(clicked)

        for a in self.this_step_clicked:
            self.Clicks[a] += r[a] # clicked
        return r

    def update_true_rewards(self):
        # clicked == self.this_step_r[a] > 0 like == mu[a] > 0
        clicked_and_liked = [self.this_step_r[a] for a in range(self.n) if self.mu[a] > 0] 
        inst_reward = sum(clicked_and_liked)

        self.this_step_inst_reward = inst_reward
        self.cum_reward += self.this_step_inst_reward
        self.cum_rewards.append(self.cum_reward)
        self.inst_rewards.append(self.this_step_inst_reward)

    def update_server_believed_rewards(self):
        server_inst_reward = sum(self.this_step_r)

        self.this_step_server_inst_reward = server_inst_reward
        self.server_cum_reward += self.this_step_server_inst_reward
        self.server_cum_rewards.append(self.server_cum_reward)
        self.server_inst_rewards.append(self.this_step_server_inst_reward)


    def update_theta(self):
        for a in self.this_step_served:
            self.theta[a] = (1.0 * self.Clicks[a]/self.A[a])
    

    def run_one_step(self):
        self.t += 1

        self.generate_ell_served()
        self.this_step_r = self.generate_clicks()

        self.update_true_rewards()
        self.update_server_believed_rewards()

        self.update_theta()

        return self.this_step_inst_reward

    def print_theta(self):
        print('theta:- ' + str([round(a, 3) for a in self.theta]))

    def estimated_reward_prob(self):
        return self.theta



