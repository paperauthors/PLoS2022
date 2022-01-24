import numpy as np
import time
import copy

from solver import BaseSolver
# abstract interface

class Baseline(BaseSolver):
    def __init__(self, size, Mu, ell, param_c, param_K):
        

        super(Baseline, self).__init__(size, ell)

        self.t = 0
        self.pt = 1.0

        self.explore_count = 0

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

        self.param_c = param_c
        self.param_K = param_K 

        self.rate_d = 0.9

    def update_pt(self):
        self.pt = min(1.0, (self.param_c * self.param_K)/(self.rate_d**2*(self.t)))

    def generate_explore_ell_served(self):
        self.explore_count += 1
        tmp_ell = np.random.choice(range(self.n) , self.ell, replace=True)
        #eg. tmp_ell [33, 22, 0, 89, 89]
        ell = [0] * self.n
        for a in tmp_ell:
            ell[a] += 1

        self.this_step_served = ell
        self.served.append(ell)
        for a in range(self.n):
            self.A[a] += ell[a]
        return ell

    def p_estimates_from_theta(self):
        theta_sum = sum(self.theta)
        p = [1.0/self.n for a in range(self.n)]
        if theta_sum > 0: 
            p = [1.0 * self.theta[a]/theta_sum for a in range(self.n)]
        return p

    def exploit_p_estimates_from_theta(self, theta):
        theta_sum = sum(theta)
        p = [1.0/self.n for a in range(self.n)]
        if theta_sum > 0: 
            p = [1.0 * theta[a]/theta_sum for a in range(self.n)]
        return p

    def generate_exploit_ell_served(self):
        ## sorting the items to be served is implicit()
        reversed_order_theta_value = [0.0 - self.theta[a] for a in range(self.n)]
        exploit_theta_idx = [a for a in np.array(reversed_order_theta_value).argsort(kind='stable')[:self.ell]]
        ell = [ 1 if a in exploit_theta_idx else 0 for a in range(self.n)]

        self.this_step_served = ell
        self.served.append(ell)
        for a in range(self.n):
            self.A[a] += ell[a]
        return ell

    def generate_ell_served(self):
        u = np.random.random(1)[0] 
        served = []
        if u < self.pt :
            served =  self.generate_explore_ell_served()
        else:
            served = self.generate_exploit_ell_served()

        return served


    # clicks on all that are liked
    def generate_clicks(self):
        r = [0] * self.n
        like = 0
        # click all liked
        for a in range(self.n):
            if (self.this_step_served[a] > 0):
                for num_served in range(self.this_step_served[a]):
                    like = np.random.binomial(1, self.mu[a]) # a is clicked based on prob. mu[a]
                    r[a] += like # 0 or 1

        clicked = [a for a in range(self.n) if r[a] > 0 ]
        self.this_step_clicked = clicked
        self.clicked.append(clicked)
        
        for a in self.this_step_clicked:
            self.Clicks[a] += r[a] # clicked
        #self.this_step_r = r
        return r

    def update_true_rewards(self):
        # clicked == self.this_step_r[a] > 0 like == mu[a] > 0
        clicked_and_liked = [self.this_step_r[a] for a in range(self.n) if self.mu[a] > 0 and self.this_step_r[a] > 0] 
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
        for a in range(self.n):
            if self.A[a] > 0:
                self.theta[a] = (1.0 * self.Clicks[a])/self.A[a]


    def run_one_step(self):
        self.t += 1

        self.generate_ell_served()
        self.this_step_r = self.generate_clicks()
        self.update_true_rewards()
        self.update_server_believed_rewards()

        self.update_theta()
        self.update_pt()

        return self.this_step_inst_reward

    def print_theta(self):
        print('theta:- ' + str([round(a, 3) for a in self.theta]))

    def estimated_reward_prob(self):
        return self.theta
