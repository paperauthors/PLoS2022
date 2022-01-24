import numpy as np
import time

#interface abstracted
class BaseBandit(object):
    def generate_rewards(self, ell):
        raise NotImplementedError

class BernoulliBandit(BaseBandit):
    # A : number of arms(categories)
    # Mu: the reward probabilities for each arm (category) 
    def __init__(self, size, Mu = None):
        assert Mu is None or len(Mu) == size
        self.size = size
        if Mu is None:
            np.random.seed(int(time.time()))
            self.Mu = [np.random.random() for _ in range(self.size)]
        else:
            self.Mu = Mu

        self.max_mu = max(self.Mu)
    
    def generate_reward(self, a):
        # bernoulli reward for arm- (category-) a
        # reward == 1 with probability p (Mu[a]), and otherwise reward == 0.
        if np.random.random() < self.Mu[a]:
            return 1
        else:
            return 0

    def generate_rewards(self, ell):
        r = [np.random.binomial(1, self.Mu[a]) if a in ell else 0 for a in range(self.size)]
        return r

    def update_mu(self, Mu_new):
        assert len(Mu_new) == self.size
        self.Mu = Mu_new



class MultinomialBandit(BaseBandit):
    # A : number of arms(categories)
    # Mu: the probabilities for each arm (category) ie. prob. of clicks
    #    -- the theta vector as prior distribution (for baysian ucb)
    def __init__(self, size, ell, Mu = None):
        assert Mu is None or len(Mu) == size
        self.size = size
        self.ell = ell
        #
        if Mu is None:
            np.random.seed(int(time.time()))
            self.Mu = [np.random.random() for _ in range(self.size)]
        else:
            self.Mu = Mu

        self.max_mu = max(self.Mu)

    # rewards is defined clicks.
    def generate_rewards(self, served):
        assert len(served) == self.size
        r = [np.random.binomial(served[a], self.Mu[a]) for a in range(self.size)]
        return r

    def update_mu(self, Mu_new):
        assert len(Mu_new) == self.size
        self.Mu = Mu_new

