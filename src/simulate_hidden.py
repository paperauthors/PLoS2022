import matplotlib.pyplot as plt
import numpy as np

from bandit import BernoulliBandit, MultinomialBandit

from hidden_ucb_solver import UCBHiddenPartialR, UCBHiddenFullR
from hidden_ucb_solver import RandomBernoulliHiddenPartialR,RandomBernoulliHiddenFullR

from hidden_solver import HiddenPartialR, HiddenFullR
from hidden_solver import RandomMultinomialHiddenPartialR, RandomMultinomialHiddenFullR

def simulate_hidden(param, folder, suffix):
    #partial_series = ['ucb_partial','bernulli_random_partial','m3_partial','multinomial_random_partial']
    Mu = [param['frac'] for a in range(param['A'])]

    # ucb_partial
    solvers = []
    for run in range(param['runs']):
        bandit = BernoulliBandit(param['A'], Mu)
        solver = UCBHiddenPartialR(bandit, param['l'], param['k'], param['frac'], 1.0)
        solvers.append(solver)
    simulate(solvers, 'ucb_partial', param, folder, suffix)

    # random_bernoulli_partial
    solvers = []
    for run in range(param['runs']):
        bandit = BernoulliBandit(param['A'], Mu)
        solver = RandomBernoulliHiddenPartialR(bandit, param['l'], param['k'], param['frac'], 1.0)
        solvers.append(solver)
    simulate(solvers, 'random_bernoulli_partial', param, folder, suffix)

    # m3_partial
    solvers = []
    for run in range(param['runs']):
        bandit = MultinomialBandit(param['A'], Mu)
        solver = HiddenPartialR(bandit, param['l'], param['k'], param['frac'])
        solvers.append(solver)
    simulate(solvers, 'm3_partial', param, folder, suffix)

    # random_multinomial_partial
    solvers = []
    for run in range(param['runs']):
        bandit = MultinomialBandit(param['A'], Mu)
        solver = RandomMultinomialHiddenPartialR(bandit, param['l'], param['k'], param['frac'])
        solvers.append(solver)
    simulate(solvers, 'random_multinomial_partial', param, folder, suffix)

    #full_r_series = ['ucb_full','bernulli_random_full','m3_full','multinomial_random_full']
    # ucb_full
    solvers = []
    for run in range(param['runs']):
        bandit = BernoulliBandit(param['A'], Mu)
        solver = UCBHiddenFullR(bandit, param['l'], param['M'], param['frac'], 1.0)
        solvers.append(solver)
    simulate(solvers, 'ucb_full', param, folder, suffix)

    # random_bernoulli_full
    solvers = []
    for run in range(param['runs']):
        bandit = BernoulliBandit(param['A'], Mu)
        solver = RandomBernoulliHiddenFullR(bandit, param['l'], param['M'], param['frac'], 1.0)
        solvers.append(solver)
    simulate(solvers, 'random_bernoulli_full', param, folder, suffix)

    # m3_full
    solvers = []
    for run in range(param['runs']):
        bandit = MultinomialBandit(param['A'], Mu)
        solver = HiddenFullR(bandit, param['l'], param['M'], param['frac'])
        solvers.append(solver)
    simulate(solvers, 'm3_full', param, folder, suffix)

    # random_multinomial_full
    solvers = []
    for run in range(param['runs']):
        bandit = MultinomialBandit(param['A'], Mu)
        solver = RandomMultinomialHiddenFullR(bandit, param['l'], param['M'], param['frac'])
        solvers.append(solver)
    simulate(solvers, 'random_multinomial_full', param, folder, suffix)



def simulate(solvers, label, param, folder, suffix):
    inst_rewards_allruns = [0.0 for x in range(param['steps'])]
    cum_rewards_allruns = [0.0 for x in range(param['steps'])]
    for run in range(param['runs']):
        solver = solvers[run]
        solver.run(param['steps'])
        for t in range(param['steps']):
            inst_rewards_allruns[t] += solver.inst_rewards[t]
            cum_rewards_allruns[t] += solver.cum_rewards[t]
    sim_inst_rewards = [inst_rewards_allruns[t]/param['runs'] for t in range(param['steps'])]
    sim_cum_rewards = [cum_rewards_allruns[t]/param['runs'] for t in range(param['steps'])]
    result= [{'label': label, 'inst_rewards': sim_inst_rewards, 'cum_rewards': sim_cum_rewards}]
    np.save('{0}{1}{2}'.format(folder, result[0]['label'], suffix), result)
    print('{} done'.format(label))


if __name__ == '__main__':

    params = {'A': 100, 'l': 20, 'steps': 50000, 'frac':0.5, 'k': 0.1, 'M': 1000, 'runs': 100}
    #params_test = {'A': 100, 'l': 20, 'steps': 1000, 'frac':0.5, 'k': 0.1, 'M': 1000, 'runs': 10}
    #params = {'A': 100, 'l': 20, 'steps': 50000, 'frac':0.5, 'k': 0.1, 'M': 1000, 'runs': 100}
    suffix = '_r{0}_s{1}.npy'.format(str(params['runs']), str(params['steps']))
    #output_root = '../npy/hidden/' # v1
    output_root = '../npy/hidden2/'
    simulate_hidden(params, output_root, suffix)
    
    #sim_result = np.load(output_root + npy)
