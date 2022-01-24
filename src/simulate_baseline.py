import matplotlib.pyplot as plt
import numpy as np

from baseline_solver import Baseline
from baseline_ucb_solver import BaselineUCB

def simulate_baseline(param, folder, suffix):

    size = param['A']
    ell = param['l']
    param_c = param['param_c']
    param_K = param['param_K']

    Mu = [0.75 if a < 50 else 0.5 for a in range(size)]

    solvers = [Baseline(size, Mu, ell, param_c, param_K) for run in range(param['runs'])]
    simulate(solvers, 'm1', param, folder, suffix)

    #ucb
    solvers = [BaselineUCB(size, Mu, ell) for run in range(param['runs'])]
    simulate(solvers, 'ucb_m1', param, folder, suffix)



def simulate(solvers, label, param, folder, suffix):
    inst_rewards_allruns = [0.0 for x in range(param['steps'])]
    cum_rewards_allruns = [0.0 for x in range(param['steps'])]
    server_inst_rewards_allruns = [0.0 for x in range(param['steps'])]
    server_cum_rewards_allruns = [0.0 for x in range(param['steps'])]

    for run in range(param['runs']):
        solver = solvers[run]
        solver.run(param['steps'])
        for t in range(param['steps']):
            inst_rewards_allruns[t] += solver.inst_rewards[t]
            cum_rewards_allruns[t] += solver.cum_rewards[t]
            server_inst_rewards_allruns[t] += solver.server_inst_rewards[t]
            server_cum_rewards_allruns[t] += solver.server_cum_rewards[t]
    sim_inst_rewards = [inst_rewards_allruns[t]/param['runs'] for t in range(param['steps'])]
    sim_cum_rewards = [cum_rewards_allruns[t]/param['runs'] for t in range(param['steps'])]
    sim_server_inst_rewards = [server_inst_rewards_allruns[t]/param['runs'] for t in range(param['steps'])]
    sim_server_cum_rewards = [server_cum_rewards_allruns[t]/param['runs'] for t in range(param['steps'])]

    result= [{'label': label, \
            'inst_rewards': sim_inst_rewards, 'cum_rewards': sim_cum_rewards, \
            'server_inst_rewards': sim_server_inst_rewards, 'server_cum_rewards': sim_server_cum_rewards}]

    np.save('{0}{1}{2}'.format(folder, result[0]['label'], suffix), result)
    print('{} done'.format(label))


if __name__ == '__main__':

    params = {'A': 100, 'l': 20, 'steps': 3000, 'param_c':2, 'param_K':5,'runs': 100}
    suffix = '_r{0}_s{1}.npy'.format(str(params['runs']), str(params['steps']))
    output_root = '../npy/m1/'
    simulate_baseline(params, output_root, suffix)
    
    #sim_result = np.load(output_root + npy)
