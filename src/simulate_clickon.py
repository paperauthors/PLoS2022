import matplotlib.pyplot as plt
import numpy as np

from divergent_solver import Divergent, RandomClickon
from divergent_ucb_solver import DivergentUCB, RandomClickonUCB

def simulate_divergent(param, folder, suffix):

    size = param['A']
    ell = param['l']
    param_c = param['param_c']
    param_K = param['param_K']

    Mu5 = [1.0 if a == 5 else 0.0 for a in range(size)]
    Mu90 = [1.0 if a == 90 else 0.0 for a in range(size)]

    cats = [3, 26, 45, 77]
    prob = {a:p for a, p in list(zip(cats , [0.88, 0.62, 0.54, 0.92]))}
    Mu4cats = [prob[a] if a in cats else 0.0 for a in range(size)]

    solvers = [Divergent(size, Mu5, ell, param_c, param_K) for run in range(param['runs'])]
    simulate(solvers, 'm2_cat5', param, folder, suffix)

    solvers = [Divergent(size, Mu90, ell, param_c, param_K) for run in range(param['runs'])]
    simulate(solvers, 'm2_cat90', param, folder, suffix)
    
    solvers = [Divergent(size, Mu4cats, ell, param_c, param_K) for run in range(param['runs'])]
    simulate(solvers, 'm2_4cats', param, folder, suffix)


    solvers = [RandomClickon(size, Mu5, ell, param_c, param_K)  for run in range(param['runs'])]
    simulate(solvers, 'random_clickon_cat5', param, folder, suffix)

    solvers = [ RandomClickon(size, Mu90, ell, param_c, param_K) for run in range(param['runs'])]
    simulate(solvers, 'random_clickon_cat90', param, folder, suffix)

    solvers = [ RandomClickon(size, Mu4cats, ell, param_c, param_K) for run in range(param['runs'])]
    simulate(solvers, 'random_clickon_4cats', param, folder, suffix)
    
    #ucb
    solvers = [DivergentUCB(size, Mu5, ell) for run in range(param['runs'])]
    simulate(solvers, 'ucb_clickon_cat5', param, folder, suffix)

    solvers = [DivergentUCB(size, Mu90, ell) for run in range(param['runs'])]
    simulate(solvers, 'ucb_clickon_cat90', param, folder, suffix)
    
    solvers = [DivergentUCB(size, Mu4cats, ell) for run in range(param['runs'])]
    simulate(solvers, 'ucb_clickon_4cats', param, folder, suffix)


    solvers = [RandomClickonUCB(size, Mu5, ell)  for run in range(param['runs'])]
    simulate(solvers, 'ucb_random_clickon_cat5', param, folder, suffix)

    solvers = [ RandomClickonUCB(size, Mu90, ell) for run in range(param['runs'])]
    simulate(solvers, 'ucb_random_clickon_cat90', param, folder, suffix)

    solvers = [ RandomClickonUCB(size, Mu4cats, ell) for run in range(param['runs'])]
    simulate(solvers, 'ucb_random_clickon_4cats', param, folder, suffix)


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

    #params = {'A': 100, 'l': 20, 'steps': 5000, 'param_c':2, 'param_K':5,'runs': 100}
    #params_test = {'A': 100, 'l': 20, 'steps': 3000, 'param_c':2, 'param_K':5,'runs': 1}
    params = {'A': 100, 'l': 20, 'steps': 5000, 'param_c':2, 'param_K':5,'runs': 100}
    suffix = '_r{0}_s{1}.npy'.format(str(params['runs']), str(params['steps']))
    output_root = '../npy/clickon/'

    simulate_divergent(params, output_root, suffix)

    # Single run
    params = {'A': 100, 'l': 20, 'steps': 5000, 'param_c':2, 'param_K':5,'runs': 1}
    size = params['A']
    ell = params['l']
    param_c = params['param_c']
    param_K = params['param_K']

    Mu90 = [1.0 if a == 90 else 0.0 for a in range(size)]
    solvers = [Divergent(size, Mu90, ell, param_c, param_K) ]
    suffix = '_r{0}_s{1}.npy'.format(str(params['runs']), str(params['steps']))
    simulate(solvers, 'm2_cat90', params, output_root, suffix)

    #sim_result = np.load(output_root + npy)
