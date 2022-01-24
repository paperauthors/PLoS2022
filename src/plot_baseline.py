import matplotlib.pyplot as plt
import numpy as np
import os

from base_plot import RewardPlot

class BaselineRewardPlot(RewardPlot):

    def __init__(self, param, folder, plot_folder):
        super(BaselineRewardPlot, self).__init__(param, folder, plot_folder)
        self.fig_prefix = 'm1_'
        self.reward_type_map = {'cum_rewards':' ', 'server_cum_rewards':'- server believed rewards'}

    def add_series(self, label, new_label):
        s = self.read_series(label)
        s['new_label'] = new_label
        self.series[label] = s

    def plot_cum_reward(self, title, fig_name, labels, reward_types, show_legend):
        fig, (ax1) = plt.subplots(1, sharex=True, sharey = False)
        skeys = labels

        for key in skeys:
            data = self.series[key]
            for rtype in reward_types[key]:
                cr = [0]
                cr.extend(data[rtype][0:self.plot_steps])
                x = [t for t in range(self.plot_steps + 1)]
                y = [cr[t]/(t+1) for t in x]
                ax1.plot(x, y, label = '{}{}'.format(data['new_label'], self.reward_type_map[rtype]), color='red')
                px = 1.0 * self.plot_steps
                py = y[-1]
                text_exp = ''
                ax1.text(px, py, '{0:.2f}{1}'.format(y[-1], text_exp), fontsize = 5)
        if show_legend:
            ax1.legend(loc='center right')
        ax1.set_title(title)
        plt.xlabel('Time steps')
        plt.ylabel('Rewards')
        pname = '{0}{1}{2}'.format(self.plot_folder, self.fig_prefix + 'cum_' + fig_name , self.plot_suffix)
        fig.savefig(pname)
        plt.close()


def plot_all(p, plot_steps):
    p.set_plot_steps(plot_steps)
    fig_name = 'm1'
    p.add_series('m1', 'model 1')
    title = 'Rewards over time for model 1, \n (ave over 100 runs)'
    slab = ['m1']
    rtypes = {'m1':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_legend = False)

    p.set_plot_steps(200)
    fig_name = 'ucb_m1'
    p.add_series('ucb_m1', 'UCB user model 1')
    title = 'UCB Rewards estimates, \n (Ave over 100 runs) A:{0}, l:{1}'.format(p.param['A'], p.param['l'])
    slab = ['ucb_m1']
    rtypes = {'ucb_m1':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_legend = True)

if __name__ == '__main__':

    param = {'A': 100, 'l': 20, 'steps': 3000, 'param_c':2, 'param_K':5,'runs': 100}
    folder = '../npy/m1/'
    sub_folder = 'r{0}steps{1:.0f}k'.format(param['runs'], param['steps']/1000)
    plot_folder = '../plot/m1/{}/'.format(sub_folder)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    p = BaselineRewardPlot(param, folder, plot_folder) 
    plot_all(p, param['steps'])
