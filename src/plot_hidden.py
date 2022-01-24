import matplotlib.pyplot as plt
import numpy as np
import os

from base_plot import RewardPlot

class HiddenRewardPlot(RewardPlot):

    def __init__(self, param, folder, plot_folder):
        super(HiddenRewardPlot, self).__init__(param, folder, plot_folder)

    # plot series of labels, if labels == None, plot all.
    def plot_cum_reward(self, title, fig_name, labels = None):
        fig, (ax1) = plt.subplots(1, sharex=True, sharey = False)
        skeys = labels
        if labels is None:
            skeys = self.series.keys()

        for key in skeys:
            data = self.series[key]
            cr = [0]
            cr.extend(data['cum_rewards'][0:self.plot_steps])
            
            x = [t for t in range(self.plot_steps + 1)]
            y = [cr[t]/(t+1) for t in x]
            x = x[1:self.plot_steps]
            y = y[1:self.plot_steps]
            ax1.plot(x, y, label = data['new_label'])

        ax1.legend()
        ax1.set_title(title)
        plt.xlabel('Time step')
        plt.ylabel('Cumulative rewards')
        pname = '{0}{1}{2}'.format(self.plot_folder, 'cum_' + fig_name , self.plot_suffix)
        fig.savefig(pname)
        plt.close()
        #plt.show()

    def plot_diff_cum_reward(self, title, fig_name, labels_a, labels_b):
        assert len(labels_a) == len(labels_b)
        fig, (ax1) = plt.subplots(1, sharex=True, sharey = False)
        skeys_a = labels_a
        rtype = 'cum_rewards'
        for key_a, key_b in zip(labels_a, labels_b):
            data_a = self.series[key_a]
            data_b = self.series[key_b]
            cr_a = [0]
            cr_b = [0]
            cr_a.extend(data_a[rtype][0:self.plot_steps])
            cr_b.extend(data_b[rtype][0:self.plot_steps])
            cr = [a-b for a, b in zip(cr_a, cr_b)]
            x = [t for t in range(self.plot_steps + 1)]
            y = [cr[t]/(t+1) for t in x]
            x = x[1:self.plot_steps]
            y = y[1:self.plot_steps]
            ax1.plot(x, y, label = '{} diff {}'.format(data_a['new_label'], data_b['new_label']))
        # add an abline(0)
        ax1.plot([min(x), max(x)], [0, 0], '--', color='red' )
        ax1.legend()
        ax1.set_title(title)
        plt.xlabel('Time step')
        plt.ylabel('Difference in cumulative rewards')
        pname = '{0}{1}{2}'.format(self.plot_folder, 'diffcum_' + fig_name , self.plot_suffix)
        fig.savefig(pname)
        plt.close()
        #plt.show()



def plot_figures(p):

    # full replenishment
    fig_name = 'Full_ucb'
    p.set_plot_steps(1000)
    # 1: ucb_ful
    p.add_series('random_bernoulli_full', r'random')
    p.add_series('ucb_full', 'ucb')
    title = 'UCB: hidden category with full replenishment'
    slab = ['random_bernoulli_full', 'ucb_full']
    p.plot_cum_reward(title, fig_name, labels = slab)
    p.plot_diff_cum_reward(title, fig_name, labels_a = [slab[0]], labels_b = [slab[1]])

    fig_name = 'Full_m3'
    p.set_plot_steps(200)
    # 2: m3_full
    p.add_series('random_multinomial_full', r'random')
    p.add_series('m3_full', 'model 3')
    title = 'Model 3: hidden category with full replenishment'
    slab = ['random_multinomial_full', 'm3_full']
    p.plot_cum_reward(title, fig_name, labels = slab)
    p.set_plot_steps(50000)
    p.plot_diff_cum_reward(title, fig_name, labels_a = [slab[0]], labels_b = [slab[1]])


    # partial replenishment
    fig_name = 'Partial_ucb'
    p.set_plot_steps(200)
    # 3: ucb_partial 
    p.add_series('random_bernoulli_partial', r'random')
    p.add_series('ucb_partial', 'ucb')
    title = 'UCB: hidden category with partial replenishment'
    slab = ['random_bernoulli_partial', 'ucb_partial']
    p.plot_cum_reward(title, fig_name, labels = slab)
    p.plot_diff_cum_reward(title, fig_name, labels_a = [slab[0]], labels_b = [slab[1]])


    fig_name = 'Partial_m3'
    p.set_plot_steps(1000)
    # 4: m3_partial
    p.add_series('random_multinomial_partial', r'random')
    p.add_series('m3_partial', 'model 3')
    title = 'Model 3: hidden category with partial replenishment'
    slab = ['random_multinomial_partial', 'm3_partial']
    p.plot_cum_reward(title, fig_name, labels = slab)
    p.plot_diff_cum_reward(title, fig_name, labels_a = [slab[0]], labels_b = [slab[1]])


if __name__ == '__main__':
    #param_test = {'A': 100, 'l': 20, 'steps': 1000, 'frac':0.5, 'k': 0.1, 'M': 1000, 'runs': 10}
    #param = {'A': 100, 'l': 20, 'steps': 50000, 'frac':0.5, 'k': 0.1, 'M': 1000, 'runs': 100}
    param = {'A': 100, 'l': 20, 'steps': 50000, 'frac':0.5, 'k': 0.1, 'M': 1000, 'runs': 100}

    folder = '../npy/hidden/'
    sub_folder = 'r{0}steps{1:.0f}k'.format(param['runs'], param['steps']/1000)
    plot_folder = '../plot/hidden/{}/'.format(sub_folder)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    p = HiddenRewardPlot(param, folder, plot_folder) 
    plot_figures(p)
