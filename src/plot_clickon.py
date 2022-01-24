import matplotlib.pyplot as plt
import numpy as np
import os

from base_plot import RewardPlot

class DivergentRewardPlot(RewardPlot):

    def __init__(self, param, folder, plot_folder):
        super(DivergentRewardPlot, self).__init__(param, folder, plot_folder)
        self.fig_prefix = 'm2_'
        self.reward_type_map = {'cum_rewards':'true rewards', 'server_cum_rewards':'server believed rewards'}

    def add_series(self, label, new_label):
        s = self.read_series(label)
        s['new_label'] = new_label
        self.series[label] = s

    def plot_cum_reward(self, title, fig_name, labels, reward_types, show_param):
        fig, (ax1) = plt.subplots(1, sharex=True, sharey = False)
        skeys = labels

        for key in skeys:
            data = self.series[key]
            for rtype in reward_types[key]:
                cr = [0]
                cr.extend(data[rtype][0:self.plot_steps])
                x = [t for t in range(self.plot_steps +1)]
                y = [cr[t]/(t+1) for t in x]
                ax1.plot(x, y, label = '{}-{}'.format(data['new_label'], self.reward_type_map[rtype]))
                px = 1.0 * self.plot_steps
                py = 0.95 * y[-1]
                text_exp = ''
                if y[-1] < 0.1 or key == 'ucb_random_clickon_cat90' :
                    py = -0.3
                    if key == 'ucb_random_clickon_cat90':
                        text_exp = ' (random)'

                ax1.text(px, py, '{0:.2f}{1}'.format(y[-1], text_exp), fontsize = 5)

        caption = 'params: {{c = {0:.1f}, K = {1:.1f}, d = {2:.1f}}}\naveraged over {3:3d} runs.'.format(self.param['param_c'], self.param['param_K'], 0.9 , self.param['runs'])
        if self.param['runs'] == 1:
            caption = 'params: {{c = {0:.1f}, K = {1:.1f}, d = {2:.1f}}}\naveraged over a single run'.format(self.param['param_c'], self.param['param_K'], 0.9)
        if (show_param == False):
            caption = ''
        fig.text(.7, .01, caption, ha='left', fontsize = 7)
        ax1.legend()
        ax1.set_title(title)
        plt.xlabel('Time step')
        plt.ylabel('Cumulative rewards')
        pname = '{0}{1}{2}'.format(self.plot_folder, self.fig_prefix + 'cum_' + fig_name , self.plot_suffix)
        fig.savefig(pname)
        plt.close()
        #plt.show()


def plot_all(p, plot_steps):
    p.set_plot_steps(plot_steps)

    # cat 5
    fig_name = 'cat5'
    # 1: cat5
    p.add_series('m2_cat5', 'model 2')
    p.add_series('random_clickon_cat5', 'random')
    title = 'Model 2: like category = 5'
    slab = ['m2_cat5', 'random_clickon_cat5']
    rtypes = {'m2_cat5':['cum_rewards', 'server_cum_rewards'], \
            'random_clickon_cat5':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_param = True)

    # cat 90 
    fig_name = 'cat90'
    # 2: cat90
    p.add_series('m2_cat90', 'model 2')
    p.add_series('random_clickon_cat90', 'random')
    title = 'Model 2: like category = 90'
    slab = ['m2_cat90', 'random_clickon_cat90']
    rtypes = {'m2_cat90':['cum_rewards', 'server_cum_rewards'], \
            'random_clickon_cat90':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_param = True)

    # 4cats 
    fig_name = 'multi4cats'
    # 3: 4cats
    p.add_series('m2_4cats', 'model 2')
    p.add_series('random_clickon_4cats', 'random')
    title = 'Model 2: like categories = [3, 26, 45, 77]'
    slab = ['m2_4cats', 'random_clickon_4cats']
    rtypes = {'m2_4cats':['cum_rewards', 'server_cum_rewards'], \
            'random_clickon_4cats':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_param = True)

    #### ucb
    # cat 5
    fig_name = 'ucb_cat5'
    # 1: cat5
    p.add_series('ucb_clickon_cat5', 'ucb')
    p.add_series('ucb_random_clickon_cat5', 'random')
    title = 'UCB: like category = 5'
    slab = ['ucb_clickon_cat5', 'ucb_random_clickon_cat5']
    rtypes = {'ucb_clickon_cat5':['cum_rewards', 'server_cum_rewards'], \
            'ucb_random_clickon_cat5':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_param = False)

    # cat 90 
    fig_name = 'ucb_cat90'
    # 2: cat90
    p.add_series('ucb_clickon_cat90', 'ucb')
    p.add_series('ucb_random_clickon_cat90', 'random')
    title = 'UCB: like category = 90'
    slab = ['ucb_clickon_cat90', 'ucb_random_clickon_cat90']
    rtypes = {'ucb_clickon_cat90':['cum_rewards', 'server_cum_rewards'], \
            'ucb_random_clickon_cat90':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_param = False)

    # 4cats 
    fig_name = 'ucb_multi4cats'
    # 3: 4cats
    p.add_series('ucb_clickon_4cats', 'ucb')
    p.add_series('ucb_random_clickon_4cats', 'random')
    title = 'UCB: like categories = [3, 26, 45, 77]'
    slab = ['ucb_clickon_4cats', 'ucb_random_clickon_4cats']
    rtypes = {'ucb_clickon_4cats':['cum_rewards', 'server_cum_rewards'], \
            'ucb_random_clickon_4cats':['cum_rewards']}
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_param = False)


if __name__ == '__main__':

    param = {'A': 100, 'l': 20, 'steps': 5000, 'param_c':2, 'param_K':5,'runs': 100}
    folder = '../npy/clickon/'
    sub_folder = 'r{0}steps{1:.0f}k'.format(param['runs'], param['steps']/1000)
    plot_folder = '../plot/clickon/{}/'.format(sub_folder)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    p = ClickonRewardPlot(param, folder, plot_folder) 
    plot_all(p, 200)


    param = {'A': 100, 'l': 20, 'steps': 5000, 'param_c':2, 'param_K':5,'runs': 1}
    folder = '../npy/clickon/'
    sub_folder = 'r{0}steps{1:.0f}k'.format(param['runs'], param['steps']/1000)
    plot_folder = '../plot/clickon/{}/'.format(sub_folder)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    p = DivergentRewardPlot(param, folder, plot_folder) 
    #plot_all(p, param['steps'])

    # Single run cat 90 
    fig_name = 'cat90_single'
    # 2: cat90
    p.add_series('m2_cat90', 'model 2')
    title = 'Model 2: like category = 90'
    slab = ['m2_cat90' ]
    rtypes = {'m2_cat90':['cum_rewards', 'server_cum_rewards']}
    p.set_plot_steps(200)
    p.plot_cum_reward(title, fig_name, labels = slab, reward_types = rtypes, show_param = True)
