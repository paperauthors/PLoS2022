import matplotlib.pyplot as plt
import numpy as np
import os

class RewardPlot(object):

    def __init__(self, param, folder, plot_folder):
        self.series = {}
        self.sparam = {}
        self.param = param
        self.suffix = '_r{0}_s{1}'.format(str(param['runs']), str(param['steps']))

        self.source_folder = folder
        self.source_suffix = '{0}.npy'.format(self.suffix)

        self.plot_steps = param['steps']
        self.plot_folder = plot_folder
        self.plot_suffix = '_x{}.pdf'.format(self.plot_steps)

        self.fig_prefix =''

    def update_plot_suffix_from_steps(self):
        s = self.plot_steps
        if s > 999:
            self.plot_suffix = '_x{0:.0f}k.pdf'.format(s/1000)
        else:
            self.plot_suffix = '_x{}.pdf'.format(s)

    def set_plot_steps(self, s):
        self.plot_steps = s
        self.update_plot_suffix_from_steps()

    def read_series(self, label):
        npy = '{0}{1}{2}'.format(self.source_folder, label, self.source_suffix)
        result = np.load(npy, allow_pickle=True)
        s = result[0];
        return s

    def add_series(self, label, new_label):
        s = self.read_series(label)
        s['new_label'] = new_label
        self.series[label] = s


