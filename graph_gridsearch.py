import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D

from helper import load_obj

def proba_mse():
    pass

param_shape = (11,9)

def plot_score(ax, gs_results, z_param, cmap):
    X = np.log10(gs_results['param_C'].data.astype('float').reshape(*param_shape))
    Y = np.log10(gs_results['param_gamma'].data.astype('float').reshape(*param_shape))
    Z = np.sqrt(-gs_results[z_param].reshape(*param_shape))
    ax.plot_surface(X, Y, Z, cmap=cmap)

@mticker.FuncFormatter
def log_tick_formatter(val, pos=None):
    return "{:.0e}".format(10**val)

if __name__ == '__main__':
    gridsearch = load_obj('gridsearch_concise')
    results = gridsearch.cv_results_

    import pdb; pdb.set_trace()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_score(ax, results, 'mean_test_score', 'viridis')
    #plot_score(ax, results, 'mean_train_score', 'inferno')

    ax.set_xlabel("\nC")
    ax.set_ylabel("\ngamma")
    ax.set_zlabel("\nMean RMSE of three CVs")

    ax.xaxis.set_major_formatter(log_tick_formatter)
    ax.yaxis.set_major_formatter(log_tick_formatter)

    plt.show()
