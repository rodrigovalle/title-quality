import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D

from helper import load_obj

def proba_mse():
    pass

gridsearch = load_obj('gridsearch_clarity')
results = gridsearch.cv_results_

fig = plt.figure()
ax = plt.axes(projection='3d')

X = np.log10(results['param_C'].data.astype('float').reshape(11, 10))
Y = np.log10(results['param_gamma'].data.astype('float').reshape(11, 10))
Z = np.sqrt(-results['mean_test_score'].reshape(11, 10))

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel("\nC")
ax.set_ylabel("\ngamma")
ax.set_zlabel("\nMean RMSE of three CVs")

@mticker.FuncFormatter
def log_tick_formatter(val, pos=None):
    return "{:.0e}".format(10**val)

ax.xaxis.set_major_formatter(log_tick_formatter)
ax.yaxis.set_major_formatter(log_tick_formatter)

plt.show()