import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_data(x, y, ax=None, alpha=0.6, s=18.0, colors=((0.61, 0.07, 0.03), (0.03, 0.57, 0.61)),
              legend=True, xrange=None, yrange=None):
    if xrange is None:
        xrange = (x[:, 0].min() - 0.5, x[:, 0].max() + 0.5)
    if yrange is None:
        yrange = (x[:, 1].min() - 0.5, x[:, 1].max() + 0.5)

    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x[y == 0, 0], x[y == 0, 1], marker='o', label='Class 1', alpha=alpha, color=colors[0], s=s)
    ax.scatter(x[y == 1, 0], x[y == 1, 1], marker='o', label='Class 2', alpha=alpha, color=colors[1], s=s)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Data')
    if legend:
        ax.legend(loc='upper left', scatterpoints=1, numpoints=1)
    return ax, xrange, yrange


def plot_predictive_distribution(predict_func, xrange=None, yrange=None, x_data=None, y_data=None, ax=None, res=300,
                                 levels=11, norm_levels=True):
    if ax is None:
        fig, ax = plt.subplots()
    if x_data is not None:
        assert y_data is not None
        ax, xrange, yrange = plot_data(x_data, y_data, ax=ax, xrange=xrange, yrange=yrange, alpha=0.5, s=10.)
    else:
        assert xrange is not None
        assert yrange is not None
    xx, yy = np.meshgrid(np.linspace(*xrange, res), np.linspace(*yrange, res))
    grid_data = np.stack((xx.ravel(), yy.ravel()), 1)
    grid_probs = predict_func(grid_data)
    grid_probs = grid_probs.reshape(xx.shape)
    if norm_levels and type(levels) is int:
        levels = np.linspace(0., 1., levels)
    cs2 = ax.contour(xx, yy, grid_probs, levels=levels, cmap='RdGy', linewidths=1.5, alpha=0.9)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=8)
    ax.figure.colorbar(cs2, ax=ax, aspect=40)
    return ax, xrange, yrange


def plot_predictive_contourf(predict_func, xrange, yrange, ax=None, res=300, num_levels=41, fit_levels=False):
    if ax is None:
        fig, ax = plt.subplots()
    xx, yy = np.meshgrid(np.linspace(*xrange, res), np.linspace(*yrange, res))
    grid_data = np.stack((xx.ravel(), yy.ravel()), 1)
    grid_probs = predict_func(grid_data)
    grid_probs = grid_probs.reshape(xx.shape)
    if fit_levels:
        levels = np.linspace(grid_probs.min(), grid_probs.max(), num_levels, endpoint=True)
    else:
        levels = np.linspace(0., 1., num_levels, endpoint=True)
        ticks = levels[::10]
    cs = ax.contourf(xx, yy, grid_probs, levels, cmap='RdGy')
    ax.figure.colorbar(cs, ax=ax, aspect=40, ticks=ticks)

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    return ax

