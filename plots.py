 #!/usr/bin/env python
 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp
from jax import grad

def plot_function(grid_1d, func, contours=50, log_contours=False, exact=[0,0]):
    '''Make a contour plot over the region described by grid_1d for function func.'''

    # make the 2D grid
    X,Y = np.meshgrid(grid_1d, grid_1d, indexing='xy')
    Z = np.zeros_like(X)

    for i in range(len(X)):
        for j in range(len(X.T)):
            Z[i, j] = func(np.array((X[i, j], Y[i, j])))  # compute function values

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ###
    ax = fig.add_subplot(1, 2, 1)

    if not log_contours:
        ax.contour(X, Y, Z, contours, cmap='Spectral_r')
    else:
        ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap='Spectral_r')

    ax.plot(*exact, '*', color='black')

    ax.set_xlabel(r'$w_0$')
    ax.set_ylabel(r'$w_1$')
    ax.set_aspect('equal')

    ###
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')

    if log_contours:
        Z = np.log(Z)
        label = r'$\ln f(\mathbf{w}$'
    else:
        label = r'$f(\mathbf{w})$'

    surf = ax3d.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap='Spectral_r',
                       linewidth=0, antialiased=True, rasterized=True)

    ax3d.plot([exact[0]], [exact[0]], [func(np.array(exact))], marker='*', ms=6, linestyle='-', color='k',lw=1, zorder=100)

    ax3d.set_xlabel(r'$w_0$',labelpad=8)
    ax3d.set_ylabel(r'$w_1$',labelpad=8)
    ax3d.set_zlabel(label,labelpad=8);

    return fig,ax,ax3d