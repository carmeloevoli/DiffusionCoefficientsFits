import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from utils import set_axes, savefig

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

def plot_diffusion():
    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    xlabel, ylabel = r'$E$', r'$D$'
    set_axes(ax, xlabel, ylabel, xscale='log', yscale='log', xlim=[1, 1e3], ylim=[1e-2, 1e1])

    filename = 'Ryy_fit_params.txt'
    logE, omega, tau = np.loadtxt(filename, unpack=True, usecols=(0, 3, 5))
    E = np.power(10., logE) 

    print(omega *  tau) 

    # Integrals
    # arXiv:astro-ph/0408054
    D = (omega *  tau) / (1. +  (omega *  tau)**2)

    y = D

    ax.plot(E, y / max(y), 'o')

    savefig(fig, f'D_xx.pdf')

if __name__ == '__main__':
    plot_diffusion()

