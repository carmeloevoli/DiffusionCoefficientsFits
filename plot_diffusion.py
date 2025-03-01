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
    set_axes(ax, xlabel, ylabel, xscale='log', yscale='log', xlim=[1, 1e3], ylim=[0.1, 1e1])

    filename = 'Rxx_fit_params.txt'
    E, omega, rho = np.loadtxt(filename, unpack=True, usecols=(0, 3, 5))
    E = np.power(10., E)

    # Integrals
    # \int_0^\infty dx exp(-rho x) cos(omega x) = rho / (rho^2 + omega^2)
    # \int_0^\infty dx exp(-rho x) sin(omega x) = omega / (rho^2 + omega^2)

    D = omega / (rho**2 + omega**2)

    print(min(D), max(D))

    y = D

    ax.plot(E, y / max(y), 'o')

    savefig(fig, f'D_xx.pdf')

if __name__ == '__main__':
    plot_diffusion()

