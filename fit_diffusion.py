import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('simprop.mplstyle')
import numpy as np
from utils import set_axes, savefig, find_first_peaks
import math

from iminuit import Minuit

MODEL = 'yx'

def get_data(string_E, N):
    x = np.loadtxt(f'data/time/time_10tothe{string_E}_PeV_eta0.5.txt')
    R = np.loadtxt(f'data/correlation_functions/R{MODEL}_10tothe{string_E}_PeV_eta0.5.txt')
    var_R = np.loadtxt(f'data/variance/Var{MODEL}_10tothe{string_E}_PeV_eta0.5.txt')
    std_R = np.sqrt(var_R) / np.sqrt(N)
    return x / 1e9, R / 1e16, std_R / 1e16 # just a normalization factor

def plot_data(ax, string_E, nparticles, npeaks):
    x, R, std_R = get_data(string_E, nparticles)
    i = (x < 1e6) 
    ax.errorbar(x[i], R[i], yerr=std_R[i], color='tab:orange', fmt='o', capsize=5)

    peak_positions, peaks = find_first_peaks(x, R, npeaks)

    for peak_position in peak_positions:
        ax.axvline(peak_position, color='tab:gray', ls='--', lw=1.5)

    return peak_positions, R[peaks], R[0]

def fit_diffusion(initial_params, string_E, max_t, N):
    # Get data
    x, R, std_R = get_data(string_E, N)
    i = (x < max_t)
    x = x[i]
    R = R[i]    
    std_R = std_R[i]

    # Define chi2 function
    def chi2_function(A, omega, tau):
        chi2 = 0.
        for x_i, R_i, std_R_i in zip(x, R, std_R):
            if MODEL == 'xx' or MODEL == 'yy': 
                model = A * np.cos(omega * x_i) * np.exp(-x_i / tau)
            elif MODEL == 'xy' or MODEL == 'yx':
                model = A * np.sin(omega * x_i) * np.exp(-x_i / tau)  
            chi2 += np.power((model - R_i) / std_R_i, 2.)
        return chi2
    
    # Initialize Minuit with initial parameters
    A, omega, tau = initial_params
    m = Minuit(chi2_function, A=A, omega=omega, tau=tau)
    m.errordef = Minuit.LEAST_SQUARES

    # Set parameter limits
    # m.limits['alpha'] = (3.1, 3.4)

    # Perform minimization
    m.simplex()
    m.migrad()
#   m.hesse()

    # Compute degrees of freedom (dof)
    dof = len(x) - m.nfit - 1

#    print(m.values)
    return m.values, m.errors, m.fval, dof

def plot_fit(string_E, nparticles, npeaks):
    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    xlabel = r'time'
    ylabel = fr'$R_{MODEL}$'

    # Plot data and get the peak positions
    peak_positions, peaks, R0 = plot_data(ax, string_E, nparticles, npeaks)
    
    # Get the maximum time as the last peak
    max_t = peak_positions[-1]

    # Set axes from 0 to max_t
    set_axes(ax, xlabel, ylabel, xlim=[0, max_t], ylim=[-3.3, 3.3])

    # Guess frequency from the distance between the first two peaks
    distance_peaks = peak_positions[1] - peak_positions[0]
    omega = 2. * math.pi / distance_peaks

    # Guess the amplitude from the first peak
    A = R0

    # Guess the decay time from the last peak
    fit = np.polyfit(np.log(peak_positions), np.log(peaks), 1)
    tau = fit[1]
    
    # Initial parameters
    values = [A, omega, tau]

    print(f'Initial parameters: {values}')

    values, errors, fval, dof = fit_diffusion(values, string_E, max_t, 5e3)

    print(f'Final parameters:')
    print(f'A = {values[0]:.3f} +/- {errors[0]:.3f}')
    print(f'omega = {values[1]:.2e} +/- {errors[1]:.2e}')
    print(f'tau = {values[2]:.3f} +/- {errors[2]:.3f}')
    print(f'Chi2 / dof = {fval:.3f} / {dof}')
   
    # Plot fit
    t = np.linspace(0, max_t, 1000) 
    A, omega, tau = values
    if MODEL == 'xx' or MODEL == 'yy': 
        y = A * np.cos(omega * t) * np.exp(-t / tau)
    elif MODEL == 'xy' or MODEL == 'yx':
       y = A * np.sin(omega * t) * np.exp(-t / tau)
    ax.plot(t, y, 'tab:blue', zorder=10, lw=3, label='Fit')

    # #ax.legend(loc='best', fontsize=24)
    savefig(fig, f'R_{MODEL}_{string_E}.pdf')

    return values, errors

if __name__ == '__main__':
    N = 5e3

    models = [('0.2', 3), 
              ('0.4', 6), 
              ('0.6', 10), 
              ('0.8', 12), 
              ('1.0', 25), 
              ('1.2', 30), 
              ('1.4', 33), 
              ('1.6', 33), 
              ('1.8', 20), 
              ('2.0', 14)]

    with open(f'R{MODEL}_fit_params.txt', 'w') as f:
        for string_E, npeaks in models:
            params, errors = plot_fit(string_E, N, npeaks)
            f.write(f'{string_E} {params[0]:.3e} {errors[0]:.3e} {params[1]:.3e} {errors[1]:.3e} {params[2]:.3e} {errors[2]:.3e}\n')

