import matplotlib
import numpy as np
import math
from iminuit import Minuit
import matplotlib.pyplot as plt
from utils import set_axes, savefig, find_first_peaks

# Configure Matplotlib backend
matplotlib.use('MacOSX')
plt.style.use('simprop.mplstyle')

MODEL = 'yy'  # Define the model type

def get_data(string_E: str, N: int):
    """
    Load and preprocess data from files.
    """
    try:
        x = np.loadtxt(f'data/time/time_10tothe{string_E}_PeV_eta0.5.txt')
        R = np.loadtxt(f'data/correlation_functions/R{MODEL}_10tothe{string_E}_PeV_eta0.5.txt')
        var_R = np.loadtxt(f'data/variance/Var{MODEL}_10tothe{string_E}_PeV_eta0.5.txt')
    except OSError as e:
        print(f"Error loading data files: {e}")
        return None, None, None

    std_R = np.sqrt(var_R) / np.sqrt(N)
    return x / 1e9, R / 1e16, std_R / 1e16  # Normalize data

def plot_data(ax, string_E: str, nparticles: int, npeaks: int):
    """
    Plot data with error bars and mark peak positions.
    """
    x, R, std_R = get_data(string_E, nparticles)
    if x is None:
        return None, None, None

    i = x < 1e6
    ax.errorbar(x[i], R[i], yerr=std_R[i], color='tab:orange', fmt='o', capsize=5)

    peak_positions, peaks = find_first_peaks(x, R, npeaks)
    for peak_position in peak_positions:
        ax.axvline(peak_position, color='tab:gray', ls='--', lw=1.5)
    
    return peak_positions, R[peaks], R[0]

def fit_diffusion(initial_params: list, string_E: str, max_t: float, N: int):
    """
    Fit diffusion model to data using Minuit.
    """
    x, R, std_R = get_data(string_E, N)

    i = x < max_t
    x, R, std_R = x[i], R[i], std_R[i]
    
    def chi2_function(A, omega, tau):
        if MODEL == 'xx' or MODEL == 'yy':
            chi2 = np.sum(((A * np.cos(omega * x) * np.exp(-x / tau) - R) / std_R) ** 2)
        elif MODEL == 'zz':
            chi2 = np.sum(((A * np.sin(omega * x) * np.exp(-x / tau) - R) / std_R) ** 2)
        return chi2
    
    A, omega, tau = initial_params
    m = Minuit(chi2_function, A=A, omega=omega, tau=tau)
    m.errordef = Minuit.LEAST_SQUARES
    
    try:
        m.simplex()
        m.migrad()
    except Exception as e:
        print(f"Error during fitting: {e}")
        return None, None, None, None
    
    dof = len(x) - m.nfit - 1
    return m.values, m.errors, m.fval, dof

def plot_fit(string_E: str, nparticles: int, npeaks: int):
    """
    Perform data plotting and fitting.
    """
    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    xlabel, ylabel = r'time', fr'$R_{MODEL}$'

    peak_positions, peaks, R0 = plot_data(ax, string_E, nparticles, npeaks)
    if peak_positions is None:
        return None, None
    
    max_t = peak_positions[-1]
    set_axes(ax, xlabel, ylabel, xlim=[0, max_t], ylim=[-3.3, 3.3])
    
    distance_peaks = peak_positions[1] - peak_positions[0]
    omega = 2 * math.pi / distance_peaks
    
    fit = np.polyfit(np.log(peak_positions), np.log(peaks), 1)
    tau = fit[1]
    
    values = [R0, omega, tau]
    print(f'Initial parameters: {values}')
    
    values, errors, fval, dof = fit_diffusion(values, string_E, max_t, nparticles)
    if values is None:
        return None, None
    
    print(f'Final parameters: A={values[0]:.3f}±{errors[0]:.3f}, omega={values[1]:.2e}±{errors[1]:.2e}, tau={values[2]:.3f}±{errors[2]:.3f}, Chi2/dof={fval:.3f}/{dof}')
    
    t = np.linspace(0, max_t, 1000)
    if MODEL == 'xx' or MODEL == 'yy':
        y = values[0] * np.cos(values[1] * t) * np.exp(-t / values[2])
    elif MODEL == 'xy':
        y = values[0] * np.sin(values[1] * t) * np.exp(-t / values[2])
    ax.plot(t, y, 'tab:blue', lw=3, label='Fit', zorder=10) 
    
    savefig(fig, f'R_{MODEL}_{string_E}.pdf')
    return values, errors

if __name__ == '__main__':
    N = int(5e3)  # Ensure integer value

    models = [('0.2', 3), ('0.4', 6), ('0.6', 10), ('0.8', 12), ('1.0', 25),
              ('1.2', 30), ('1.4', 33), ('1.6', 33), ('1.8', 20), ('2.0', 14)]
    
    with open(f'R{MODEL}_fit_params.txt', 'w') as f:
        for string_E, npeaks in models:
            params, errors = plot_fit(string_E, N, npeaks)
            if params:
                f.write(f'{string_E} {params[0]:.3e} {errors[0]:.3e} {params[1]:.3e} {errors[1]:.3e} {params[2]:.3e} {errors[2]:.3e}\n')
