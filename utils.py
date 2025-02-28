import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def savefig(fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight', pad_inches: float = 0.1, transparent: bool = False) -> None:
    """
    Save the given matplotlib figure to a file with enhanced options for better quality.
    
    Parameters:
    - fig: The matplotlib Figure object to save.
    - filename: The file name or path where the figure will be saved.
    - dpi: The resolution in dots per inch (default is 300 for high-quality).
    - bbox_inches: Adjusts bounding box ('tight' will minimize excess whitespace).
    - pad_inches: Amount of padding around the figure when bbox_inches is 'tight' (default is 0.1).
    - transparent: Whether to save the plot with a transparent background (default is False).
    """
    
    try:
        fig.savefig(f'figs/{filename}', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent, format='pdf')
        print(f'Plot successfully saved to {filename} with dpi={dpi}, bbox_inches={bbox_inches}, pad_inches={pad_inches}, transparent={transparent}')
    except Exception as e:
        print(f"Error saving plot to {filename}: {e}")

def set_axes(ax: plt.Axes, xlabel: str, ylabel: str, xscale: str = 'linear', yscale: str = 'linear', xlim: tuple = None, ylim: tuple = None) -> None:
    """
    Set the properties for the axes of a plot.
    
    Parameters:
    - ax: Matplotlib Axes object.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - xscale: Scale of the x-axis ('linear' or 'log').
    - xlim: Limits for the x-axis (min, max).
    - ylim: Limits for the y-axis (min, max).
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Validate and set axis scale
    if xscale in ['linear', 'log']:
        ax.set_xscale(xscale)
    else:
        print(f"Invalid xscale '{xscale}', defaulting to 'log'.")
        ax.set_xscale('log')

    # Validate and set axis scale
    if yscale in ['linear', 'log']:
        ax.set_yscale(yscale)
    else:
        print(f"Invalid yscale '{yscale}', defaulting to 'log'.")
        ax.set_yscale('log')
        
    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

def find_first_peaks(x, y, npeaks=3):
    # Find all peaks
    peaks, _ = find_peaks(y)
    
    # Check if there are at least N peaks
    if len(peaks) < npeaks:
        raise ValueError(f'Less than {npeaks} peaks found in the data.')
    
    # Get the positions of the first three peaks
    first_peaks = peaks[:npeaks]
    
    # Get the x positions of the first three peaks
    peak_positions = x[first_peaks]
    
    return peak_positions, first_peaks