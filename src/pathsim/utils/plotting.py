#########################################################################################
##
##                              PLOTTING HELPERS
##                            (utils/plotting.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt

from matplotlib.backend_bases import FigureManagerBase


# FUNCTIONS =============================================================================

def show(fig):
    """Show a figure without blocking the following code, if the matplotlib
    backend opens figures in a window.

    Backends without a window, such as the inline backend of Jupyter or 'Agg',
    use the generic 'FigureManagerBase'. Jupyter displays the figure at the end
    of the cell by itself, so the figure stays open for changes after the plot
    call, for example 'ax.set_title(...)'.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to show
    """
    if type(fig.canvas.manager) is FigureManagerBase:
        return
    plt.show(block=False)
