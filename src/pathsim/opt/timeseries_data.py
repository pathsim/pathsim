#########################################################################################
##
##                             TIME SERIES DATA CONTAINER
##                               (timeseries_data.py)
##
##                                 Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt


# CLASS =================================================================================

class TimeSeriesData:
    
    """Time series measurement container.

    Stores a measurement time base and associated data array. The time base is
    required to be strictly increasing. Data is normalized to be 1D or 2D with
    time aligned on axis 0.

    Parameters
    ----------
    time : array_like
        Time vector of shape (n,).
    data : array_like
        Measurement array of shape (n,), (n, m), or (m, n). If time is aligned on
        axis 1, the input is transposed automatically.
    name : str, optional
        Signal name for display and plotting.
    unit : str, optional
        Time unit label used for plotting.

    Notes
    -----
    The `time_info` dictionary stores simple plotting metadata:
    - `time_range`: dict with `start` and `end`
    - `units`: display string for the time axis
    """

    def __init__(self, time: np.ndarray, data: np.ndarray, name: str = "measurement", unit: str = "s"):
        t = np.asarray(time, dtype=float).reshape(-1)
        y = np.asarray(data, dtype=float)

        # Normalize data to 1D or 2D with time aligned on axis 0.
        if y.ndim == 1:
            y = y.reshape(-1)
            if t.size != y.size:
                raise ValueError("TimeSeriesData requires time and data with same length")
        elif y.ndim == 2:
            if y.shape[0] == t.size:
                pass
            elif y.shape[1] == t.size:
                y = y.T
            else:
                raise ValueError("TimeSeriesData requires data to align with time on one axis")
        else:
            raise ValueError("TimeSeriesData supports 1D or 2D data only")

        if t.size < 2:
            raise ValueError("TimeSeriesData requires at least 2 samples")
        if not np.all(np.diff(t) > 0):
            raise ValueError("TimeSeriesData requires strictly increasing time")

        self.time = t
        self.data = y
        self.name = str(name)
        
        self.time_info = {
            'time_range': 
                {
                    'start': t[0], 
                    'end': t[-1]
                },
            'units' : unit,
            }
        
        
    def plot(
        self,
        *,
        marker: str = "o",
        markersize: float = 6.0,
        markevery: int | None = None,
        linewidth: float = 1.5,
        alpha: float = 0.6,
    ):
        """Plot the time series.

        Parameters
        ----------
        marker : str, optional
            Marker style passed to `matplotlib.pyplot.plot`.
        markersize : float, optional
            Marker size passed to `matplotlib.pyplot.plot`.
        markevery : int, optional
            Plot every Nth marker. Use `None` to plot markers for all samples.
        linewidth : float, optional
            Line width passed to `matplotlib.pyplot.plot`.
        alpha : float, optional
            Alpha transparency passed to `matplotlib.pyplot.plot`.

        Notes
        -----
        For 2D data, each column is treated as an independent channel and plotted
        as a separate trace.
        """
        
        plt.figure(figsize=(8, 4))
        plot_kws = dict(
            marker=marker,
            markersize=markersize,
            markevery=markevery,
            linewidth=linewidth,
            alpha=alpha,
        )

        if self.data.ndim == 1:
            plt.plot(self.time, self.data, label=self.name, **plot_kws)
        else:
            for i in range(self.data.shape[1]):
                plt.plot(self.time, self.data[:, i], label=f"{self.name}_{i}", **plot_kws)
        
        plt.xlabel(f"Time ({self.time_info['units']})")
        plt.ylabel("Measurement")
        plt.title(f"Time Series: {self.name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

    @property
    def length(self) -> int:
        """Number of samples."""
        return self.time.size


    @property
    def duration(self) -> float:
        """Signal duration in time units."""
        return float(self.time[-1] - self.time[0])
    