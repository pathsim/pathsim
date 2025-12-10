#########################################################################################
##
##                                       SCOPE BLOCK 
##                                (pathsim/blocks/scope.py)
##
##               This module defines blocks for recording time domain data
##
#########################################################################################

# IMPORTS ===============================================================================

import csv

import warnings

import numpy as np
import matplotlib.pyplot as plt

from ._block import Block
from ..events.schedule import Schedule 
from ..utils.realtimeplotter import RealtimePlotter

from .._constants import COLORS_ALL


# BLOCKS FOR DATA RECORDING =============================================================

class Scope(Block):
    """Block for recording time domain data with variable sampling rate.
    
    A time threshold can be set by `t_wait` to start recording data after the simulation 
    time is larger then the specified waiting time, i.e. `t - t_wait > 0`. 
    This is useful for recording data only after all the transients have settled.
    
    The block uses an interal `Schedule` event, when `sampling_rate` is provided, 
    otherwise it just samples at every simulation timestep.

    Parameters
    ----------
    sampling_rate : int, None
        number of samples per time unit, default is every timestep
    t_wait : float
        wait time before starting recording, optional
    labels : list[str]
        labels for the scope traces, and for the csv, optional

    Attributes
    ----------
    recording : dict
        recording, where key is time, and value the recorded values
    _sample_next_timestep : bool
        flag to indicate this is a timestep to sample, only used for event based sampling 
        when `sampling_rate` is provided as an arg
    events : list[Schedule]
        internal scheduled event for periodic input sampling when `sampling_rate` is provided
    """
    
    def __init__(self, sampling_rate=None, t_wait=0.0, labels=None):
        super().__init__()
        
        #time delay until start recording
        self.t_wait = t_wait

        #params for sampling
        self.sampling_rate = sampling_rate

        #labels for plotting and saving data
        self.labels = labels if labels is not None else []

        #set recording data and time
        self.recording = {}

        #sampling produces discrete time behavior
        if not (sampling_rate is None):

            #flag to indicate this is a timestep to sample
            self._sample_next_timestep = False
            
            #internal scheduled list event
            def _sample(t):
                self._sample_next_timestep = True

            self.events = [
                Schedule(
                    t_start=t_wait,
                    t_period=sampling_rate,
                    func_act=_sample
                    )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()

        #reset recording data and time
        self.recording = {}


    def read(self):
        """Return the recorded time domain data and the 
        corresponding time for all input ports

        Returns
        -------
        time : array[float]
            recorded time points
        data : array[obj]
            recorded data points
        """

        #just return 'None' if no recording available
        if not self.recording: return None, None

        #reformat the data from the recording dict
        time = np.array(list(self.recording.keys()))
        data = np.array(list(self.recording.values())).T
        return time, data


    def sample(self, t, dt):
        """Sample the data from all inputs, and overwrites existing timepoints, 
        since we use a dict for storing the recorded data.
 
        If `sampling_rate` is provided, this depends on the flag `_sample_next_timestep`, 
        set by the internal `Schedule` event.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        """
        if self.sampling_rate is None: 
            if t >= self.t_wait:
                self.recording[t] = self.inputs.to_array()
        elif self._sample_next_timestep:
            self.recording[t] = self.inputs.to_array()
            self._sample_next_timestep = False


    def plot(self, *args, **kwargs):
        """Directly create a plot of the recorded data for quick visualization and debugging.

        Parameters
        ----------
        args : tuple
            args for ax.plot
        kwargs : dict
            kwargs for ax.plot

        Returns
        -------
        fig : matplotlib.figure
            internal figure instance
        ax : matplotlib.axis
            internal axis instance
        """ 

        #just return 'None' if no recording available
        if not self.recording:
            warnings.warn("no recording available for plotting in 'Scope.plot'")
            return None, None

        #get data
        time, data = self.read() 

        #initialize figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4), tight_layout=True, dpi=120)
        
        #custom colors
        ax.set_prop_cycle(color=COLORS_ALL)
        
        #plot the recorded data
        for p, d in enumerate(data):
            lb = self.labels[p] if p < len(self.labels) else f"port {p}"
            ax.plot(time, d, *args, **kwargs, label=lb)

        #legend labels from ports
        ax.legend(fancybox=False)

        #other plot settings
        ax.set_xlabel("time [s]")
        ax.grid()

        # Legend picking functionality
        lines = ax.get_lines()  # Get the lines from the plot
        leg = ax.get_legend()   # Get the legend

        # Map legend lines to original plot lines
        lined = dict()  
        for legline, origline in zip(leg.get_lines(), lines):
            # Enable picking within 5 points tolerance
            legline.set_picker(5)  
            lined[legline] = origline

        def on_pick(event):
            legline = event.artist
            origline = lined[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)
            legline.set_alpha(1.0 if visible else 0.2)
            # Redraw the figure
            fig.canvas.draw()  

        #enable picking
        fig.canvas.mpl_connect("pick_event", on_pick)

        #show the plot without blocking following code
        plt.show(block=False)

        #return figure and axis for outside manipulation
        return fig, ax


    def plot2D(self, *args, axes=(0, 1), **kwargs):
        """Directly create a 2D plot of the recorded data for quick visualization and debugging.

        Parameters
        ----------
        args : tuple
            args for ax.plot
        axes : tuple[int]
            axes / ports to select for 2d plot
        kwargs : dict
            kwargs for ax.plot

        Returns
        -------
        fig : matplotlib.figure
            internal figure instance
        ax : matplotlib.axis
            internal axis instance
        """ 

        #just return 'None' if no recording available
        if not self.recording:
            warnings.warn("no recording available for plotting in 'Scope.plot2D'")
            return None, None

        #get data
        time, data = self.read() 

        #not enough channels -> early exit
        if len(data) < 2 or len(axes) != 2:
            warnings.warn("not enough channels for plotting in 'Scope.plot2D'")
            return None, None

        #axes selected not available -> early exit
        ax1_idx, ax2_idx = axes
        if not (0 <= ax1_idx < data.shape[0] and 0 <= ax2_idx < data.shape[0]):
             warnings.warn(f"Selected axes {axes} out of bounds for data shape {data.shape}")
             return None, None 

        #initialize figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), tight_layout=True, dpi=120)
        
        #custom colors
        ax.set_prop_cycle(color=COLORS_ALL)

        #unpack data for selected axes
        d1 = data[ax1_idx]
        d2 = data[ax2_idx]

        #plot the data
        ax.plot(d1, d2, *args, **kwargs)

        #axis labels
        l1 = self.labels[ax1_idx] if ax1_idx < len(self.labels) else f"port {ax1_idx}"
        l2 = self.labels[ax2_idx] if ax2_idx < len(self.labels) else f"port {ax2_idx}"
        ax.set_xlabel(l1)
        ax.set_ylabel(l2)
        
        ax.grid()

        #show the plot without blocking following code
        plt.show(block=False)

        #return figure and axis for outside manipulation
        return fig, ax


    def plot3D(self, *args, axes=(0, 1, 2), **kwargs):
        """Directly create a 3D plot of the recorded data for quick visualization.

        Parameters
        ----------
        args : tuple
            args for ax.plot
        axes : tuple[int]
            indices of the three data channels (ports) to plot (default: (0, 1, 2)).
        kwargs : dict
            kwargs for ax.plot

        Returns
        -------
        fig : matplotlib.figure
            internal figure instance.
        ax : matplotlib.axes._axes.Axes3D
            internal 3D axis instance.
        """
        
        #check if recording is available
        if not self.recording:
            warnings.warn("no recording available for plotting in 'Scope.plot3D'")
            return None, None 

        #read the recorded data
        time, data = self.read()

        #check if enough channels are available
        if data.shape[0] < 3 or len(axes) != 3:
            warnings.warn(f"Need at least 3 channels for plot3D, got {data.shape[0]}. Or axes argument length is not 3.")
            return None, None

        #check if selected axes are valid
        ax1_idx, ax2_idx, ax3_idx = axes
        if not (0 <= ax1_idx < data.shape[0] and
                0 <= ax2_idx < data.shape[0] and
                0 <= ax3_idx < data.shape[0]):
            warnings.warn(f"Selected axes {axes} out of bounds for data shape {data.shape}")
            return None, None 

        #initialize 3D figure
        fig = plt.figure(figsize=(6, 6), dpi=120)
        ax = fig.add_subplot(111, projection='3d')

        #custom colors
        ax.set_prop_cycle(color=COLORS_ALL)

        #unpack data for selected axes
        d1 = data[ax1_idx]
        d2 = data[ax2_idx]
        d3 = data[ax3_idx]

        #plot the 3D data
        ax.plot(d1, d2, d3, *args, **kwargs)

        #set axis labels using provided labels or default port numbers
        label1 = self.labels[ax1_idx] if ax1_idx < len(self.labels) else f"port {ax1_idx}"
        label2 = self.labels[ax2_idx] if ax2_idx < len(self.labels) else f"port {ax2_idx}"
        label3 = self.labels[ax3_idx] if ax3_idx < len(self.labels) else f"port {ax3_idx}"
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        ax.set_zlabel(label3)

        #show the plot without blocking
        plt.show(block=False)

        return fig, ax


    def save(self, path="scope.csv"):
        """Save the recording of the scope to a csv file.

        Parameters
        ----------
        path : str
            path where to save the recording as a csv file
        """

        #check path ending
        if not path.lower().endswith(".csv"):
            path += ".csv"

        #get data
        time, data = self.read() 

        #number of ports and labels
        P, L = len(data), len(self.labels)

        #make csv header
        header = ["time [s]", *[self.labels[p] if p < L else f"port {p}" for p in range(P)]]

        #write to csv file
        with open(path, "w", newline="") as file:
            wrt = csv.writer(file)

            #write the header to csv file
            wrt.writerow(header)

            #write each sample to the csv file
            for sample in zip(time, *data):
                wrt.writerow(sample)


    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs
    
        Note
        ----
        Scope has no passthrough, so the 'update' method 
        is optimized for this case (does nothing)       

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence 
            control (always '0.0' because sink-type)
        """
        return 0.0




class RealtimeScope(Scope):
    """An extension of the 'Scope' block that also initializes a realtime plotter 
    that creates an interactive plotting window while the simulation is running.

    Otherwise implements the same functionality as the regular 'Scope' block.

    Note
    -----
    Due to the plotting being relatively expensive, including this block 
    slows down the simulation significantly but may still be valuable for 
    debugging and testing.

    Parameters
    ----------
    sampling_rate : int, None
        number of samples time unit, default is every timestep
    t_wait : float
        wait time before starting recording
    labels : list[str] 
        labels for the scope traces, and for the csv
    max_samples : int, None
        number of samples for realtime display, all per default

    Attributes
    ----------
    plotter : RealtimePlotter
        instance of a RealtimePlotter
    """

    def __init__(self, sampling_rate=None, t_wait=0.0, labels=[], max_samples=None):
        super().__init__(sampling_rate, t_wait, labels)

        #initialize realtime plotter
        self.plotter = RealtimePlotter(
            max_samples=max_samples, 
            update_interval=0.1, 
            labels=labels, 
            x_label="time [s]", 
            y_label=""
            )

        warnings.warn("'RealtimeScope' block will be deprecated with release version 1.0.0")


    def sample(self, t, dt):
        """Sample the data from all inputs, and overwrites existing timepoints, 
        since we use a dict for storing the recorded data.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        """
        if (self.sampling_rate is None or t * self.sampling_rate > len(self.recording)):
            values = self.inputs.to_array()
            self.plotter.update(t, values)
            if t >= self.t_wait: 
                self.recording[t] = values
