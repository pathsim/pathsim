#########################################################################################
##
##                   SPECTRUM ANALYZER BLOCK (blocks/spectrum.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import csv

import numpy as np
import matplotlib.pyplot as plt

from ._block import Block

from ..utils.realtimeplotter import RealtimePlotter
from ..utils.deprecation import deprecated

from .._constants import COLORS_ALL


# BLOCKS FOR DATA RECORDING =============================================================

class Spectrum(Block):
    """Block for fourier spectrum analysis (basically a spectrum analyzer), computes 
    continuous time running fourier transform (RFT) of the incoming signal.
    
    A time threshold can be set by 't_wait' to start recording data only after the 
    simulation time is larger then the specified waiting time, i.e. 't - t_wait > dt'. 
    This is useful for recording the steady state after all the transients have settled.

    An exponential forgetting factor 'alpha' can be specified for realtime spectral 
    analysis. It biases the spectral components exponentially to the most recent signal 
    values by applying a single sided exponential window like this:

    .. math::

        \\int_0^t u(\\tau) \\exp(\\alpha (t-\\tau))  \\exp(-j \\omega \\tau)\\ d \\tau

    It is also known as the 'exponentially forgetting transform' (EFT) and a form of 
    short time fourier transform (STFT). It is implemented as a 1st order statespace model 
        
    .. math::

        \\dot{x} = - \\alpha  x +  \\exp(-j \\omega t) u

    where 'u' is the input signal and 'x' is the state variable that represents the 
    complex fourier coefficient to the frequency 'omega'. The ODE is integrated using the 
    numerical integration engine of the block.
    
    Example
    -------
    This is how to initialize it: 

    .. code-block:: python
    
        import numpy as np
        
        #linear frequencies (0Hz, DC -> 1kHz)
        sp1 = Spectrum(
            freq=np.linspace(0, 1e3, 100),
            labels=['x1', 'x2'] #labels for two inputs
            )

        #log frequencies (1Hz -> 1kHz)
        sp2 = Spectrum(
            freq=np.logspace(0, 3, 100)
            )
        
        #log frequencies including DC (0Hz, DC + 1Hz -> 1kHz)
        sp3 = Spectrum(
            freq=np.hstack([0.0, np.logspace(0, 3, 100)])
            )

        #arbitrary frequencies
        sp4 = Spectrum(
            freq=np.array([0, 0.5, 20, 1e3])
            )

    Note
    ----
    This block is relatively slow! But it is valuable for long running simulations 
    with few evaluation frequencies, where just FFT'ing the time series data 
    wouldnt be efficient OR if only the evaluation at weirdly spaced frequencies 
    is required. Otherwise its more efficient to just do an FFT on the time 
    series recording after the simulation has finished.
    

    Parameters
    ----------
    freq : array[float] 
        list of evaluation frequencies for RFT, can be arbitrarily spaced
    t_wait : float 
        wait time before starting RFT
    alpha : float
        exponential forgetting factor for realtime spectrum
    labels : list[str]
        labels for the inputs
    """

    def __init__(self, freq=[], t_wait=0.0, alpha=0.0, labels=[]):
        super().__init__()
        
        #time delay until start recording
        self.t_wait = t_wait

        #last valid timestep sample for waiting
        self.t_sample = 0.0

        #local integration time
        self.time = 0.0

        #forgetting factor
        self.alpha = alpha

        #labels for plotting and saving data
        self.labels = labels

        #frequency
        self.freq = np.array(freq)
        self.omega = 2.0 * np.pi * self.freq

        #initial state for integration engine
        self.initial_value = 0.0


    def __len__(self):
        return 0

    
    def _kernel(self, x, u, t):
        """Helper method that defines the kernel for the internal 
        running fourier transform (RFT)
        """
        if self.alpha == 0: return np.kron(u, np.exp(-1j * self.omega * t))
        return np.kron(u, np.exp(-1j * self.omega * t)) - self.alpha * x


    def reset(self):
        super().reset()

        #local integration time
        self.time = 0.0


    def read(self):
        """Read the recorded spectrum

        Example
        -------
        This is how to get the recorded spectrum:

        .. code-block:: python

            import numpy as np
            
            #linear frequencies (0Hz, DC -> 1kHz)
            spc = Spectrum(
                freq=np.linspace(0, 1e3, 100),
                labels=['x1', 'x2'] #labels for two inputs
                )
            
            #... run the simulation ...
            
            #read the complex spectra of x1 and x2
            freq, [X1, X2] = spc.read()

        Returns
        -------
        freq : array[float]
            evaluation frequencies
        spec : array[complex]
            complex spectrum
        """

        #just return zeros if no engine initialized
        if self.engine is None:
            return self.freq, [np.zeros_like(self.freq)]*len(self.inputs)

        #catch case where time is still zero
        if self.time == 0.0:
            return self.freq, [np.zeros_like(self.freq)]*len(self.inputs)

        #get state from engine
        state = self.engine.state

        #catch case where state has not been updated
        if np.all(state == self.engine.initial_value):
            return self.freq, [np.zeros_like(self.freq)]*len(self.inputs)

        #reshape state into spectra
        spec = np.reshape(state, (-1, len(self.freq)))

        #rescale spectrum and return it
        if self.alpha != 0.0:
            return self.freq, spec * self.alpha / (1.0 - np.exp(-self.alpha*self.time))

        #return spectrum from RFT
        return self.freq, spec/self.time


    @deprecated(version="1.0.0", reason="its against pathsims philosophy")
    def collect(self):
        """Yield (category, id, data) tuples for recording blocks to simplify 
        global data collection from all recording blocks.
        """
        freq, data = self.read()
        if data is not None:
            yield (
                "spectrum", 
                id(self), 
                {
                    "freq": freq,
                    "data": data,
                    "labels": self.labels,
                    }
                )


    def solve(self, t, dt):
        """advance solution of implicit update equation of the solver

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        ------- 
        error : float
            solver residual norm
        """
        if self.t_sample >= self.t_wait:

            #effective time for integration
            self.time = t - self.t_wait

            #advance solution of implicit update equation (no jacobian)
            f = self._kernel(self.engine.state, self.inputs.to_array(), self.time)
            return self.engine.solve(f, None, dt)

        #no error 
        return 0.0


    def step(self, t, dt):
        """compute timestep update with integration engine
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """
        if self.t_sample >= self.t_wait:

            #effective time for integration
            self.time = t - self.t_wait
            
            #compute update step with integration engine
            f = self._kernel(self.engine.state, self.inputs.to_array(), self.time)
            return self.engine.step(f, dt)

        #no error estimate
        return True, 0.0, None


    def sample(self, t, dt):
        """sample time of successfull timestep for waiting period
    
        Parameters
        ----------
        t : float
            sampling time
        dt : float
            integration timestep
        """
        self.t_sample = t



    def plot(self, *args, **kwargs):
        """Directly create a plot of the recorded data for visualization.

        The 'fig' and 'ax' objects are accessible as attributes of the 'Spectrum' instance 
        from the outside for saving, or modification, etc.

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

        #just return 'None' if no engine initialized
        if self.engine is None:
            return None

        #get data
        freq, data = self.read()        

        #initialize figure
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4), tight_layout=True, dpi=120)

        #custom colors
        self.ax.set_prop_cycle(color=COLORS_ALL)

        #plot magnitude in dB and add label
        for p, d in enumerate(data):
            lb = self.labels[p] if p < len(self.labels) else f"port {p}"
            self.ax.plot(freq, abs(d), *args, **kwargs, label=lb)

        #legend labels from ports
        self.ax.legend(fancybox=False)

        #other plot settings
        self.ax.set_xlabel("freq [Hz]")
        self.ax.set_ylabel("magnitude")
        self.ax.grid()

        # Legend picking functionality
        lines = self.ax.get_lines()  # Get the lines from the plot
        leg = self.ax.get_legend()   # Get the legend

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
            self.fig.canvas.draw()  

        #enable picking
        self.fig.canvas.mpl_connect("pick_event", on_pick)

        #show the plot without blocking following code
        plt.show(block=False)

        #return figure and axis for outside manipulation
        return self.fig, self.ax


    def save(self, path="spectrum.csv"):
        """Save the recording of the spectrum to a csv file        
        
        Parameters
        ----------
        path : str
            path where to save the recording as a csv file
        """

        #check path ending
        if not path.lower().endswith(".csv"):
            path += ".csv"

        #get data
        freq, data = self.read() 

        #number of ports and labels
        P, L = len(data), len(self.labels)

        #construct port labels
        port_labels = [self.labels[p] if p < L else f"port {p}" for p in range(P)]

        #make csv header
        header = ["freq [Hz]"]
        for l in port_labels:
            header.extend([f"Re({l})", f"Im({l})"])

        #write to csv file
        with open(path, "w", newline="") as file:
            wrt = csv.writer(file)

            #write the header to csv file
            wrt.writerow(header)

            #write each sample to the csv file
            for f, *dta in zip(freq, *data):
                sample = [f]
                for d in dta:
                    sample.extend([np.real(d), np.imag(d)])
                wrt.writerow(sample)


    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs
    
        Note
        ----
        Spectrum block has no passthrough, so the 'update' method 
        is optimized for this case        

        Parameters
        ----------
        t : float
            evaluation time
        """
        pass


@deprecated(version="1.0.0")
class RealtimeSpectrum(Spectrum):
    """An extension of the 'Spectrum' block that also initializes a realtime plotter.

    Creates an interactive plotting window while the simulation is running.
    Otherwise implements the same functionality as the regular 'Spectrum' block.

    Note
    ----
    Due to the plotting being relatively expensive, including this block slows down
    the simulation significantly but may still be valuable for debugging and testing.

    Parameters
    ----------
    freq : array[float]
        list of evaluation frequencies for RFT, can be arbitrarily spaced
    t_wait : float
        wait time before starting RFT
    alpha : float
        exponential forgetting factor for realtime spectrum
    labels : list[str]
        labels for the inputs
    """

    def __init__(self, freq=[], t_wait=0.0, alpha=0.0, labels=[]):
        super().__init__(freq, t_wait, alpha, labels)

        #initialize realtime plotter
        self.plotter = RealtimePlotter(
            update_interval=0.1,
            labels=labels,
            x_label="freq [Hz]",
            y_label="magnitude"
            )


    def step(self, t, dt):
        """compute timestep update with integration engine
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """

        #effective time for integration
        _t = t - self.t_wait
        if _t > dt:

            #update local integtration time
            self.time = _t

            if self.time > 2*dt:
                #update realtime plotter
                _, data = self.read()
                self.plotter.update_all(self.freq, abs(data))

            #compute update step with integration engine
            f = self._kernel(self.engine.state, self.inputs.to_array(), _t)
            return self.engine.step(f, dt)

        #no error estimate
        return True, 0.0, None