#########################################################################################
##
##                             TIME DOMAIN NOISE SOURCES 
##                                  (blocks/noise.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..events.schedule import Schedule 


# NOISE SOURCE BLOCKS ===================================================================

class WhiteNoise(Block):
    """White noise source with uniform spectral density. Samples from
    distribution with 'sampling_period' and holds noise values constant
    for time bins (zero-order-hold).

    If no 'sampling_period' (None) is specified, every simulation timestep
    gets a new noise value. This is the default setting.

    Parameters
    ----------
    spectral_density : float
        noise spectral density
    noise : float
        internal noise value
    sampling_period : float, None
        time between noise samples

    Attributes
    ----------
    events : list[Schedule]
        scheduled event for periodic sampling
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(self, spectral_density=1, sampling_period=None):
        super().__init__()

        #block parameters
        self.spectral_density = spectral_density
        self.sampling_period = sampling_period

        #sampling produces discrete time behavior
        if sampling_period is None:

            #initial sample for non-discrete block
            self._sample = np.random.rand()

        else:

            #internal scheduled list event
            def _set(t):
                self.outputs[0] = self._random(self.sampling_period)

            self.events = [
                Schedule(
                    t_start=0,
                    t_period=sampling_period,
                    func_act=_set
                    )
                ]


    def __len__(self):
        return 0


    def _random(self, dt):
        """Generate random sample from scaled normal distribution"""
        return np.random.normal(0, 1) * np.sqrt(self.spectral_density/dt) 


    def sample(self, t, dt):
        """Sample from a normal distribution after successful timestep

        Parameters
        ----------
        t : float
            evaluation time for sampling
        dt : float
            integration timestep
        """
        if self.sampling_period is None:
            self.outputs[0] = self._random(dt)
           

    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs
    
        Parameters
        ----------
        t : float
            evaluation time
        """
        pass


class PinkNoise(Block):
    """Pink noise (1/f) source using the Voss-McCartney algorithm.

    Generates noise with power spectral density inversely proportional to 
    frequency. Samples from distribution with 'sampling_rate' and holds 
    noise values constant for time bins (zero-order-hold).
    
    The Voss-McCartney algorithm maintains ``num_octaves`` independent 
    random values. At each sample n, octaves are selectively updated based 
    on the binary representation of n:
    
    - Octave 0: updated every sample (when n & 1 == 1)
    - Octave 1: updated every 2nd sample (when n & 2 == 2)  
    - Octave 2: updated every 4th sample (when n & 4 == 4)
    - Octave k: updated every :math:`2^k` samples
    
    The pink noise output is the sum of all octaves, scaled to achieve the 
    desired spectral density:
    
    .. math::
    
        y[n] = \\sqrt{\\frac{S_0}{N \\cdot dt}} \\sum_{k=0}^{N-1} x_k[n]
    
    where :math:`S_0` is the spectral density, :math:`N` is ``num_octaves``,
    :math:`dt` is the sampling timestep, and :math:`x_k[n]` are the octave 
    values (each drawn from :math:`\\mathcal{N}(0, 1)` when updated).
    
    Note
    ----
    If no 'sampling_period' (None) is specified, every simulation timestep
    gets a new noise value. This is the default setting.

    Parameters
    ----------
    spectral_density : float
        noise spectral density :math:`S_0`
    num_octaves : int
        number of octaves (levels of randomness), default is 16
    sampling_period : float, None
        time between noise samples 

    Attributes
    ----------
    n_samples : int
        internal sample counter 
    octave_values : array[float]
        internal random numbers for octaves in the Voss-McCartney algorithm
    events : list[Schedule]
        scheduled event for periodic sampling
        
    References
    ----------
    .. [1] Voss, R. F., & Clarke, J. (1978). "1/f noise" in music: Music from 
           1/f noise. The Journal of the Acoustical Society of America, 63(1), 
           258-263.
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(self, spectral_density=1, num_octaves=16, sampling_period=None):
        super().__init__()

        #block parameters
        self.spectral_density = spectral_density
        self.num_octaves = num_octaves
        self.sampling_period = sampling_period
        self.n_samples = 0

        # Initialize the random values for each octave
        self.octave_values = np.random.normal(0, 1, self.num_octaves)

        #sampling produces discrete time behavior
        if sampling_period is not None:

            #internal scheduled list event
            def _set(t):
                self.outputs[0] = self._random(self.sampling_period)

            self.events = [
                Schedule(
                    t_start=0,
                    t_period=sampling_period,
                    func_act=_set
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()

        #reset counters and octave values
        self.n_samples = 0
        self.octave_values = np.random.normal(0, 1, self.num_octaves)


    def _random(self, dt):
        """Generate a pink noise sample using the Voss-McCartney algorithm.
        
        Uses bitwise operations on the sample counter to determine which 
        octaves to update. Octave k is updated every :math:`2^k` samples, 
        creating the characteristic 1/f power spectrum when all octaves 
        are summed.
        
        Parameters
        ----------
        dt : float
            timestep for scaling spectral density
            
        Returns
        -------
        float
            Pink noise sample scaled as :math:`\\sqrt{S_0 / (N \\cdot dt)}`
            where :math:`S_0` is spectral density and :math:`N` is num_octaves
        """
        # Increment the counter
        self.n_samples += 1

        # Use bitwise operations to determine which octaves to update
        # Find the rightmost zero bit in the binary representation
        mask, idx = self.n_samples, 0
        while mask & 1 == 0 and idx < self.num_octaves:
            mask >>= 1
            idx += 1

        # Update the selected octave with a new random value
        if idx < self.num_octaves:    
            self.octave_values[idx] = np.random.normal(0, 1)

        # Sum the octave values to produce the pink noise sample
        pink_sample = np.sum(self.octave_values)

        # Scale by sqrt(spectral_density / num_octaves / dt) to maintain 
        # consistent spectral density across different timesteps
        return pink_sample * np.sqrt(self.spectral_density/self.num_octaves/dt)


    def sample(self, t, dt):
        """Sample pink noise after successful timestep.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        dt : float
            integration timestep
        """
        if self.sampling_period is None:
            self.outputs[0] = self._random(dt)

            
    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs
    
        Parameters
        ----------
        t : float
            evaluation time
        """
        pass