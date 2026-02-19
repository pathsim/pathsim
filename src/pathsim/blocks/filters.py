#########################################################################################
##
##                               FILTERS (filters.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from scipy.signal import butter, tf2ss

from math import factorial

from .lti import StateSpace
from ..utils.register import Register
from ..utils.mutable import mutable


# FILTER BLOCKS =========================================================================

@mutable
class ButterworthLowpassFilter(StateSpace):
    """Direct implementation of a low pass butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    Parameters
    ----------
    Fc : float
        corner frequency of the filter in [Hz]
    n : int
        filter order
    """

    input_port_labels = {"in":0}
    output_port_labels = {"out":0}

    def __init__(self, Fc=100, n=2):

        #filter parameters
        self.Fc = Fc
        self.n = n

        #use scipy.signal for filter design for unit frequency
        num, den = butter(n, 1.0, btype="low", analog=True, output="ba")
        A, B, C, D = tf2ss(num, den)

        #rescale to actual bandwidth and make statespace model
        omega_c = 2*np.pi*self.Fc
        super().__init__(omega_c*A, omega_c*B, C, D)


@mutable
class ButterworthHighpassFilter(StateSpace):
    """Direct implementation of a high pass butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    Parameters
    ----------
    Fc : float
        corner frequency of the filter in [Hz]
    n : int
        filter order
    """
    input_port_labels = {"in":0}
    output_port_labels = {"out":0}

    def __init__(self, Fc=100, n=2):

        #filter parameters
        self.Fc = Fc
        self.n = n

        #use scipy.signal for filter design for unit frequency
        num, den = butter(n, 1.0, btype="high", analog=True, output="ba")
        A, B, C, D = tf2ss(num, den)

        #rescale to actual bandwidth and make statespace model
        omega_c = 2*np.pi*self.Fc
        super().__init__(omega_c*A, omega_c*B, C, D)


@mutable
class ButterworthBandpassFilter(StateSpace):
    """Direct implementation of a bandpass butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    Parameters
    ----------
    Fc : list[float]
        corner frequencies (left, right) of the filter in [Hz]
    n : int
        filter order
    """
    input_port_labels = {"in":0}
    output_port_labels = {"out":0}

    def __init__(self, Fc=[50, 100], n=2):

        #filter parameters
        self.Fc = np.asarray(Fc)
        self.n = n

        if len(Fc) != 2:
            raise ValueError("'ButterworthBandpassFilter' requires two corner frequencies!")

        #use scipy.signal for filter design
        num, den = butter(n, 2*np.pi*self.Fc, btype="bandpass", analog=True, output="ba")

        #initialize parent block
        super().__init__(*tf2ss(num, den))


@mutable
class ButterworthBandstopFilter(StateSpace):
    """Direct implementation of a bandstop butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    Parameters
    ----------
    Fc : tuple[float], list[float]
        corner frequencies (left, right) of the filter in [Hz]
    n : int
        filter order
    """
    input_port_labels = {"in":0}
    output_port_labels = {"out":0}

    def __init__(self, Fc=[50, 100], n=2):

        #filter parameters
        self.Fc = np.asarray(Fc)
        self.n = n

        if len(Fc) != 2:
            raise ValueError("'ButterworthBandstopFilter' requires two corner frequencies!")

        #use scipy.signal for filter design
        num, den = butter(n, 2*np.pi*self.Fc, btype="bandstop", analog=True, output="ba")

        #initialize parent block
        super().__init__(*tf2ss(num, den))


@mutable
class AllpassFilter(StateSpace):
    """Direct implementation of a first order allpass filter, or a cascade 
    of n 1st order allpass filters
    
    .. math:: 

        H(s) = \\frac{s - 2\\pi f_s}{s + 2\\pi f_s}

    where f_s is the frequency, where the 1st order allpass has a 90 deg phase shift.
    
    Parameters
    ----------
    fs : float
        frequency for 90 deg phase shift of 1st order allpass
    n : int
        number of cascades
    """
    input_port_labels = {"in":0}
    output_port_labels = {"out":0}

    def __init__(self, fs=100, n=1):

        #filter parameters
        self.fs = fs
        self.n = n

        #1st order allpass for numerator and denominator (normalized frequency)
        num = [-1, 1]
        den = [1, 1]

        #higher order by convolution
        for _ in range(1, self.n):
            num = np.convolve(num, [-1, 1])
            den = np.convolve(den, [1, 1])

        #create statespace model
        A, B, C, D = tf2ss(num, den)

        #rescale to actual frequency and make statespace model
        omega_s = 2*np.pi*fs

        #initialize parent block
        super().__init__(omega_s*A, omega_s*B, C, D)