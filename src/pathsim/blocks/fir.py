#########################################################################################
##
##               DISCRETE-TIME FINITE-IMPULSE-RESPONSE (FIR) FILTER BLOCK
##                                  (blocks/fir.py) 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
from collections import deque

from ._block import Block
from ..utils.register import Register
from ..events.schedule import Schedule
from ..utils.mutable import mutable


# FIR FILTER BLOCK ======================================================================

@mutable
class FIR(Block):
    """Models a discrete-time Finite-Impulse-Response (FIR) filter.

    This block applies an FIR filter to an input signal sampled periodically.
    The output at each sample time is a weighted sum of the current and a finite number 
    of past input samples. The operation is triggered by a scheduled event.

    Functionality:
    
    .. math::

        y[n] = b[0] x[n] + b[1] x[n-1] + \\dots + b[N] x[n-N]
    
    where `b` are the filter coefficients and `N` is the filter order (number of 
    coefficients - 1).

    1. Samples the input `inputs[0]` at intervals of `T`, starting after delay `tau`.
    2. Stores the current and past `len(coefficients) - 1` input samples in an internal buffer.
    3. Computes the filter output using the dot product of the coefficients
       and the buffered input samples.
    4. Outputs the result on `outputs[0]`.
    5. Holds the output constant between updates.

    Parameters
    ----------
    coeffs : array_like
        List or numpy array of FIR filter coefficients [b0, b1, ..., bN].
        The number of coefficients determines the filter's order and memory.
    T : float, optional
        Sampling period (time between input samples and output updates). Default is 1.
    tau : float, optional
        Initial delay before the first sample is processed. Default is 0.

    Input Ports
    -----------
    inputs[0] : float
        Input signal sample at the current time step.

    Output Ports
    ------------
    outputs[0] : float
        Filtered output signal sample.

    Attributes
    ----------
    buffer : deque
        Internal buffer storing the most recent input samples.
    events : list[Schedule]
        Internal scheduled event triggering the filter calculation.
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}


    def __init__(self, coeffs=[1.0], T=1, tau=0):
        super().__init__()

        #block params
        self.coeffs = np.array(coeffs)
        self.T = T
        self.tau = tau

        #buffer to store the last N+1 input samples (current + N past)
        n = len(self.coeffs)
        self._buffer = deque([0.0]*n, maxlen=n)

        def _update_fir(t):

            #update internal buffer
            self._buffer.appendleft(self.inputs[0])

            #compute the FIR output: y[n] = sum(b[k] * x[n-k])
            current_output = np.dot(self.coeffs, self._buffer)

            #update the block's output port
            self.outputs[0] = current_output

        #internal scheduled event
        self.events = [
            Schedule(
                t_start=self.tau,
                t_period=self.T,
                func_act=_update_fir
                )
            ]


    def reset(self):
        """Resets the filter state (buffer) and output."""
        super().reset()
        n = len(self.coeffs)
        self._buffer = deque([0.0]*n, maxlen=n)


    def __len__(self):
        """This block has no direct passthrough"""
        return 0