#########################################################################################
##
##                             TIME DOMAIN DELAY BLOCK
##                                (blocks/delay.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from collections import deque

from ._block import Block

from ..utils.adaptivebuffer import AdaptiveBuffer
from ..events.schedule import Schedule
from ..utils.mutable import mutable


# BLOCKS ================================================================================

@mutable
class Delay(Block):
    """Delays the input signal by a time constant 'tau' in seconds.

    Supports two modes of operation:

    **Continuous mode** (default, ``sampling_period=None``):
    Uses an adaptive interpolating buffer for continuous-time delay.

    .. math::

        y(t) =
        \\begin{cases}
        x(t - \\tau) & , t \\geq \\tau \\\\
        0            & , t < \\tau
        \\end{cases}

    **Discrete mode** (``sampling_period`` provided):
    Uses a ring buffer with scheduled sampling events for N-sample delay,
    where ``N = round(tau / sampling_period)``.

    .. math::

        y[k] = x[k - N]

    Note
    ----
    In continuous mode, the internal adaptive buffer uses interpolation for
    the evaluation. This is required to be compatible with variable step solvers.
    It has a drawback however. The order of the ode solver used will degrade
    when this block is used, due to the interpolation.


    Note
    ----
    This block supports vector input, meaning we can have multiple parallel
    delay paths through this block.


    Example
    -------
    Continuous-time delay:

    .. code-block:: python

        #5 time units delay
        D = Delay(tau=5)

    Discrete-time N-sample delay (10 samples):

    .. code-block:: python

        D = Delay(tau=0.01, sampling_period=0.001)

    Parameters
    ----------
    tau : float
        delay time constant in seconds
    sampling_period : float, None
        sampling period for discrete mode, default is continuous mode

    Attributes
    ----------
    _buffer : AdaptiveBuffer
        internal interpolatable adaptive rolling buffer (continuous mode)
    _ring : deque
        internal ring buffer for N-sample delay (discrete mode)
    """

    def __init__(self, tau=1e-3, sampling_period=None):
        super().__init__()

        #time delay in seconds
        self.tau = tau

        #params for sampling
        self.sampling_period = sampling_period

        if sampling_period is None:

            #continuous mode: adaptive buffer with interpolation
            self._buffer = AdaptiveBuffer(self.tau)

        else:

            #discrete mode: ring buffer with N-sample delay
            self._n = max(1, round(self.tau / self.sampling_period))
            self._ring = deque([0.0] * self._n, maxlen=self._n + 1)

            #flag to indicate this is a timestep to sample
            self._sample_next_timestep = False

            #internal scheduled event for periodic sampling
            def _sample(t):
                self._sample_next_timestep = True

            self.events = [
                Schedule(
                    t_start=0,
                    t_period=sampling_period,
                    func_act=_sample
                    )
            ]


    def __len__(self):
        #no passthrough by definition
        return 0


    def reset(self):
        super().reset()

        if self.sampling_period is None:
            #clear the adaptive buffer
            self._buffer.clear()
        else:
            #clear the ring buffer
            self._ring.clear()
            self._ring.extend([0.0] * self._n)


    def update(self, t):
        """Evaluation of the buffer at different times
        via interpolation (continuous) or ring buffer lookup (discrete).

        Parameters
        ----------
        t : float
            evaluation time
        """

        if self.sampling_period is None:
            #continuous mode: retrieve value from buffer
            y = self._buffer.get(t)
            self.outputs.update_from_array(y)
        else:
            #discrete mode: output the oldest value in the ring buffer
            self.outputs[0] = self._ring[0]


    def sample(self, t, dt):
        """Sample input values and time of sampling
        and add them to the buffer.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        dt : float
            integration timestep
        """

        if self.sampling_period is None:
            #continuous mode: add new value to buffer
            self._buffer.add(t, self.inputs.to_array())
        else:
            #discrete mode: only sample on scheduled events
            if self._sample_next_timestep:
                self._ring.append(self.inputs[0])
                self._sample_next_timestep = False