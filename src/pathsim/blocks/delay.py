#########################################################################################
##
##                             TIME DOMAIN DELAY BLOCK 
##                                (blocks/delay.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.adaptivebuffer import AdaptiveBuffer


# BLOCKS ================================================================================

class Delay(Block):
    """Delays the input signal by a time constant 'tau' in seconds. 

    Mathematically this block creates a time delay of the input signal like this:

    .. math::
    
        y(t) = 
        \\begin{cases}
        x(t - \\tau) & , t \\geq \\tau \\\\
        0            & , t < \\tau
        \\end{cases}

    Note
    ----
    The internal adaptive buffer uses interpolation for the evaluation. This is 
    required to be compatible with variable step solvers. It has a drawback however. 
    The order of the ode solver used will degrade when this block is used, due to 
    the interpolation.

    
    Note
    ----
    This block supports vector input, meaning we can have multiple parallel 
    delay paths through this block.


    Example
    -------
    The block is initialized like this:

    .. code-block:: python
        
        #5 time units delay
        D = Delay(tau=5)
    
    Parameters
    ----------
    tau : float
        delay time constant

    Attributes
    ----------
    _buffer : AdaptiveBuffer
        internal interpolatable adaptive rolling buffer
    """

    def __init__(self, tau=1e-3):
        super().__init__()

        #time delay in seconds 
        self.tau = tau

        #create adaptive buffer
        self._buffer = AdaptiveBuffer(self.tau)


    def __len__(self):
        #no passthrough by definition
        return 0


    def reset(self):
        super().reset()

        #clear the buffer
        self._buffer.clear()


    def update(self, t):
        """Evaluation of the buffer at different times 
        via interpolation.

        Parameters
        ----------
        t : float
            evaluation time
        """

        #retrieve value from buffer
        y = self._buffer.get(t)
        self.outputs.update_from_array(y)


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

        #add new value to buffer
        self._buffer.add(t, self.inputs.to_array())