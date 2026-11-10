#########################################################################################
##
##                                 AMPLIFIER BLOCK 
##                              (blocks/amplifier.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..optim.operator import Operator
from ..utils.carrier import ports


# SISO BLOCKS ===========================================================================

class Amplifier(Block):
    """Amplifies the input signal by multiplication with a constant gain term.

    Like this:

    .. math::
        
        y(t) = \\mathrm{gain} \\cdot u(t)

    
    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called 
    multiple times per timestep, each time when `Simulation._update(t)` is 
    called in the global simulation loop.

        
    Example
    -------
    The block is initialized like this:

    .. code-block:: python
        
        #amplification by factor 5
        A = Amplifier(gain=5)


    With a carrier, input and output are a stream of multiple channels and
    every channel is amplified, by one gain or by a gain per channel

    .. code-block:: python

        A = Amplifier(gain=[1.0, 2.0, 3.0], carrier=Species())

        C = Connection(A["out"], B["in"])

    Parameters
    ----------
    gain : float | array_like
        amplifier gain, one per channel with a carrier
    carrier : Carrier | None
        layout of the input and output stream, the operation applies to
        every channel; a single channel if 'None'

        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, gain=1.0, carrier=None):

        self.carrier = carrier
        if carrier is not None:
            self.input_port_labels = ports(carrier, ["in"])
            self.output_port_labels = ports(carrier, ["out"])

        super().__init__()

        self.gain = gain

        self.op_alg = Operator(
            func=lambda x: x*self.gain, 
            jac=lambda x: self.gain*np.eye(len(x))
            )


    def update(self, t):
        """update system equation in fixed point loop

        Parameters
        ----------
        t : float
            evaluation time
        """
        u = self.inputs.to_array()
        y = self.op_alg(u)
        self.outputs.update_from_array(y)
