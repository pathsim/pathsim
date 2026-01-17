#########################################################################################
##
##                                 COMPARATOR BLOCK
##                              (blocks/comparator.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..events.zerocrossing import ZeroCrossing


# MIXED SIGNAL BLOCKS ===================================================================

class Comparator(Block):
    """Comparator block that sets the output to '1' it the input 
    signal crosses a predefined threshold and to '-1' if it 
    crosses in the reverse direction. 

    This is realized by the block spawning a zero-crossing 
    event detector that watches the input of the block and 
    locates the transition up to a tolerance. 
    
    The block output is determined by a simple sign check in
    the 'update' method.

    Parameters
    ----------
    threshold : float
        threshold value for the comparator
    tolerance : float
        tolerance for zero crossing detection    
    span : list[float] or tuple[float], optional
        output value range [min, max]
    
    Attributes
    ----------
    events : list[ZeroCrossing]
        internal zero crossing event
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out":0}


    def __init__(self, threshold=0, tolerance=1e-4, span=[-1, 1]):
        super().__init__()

        #block parameters
        self.threshold = threshold
        self.tolerance = tolerance
        self.span = span

        def func_evt(t):
            return self.inputs[0] - self.threshold

        #internal event for transition detection
        self.events = [
            ZeroCrossing(
                func_evt=func_evt, 
                tolerance=tolerance
                )
            ]


    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs
    
        Note
        ----
        no direct passthrough, so the 'update' method 
        is optimized for this case        

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence 
            control (here '0.0' because discrete block)
        """

        if self.inputs[0] >= self.threshold:
            self.outputs[0] = max(self.span)
        else:
            self.outputs[0] = min(self.span)
