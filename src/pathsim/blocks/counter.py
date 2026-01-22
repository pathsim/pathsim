#########################################################################################
##
##                                  COUNTER BLOCK
##                               (blocks/counter.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..events.zerocrossing import ZeroCrossing, ZeroCrossingUp, ZeroCrossingDown


# MIXED SIGNAL BLOCKS ===================================================================

class Counter(Block):
    """Counts the number of detected bidirectional threshold crossings.

    Uses zero-crossing events for the detection and sets the output 
    accordingly.

    Parameters
    ----------
    start : int
        counter start (initial condition)
    threshold : float
        threshold for zero crossing
    
    Attributes
    ----------
    E : ZeroCrossing
        internal event manager
    events : list[ZeroCrossing]
        internal zero crossing event
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}
    

    def __init__(self, start=0, threshold=0.0):
        super().__init__()

        self.start = start
        self.threshold = threshold

        #internal event
        self.E = ZeroCrossing(
            func_evt=lambda t: self.inputs[0] - self.threshold
            )

        #internal event for transition detection
        self.events = [self.E]


    def __len__(self):
        """This block has no direct passthrough"""
        return 0


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
        """
        
        #start + number of detected events
        self.outputs[0] = self.start + len(self.E)


class CounterUp(Counter):
    """Counts the number of detected unidirectional (lo->hi) threshold crossings.
    
    Note
    ----
    This is a modification of 'Counter' which only counts 
    unidirectional zero-crossings (low -> high)

    Parameters
    ----------
    start : int
        counter start (initial condition)
    threshold : float
        threshold for zero crossing
    
    Attributes
    ----------
    E : ZeroCrossingUp
        internal event manager
    events : list[ZeroCrossing]
        internal zero crossing event
    """

    def __init__(self, start=0, threshold=0.0):
        super().__init__(start, threshold)

        #internal event
        self.E = ZeroCrossingUp(
            func_evt=lambda t: self.inputs[0] - self.threshold
            )

        #internal event for transition detection
        self.events = [self.E]


class CounterDown(Counter):
    """Counts the number of detected unidirectional (hi->lo) threshold crossings.
    
    Note
    ----
    This is a modification of 'Counter' which only counts 
    unidirectional zero-crossings (high -> low)

    Parameters
    ----------
    start : int
        counter start (initial condition)
    threshold : float
        threshold for zero crossing
    
    Attributes
    ----------
    E : ZeroCrossingDown
        internal event manager
    events : list[ZeroCrossing]
        internal zero crossing event
    """

    def __init__(self, start=0, threshold=0.0):
        super().__init__(start, threshold)

        #internal event
        self.E = ZeroCrossingDown(
            func_evt=lambda t: self.inputs[0] - self.threshold
            )

        #internal event for transition detection
        self.events = [self.E]