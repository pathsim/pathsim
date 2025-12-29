#########################################################################################
##
##                                  RELAY BLOCK
##                               (blocks/relay.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..events.zerocrossing import ZeroCrossingUp, ZeroCrossingDown


# MIXED SIGNAL BLOCKS ===================================================================

class Relay(Block):
    """Relay block with hysteresis (Schmitt trigger).
    
    Switches output between two values based on input crossing upper and lower 
    thresholds. The hysteresis prevents rapid switching when input is noisy.
    
    When input rises above `threshold_up`, output switches to `value_up`.
    When input falls below `threshold_down`, output switches to `value_down`.

    Examples
    --------
    Basic thermostat that turns heater on below 19°C, off above 21°C:
    
    .. code-block:: python
    
        from pathsim.blocks import Relay
        
        thermostat = Relay(
            threshold_up=21.0, 
            threshold_down=19.0,
            value_up=0.0, 
            value_down=1.0
            )

    Parameters
    ----------
    threshold_up : float
        threshold for transitioning to upper relay state `value_up` (default: 1.0)
    threshold_down : float
        threshold for transitioning to lower relay state `value_down` (default: 0.0)
    value_up : float
        value for upper relay state (default: 1.0)
    value_down : float
        value for lower relay state (default: 0.0)

    Attributes
    ----------
    events : list[ZeroCrossingUp, ZeroCrossingDown]
        internal zero crossing events for relay state transitions
    """

    def __init__(
        self, 
        threshold_up=1.0, 
        threshold_down=0.0, 
        value_up=1.0, 
        value_down=0.0
        ):
        super().__init__()

        # block params
        self.threshold_up = threshold_up 
        self.threshold_down = threshold_down 
        self.value_up = value_up 
        self.value_down = value_down 

        # block io with port labels
        self.inputs = Register(mapping={"in": 0})
        self.outputs = Register(mapping={"out": 0})

        # internal event function factories
        def _check(val):
            return lambda t: self.inputs[0] - val
        def _set(val):
            def __set(t):
                self.outputs[0] = val
            return __set 

        # internal events for transition detection
        self.events = [
            ZeroCrossingUp(
                func_evt=_check(self.threshold_up),
                func_act=_set(self.value_up)
                ),
            ZeroCrossingDown(
                func_evt=_check(self.threshold_down),
                func_act=_set(self.value_down)
                ),
            ]


    def __len__(self):
        """This block has no direct passthrough"""
        return 0


    def update(self, t):
        """update system equation for fixed point loop, 
        here just setting the outputs
    
        Note
        ----
        No pasthrough, setting block outputs is done 
        by the internal events.
        
        Parameters
        ----------
        t : float
            evaluation time
        """
        pass
