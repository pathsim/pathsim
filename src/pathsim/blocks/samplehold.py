#########################################################################################
##
##                              SAMPLE AND HOLD BLOCK
##                             (blocks/samplehold.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from ._block import Block
from ..events.schedule import Schedule
from ..utils.mutable import mutable


# MIXED SIGNAL BLOCKS ===================================================================

@mutable
class SampleHold(Block):
    """Samples the inputs periodically and produces them at the output.
    
    Parameters
    ----------
    T : float
        sampling period
    tau : float
        delay 
        
    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic sampling
    """

    def __init__(self, T=1, tau=0):
        super().__init__()

        self.T   = T
        self.tau = tau

        def _sample(t):
            self.outputs.update_from_array(
                self.inputs.to_array()
                )

        #internal scheduled events
        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_sample
                ),
            ]

    def __len__(self):
        return 0