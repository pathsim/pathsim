#########################################################################################
##
##                            RANDOM NUMBER GENERATOR BLOCK 
##                               (pathsim/blocks/rng.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..utils.deprecation import deprecated
from ..events.schedule import Schedule 


# BLOCKS ================================================================================

class RandomNumberGenerator(Block):
    """Generates a random output value using `numpy.random.rand`.

    If no `sampling_period` (None) is specified, every simulation timestep gets
    a random value. Otherwise an internal `Schedule` event is used to periodically
    sample a random value and set the output like a zero-order-hold stage.

    Parameters
    ----------
    sampling_period : float, None
        time between random samples

    Attributes
    ----------
    _sample : float
        internal random number state in case that
        no `sampling_period` is provided
    Evt : Schedule
        internal event that periodically samples a random
        value in case `sampling_period` is provided
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(self, sampling_period=None):
        super().__init__()

        #block parameter
        self.sampling_period = sampling_period 

        #sampling produces discrete time behavior
        if sampling_period is None:

            #initial sample for non-discrete block
            self._sample = np.random.rand()

        else:
            
            #internal scheduled list event
            def _set(t):
                self.outputs[0] = np.random.rand()

            self.Evt = Schedule(
                t_start=0,
                t_period=sampling_period,
                func_act=_set
                )
            self.events = [self.Evt]


    def update(self, t):
        """Setting output with random sample in case
        of `sampling_period==None`, otherwise does nothing.

        Parameters
        ----------
        t : float
            evaluation time
        """
        if self.sampling_period is None:
            self.outputs[0] = self._sample


    def sample(self, t, dt):
        """Generating a new random sample at each timestep
        in case of `sampling_period==None`, otherwise does nothing.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
        """
        if self.sampling_period is None:
            self._sample = np.random.rand()


    def to_checkpoint(self, prefix, recordings=False):
        """Serialize RNG state including current sample."""
        json_data, npz_data = super().to_checkpoint(prefix, recordings=recordings)
        if self.sampling_period is None:
            json_data["_sample"] = float(self._sample)
        return json_data, npz_data


    def load_checkpoint(self, prefix, json_data, npz):
        """Restore RNG state including current sample."""
        super().load_checkpoint(prefix, json_data, npz)
        if self.sampling_period is None:
            self._sample = json_data.get("_sample", 0.0)


    def __len__(self):
        """Essentially a source-like block without passthrough"""
        return 0


@deprecated(version="1.0.0", replacement="RandomNumberGenerator")
class RNG(RandomNumberGenerator):
    """Alias for RandomNumberGenerator."""
    pass