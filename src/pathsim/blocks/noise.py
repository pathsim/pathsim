#########################################################################################
##
##                             TIME DOMAIN NOISE SOURCES
##                                  (blocks/noise.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..events.schedule import Schedule


# NOISE SOURCE BLOCKS ===================================================================

class WhiteNoise(Block):
    """White noise source with Gaussian distribution.

    Generates uncorrelated random samples with either constant amplitude
    (``standard_deviation`` mode) or timestep-scaled amplitude for stochastic
    integration (``spectral_density`` mode).

    In spectral density mode, output is scaled as √(S₀/dt) so that integrating
    the noise yields correct statistical properties (Wiener process).


    Note
    ----
    If ``spectral_density`` is provided, it takes precedence over ``standard_deviation``.
    If ``sampling_period`` is set, noise is sampled at fixed intervals (zero-order hold).


    Parameters
    ----------
    standard_deviation : float
        output standard deviation for constant-amplitude mode (default: 1.0)
    spectral_density : float, optional
        power spectral density S₀ in [signal²/Hz]
    sampling_period : float, optional
        time between samples, if None samples every timestep
    seed : int, optional
        random seed for reproducibility
    """

    input_port_labels = {}
    output_port_labels = {"out": 0}

    def __init__(self, standard_deviation=1.0, spectral_density=None,
                 sampling_period=None, seed=None):
        super().__init__()

        #block parameters
        self.standard_deviation = standard_deviation
        self.spectral_density = spectral_density
        self.sampling_period = sampling_period

        #random number generator (with optional seed for reproducibility)
        self._rng = np.random.default_rng(seed)

        #current noise sample
        self._current_sample = 0.0

        #sampling produces discrete time behavior
        if sampling_period is not None:

            #generate initial sample for discrete mode
            self._current_sample = self._generate_sample(sampling_period)
            self.outputs[0] = self._current_sample

            #internal scheduled event
            def _set(t):
                self._current_sample = self._generate_sample(self.sampling_period)
                self.outputs[0] = self._current_sample

            self.events = [
                Schedule(
                    t_start=0,
                    t_period=sampling_period,
                    func_act=_set
                )
            ]


    def __len__(self):
        return 0


    def _generate_sample(self, dt):
        """Generate a random sample from the noise distribution.

        Parameters
        ----------
        dt : float
            integration timestep (used for spectral density scaling)
        """
        if self.spectral_density is not None:
            #spectral density mode: scale for correct integration
            return self._rng.normal(0, 1) * np.sqrt(self.spectral_density / dt)
        else:
            #constant amplitude mode
            return self._rng.normal(0, self.standard_deviation)


    def sample(self, t, dt):
        """Generate new noise sample after successful timestep.

        Only generates new samples in continuous mode (sampling_period=None).

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
        """
        if self.sampling_period is None:
            self._current_sample = self._generate_sample(dt)
            self.outputs[0] = self._current_sample


    def update(self, t):
        pass


class PinkNoise(Block):
    """Pink noise (1/f noise) source using the Voss-McCartney algorithm.

    Generates noise with power spectral density proportional to 1/f, where
    lower frequencies have more power than higher frequencies.

    The algorithm maintains ``num_octaves`` independent random values representing
    different frequency bands. At each sample, one octave is updated based on the
    binary representation of the sample counter, creating the characteristic 1/f
    spectrum through the superposition of different update rates.


    Note
    ----
    If ``spectral_density`` is provided, it takes precedence over ``standard_deviation``.
    If ``sampling_period`` is set, noise is sampled at fixed intervals (zero-order hold).


    Parameters
    ----------
    standard_deviation : float
        approximate output standard deviation (default: 1.0)
    spectral_density : float, optional
        power spectral density, output scaled as √(S₀/(N·dt))
    num_octaves : int
        number of frequency bands in algorithm (default: 16)
    sampling_period : float, optional
        time between samples, if None samples every timestep
    seed : int, optional
        random seed for reproducibility
    """

    input_port_labels = {}
    output_port_labels = {"out": 0}

    def __init__(self, standard_deviation=1.0, spectral_density=None,
                 num_octaves=16, sampling_period=None, seed=None):
        super().__init__()

        #block parameters
        self.standard_deviation = standard_deviation
        self.spectral_density = spectral_density
        self.num_octaves = num_octaves
        self.sampling_period = sampling_period

        #random number generator (with optional seed)
        self._rng = np.random.default_rng(seed)

        #algorithm state
        self.n_samples = 0
        self.octave_values = self._rng.normal(0, 1, self.num_octaves)

        #current noise sample
        self._current_sample = 0.0

        #sampling produces discrete time behavior
        if sampling_period is not None:

            #generate initial sample for discrete mode
            self._current_sample = self._generate_sample(sampling_period)
            self.outputs[0] = self._current_sample

            #internal scheduled event
            def _set(t):
                self._current_sample = self._generate_sample(self.sampling_period)
                self.outputs[0] = self._current_sample

            self.events = [
                Schedule(
                    t_start=0,
                    t_period=sampling_period,
                    func_act=_set
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        """Reset the noise generator state.

        Resets the sample counter and reinitializes all octave values.
        """
        super().reset()
        self.n_samples = 0
        self.octave_values = self._rng.normal(0, 1, self.num_octaves)
        self._current_sample = 0.0


    def _generate_sample(self, dt):
        """Generate a pink noise sample using the Voss-McCartney algorithm.

        Parameters
        ----------
        dt : float
            integration timestep (used for spectral density scaling)
        """
        #increment sample counter
        self.n_samples += 1

        #find position of least significant 1-bit (trailing zeros)
        #this determines which octave to update
        n = self.n_samples
        octave_idx = 0
        while (n & 1) == 0 and octave_idx < self.num_octaves - 1:
            n >>= 1
            octave_idx += 1

        #update the selected octave
        self.octave_values[octave_idx] = self._rng.normal(0, 1)

        #sum all octaves for pink noise output
        pink_sample = np.sum(self.octave_values)

        #scale output based on parameterization mode
        if self.spectral_density is not None:
            #spectral density mode
            return pink_sample * np.sqrt(self.spectral_density / self.num_octaves / dt)
        else:
            #constant amplitude mode
            #normalize by sqrt(num_octaves) since Var(sum) ≈ num_octaves
            return pink_sample * self.standard_deviation / np.sqrt(self.num_octaves)


    def sample(self, t, dt):
        """Generate new noise sample after successful timestep.

        Only generates new samples in continuous mode (sampling_period=None).

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
        """
        if self.sampling_period is None:
            self._current_sample = self._generate_sample(dt)
            self.outputs[0] = self._current_sample


    def update(self, t):
        pass