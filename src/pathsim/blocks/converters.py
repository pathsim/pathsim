#########################################################################################
##
##                            IDEAL AD AND DA CONVERTERS
##                              (blocks/converters.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..events.schedule import Schedule
from ..utils.mutable import mutable


# MIXED SIGNAL BLOCKS ===================================================================

@mutable
class ADC(Block):
    """Models an ideal Analog-to-Digital Converter (ADC).

    This block samples an analog input signal periodically, quantizes it
    according to the specified number of bits and input span, and outputs
    the resulting digital code on multiple output ports. The sampling
    is triggered by a scheduled event.

    Functionality:

    1. Samples the analog input `inputs[0]` at intervals of `T`, starting after delay `tau`.
    2. Clips the input voltage to the defined `span` [min_voltage, max_voltage].
    3. Scales the clipped voltage to the range [0, 1].
    4. Quantizes the scaled value to an integer code between 0 and 2^n_bits - 1 using flooring.
    5. Converts the integer code to an n_bits binary representation.
    6. Outputs the binary code on ports 0 (LSB) to n_bits-1 (MSB).

    Ideal characteristics:

    - Instantaneous sampling at scheduled times.
    - Perfect, noise-free quantization.
    - No aperture jitter or other dynamic errors.


    Parameters
    ----------
    n_bits : int, optional
        Number of bits for the digital output code. Default is 4.
    span : list[float] or tuple[float], optional
        The valid analog input value range [min_voltage, max_voltage].
        Inputs outside this range will be clipped. Default is [-1, 1].
    T : float, optional
        Sampling period (time between samples). Default is 1 time unit.
    tau : float, optional
        Initial delay before the first sample is taken. Default is 0.

    
    Attributes
    ----------
    events : list[Schedule]
        Internal scheduled event responsible for periodic sampling and conversion.
    """

    input_port_labels = {"in": 0}
    output_port_labels = None

    def __init__(self, n_bits=4, span=[-1, 1], T=1, tau=0):
        super().__init__()

        #block params
        self.n_bits = n_bits
        self.span = span
        self.T = T
        self.tau = tau

        def _sample(t):

            #clip and scale to ADC span
            lower, upper = self.span
            analog_in = self.inputs[0]

            clipped_val = np.clip(analog_in, lower, upper)
            scaled_val = (clipped_val - lower) / (upper - lower)
            int_val = np.floor(scaled_val * (2**self.n_bits))
            int_val = min(int_val, 2**self.n_bits - 1)

            #convert to bits
            bits = format(int(int_val), "b").zfill(self.n_bits)

            #set bits to block outputs LSB -> MSB
            for i, b in enumerate(bits):
                self.outputs[self.n_bits-1-i] = int(b)

        #internal scheduled events
        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_sample
                ),
            ]


    def __len__(self):
        """This block has no direct passthrough"""
        return 0


@mutable
class DAC(Block):
    """Models an ideal Digital-to-Analog Converter (DAC).

    This block reads a digital input code periodically from its input ports,
    reconstructs the corresponding analog value based on the number of bits
    and output span, and holds the output constant between updates. The update
    is triggered by a scheduled event.

    Functionality:

    1. Reads the digital code from input ports 0 (LSB) to n_bits-1 (MSB) at intervals of `T`, starting after delay `tau`.
    2. Interprets the inputs as an unsigned binary integer code.
    3. Converts the integer code to a fractional value between 0 and (2^n_bits - 1) / 2^n_bits.
    4. Scales this fractional value to the specified analog output `span`.
    5. Outputs the resulting analog value on `outputs[0]`.
    6. Holds the output value constant until the next scheduled update.

    Ideal characteristics:

    - Instantaneous update at scheduled times.
    - Perfect, noise-free reconstruction.
    - No glitches or settling time.


    Parameters
    ----------
    n_bits : int, optional
        Number of digital input bits expected. Default is 4.
    span : list[float] or tuple[float], optional
        The analog output value range [min_voltage, max_voltage] corresponding
        to the digital codes 0 and 2^n_bits - 1, respectively (approximately).
        Default is [-1, 1].
    T : float, optional
        Update period (time between output updates). Default is 1 time unit.
    tau : float, optional
        Initial delay before the first output update. Default is 0.


    Attributes
    ----------
    events : list[Schedule]
        Internal scheduled event responsible for periodic updates.
    """

    input_port_labels = None
    output_port_labels = {"out": 0}

    def __init__(self, n_bits=4, span=[-1, 1], T=1, tau=0):
        super().__init__()

        self.n_bits = n_bits
        self.span = span
        self.T = T
        self.tau = tau

        def _sample(t):
            
            #convert bits to integer LSB -> MSB
            val = sum(self.inputs[i] * (2**i) for i in range(self.n_bits))

            #scale to DAC span and set output
            lower, upper = self.span
            levels = 2**self.n_bits

            scaled_val =  val / (levels - 1) if levels > 1 else 0.0
            self.outputs[0] = lower + (upper - lower) * scaled_val

        #internal scheduled events
        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_sample
                ),
            ]


    def __len__(self):
        """This block has no direct passthrough"""
        return 0