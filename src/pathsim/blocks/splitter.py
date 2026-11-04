#########################################################################################
##
##                                  SPLITTER BLOCK
##                               (blocks/splitter.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..optim.operator import Operator
from ..utils.carrier import Carrier, Vector
from ..utils.mutable import mutable


# SIMO BLOCKS ===========================================================================

@mutable
class Splitter(Block):
    """Splits the input into multiple outputs, weighted with fractions (SIMO).

    .. math::

        y_i(t) = f_i \\cdot u(t)

    The fractions have to sum up to one, so the splitter conserves what flows
    through it, for example a mass or molar flow.

    With a carrier, the input is a stream of multiple channels, for example
    the flows of multiple species. Every output is a stream with the same
    layout and each channel is split with the same fractions.


    Example
    -------
    Split a flow into two outputs with 90 and 10 percent:

    .. code-block:: python

        S = Splitter([0.9, 0.1])

    Split a stream of species flows, the ports 'in', 'out_0' and 'out_1'
    have the layout of the carrier:

    .. code-block:: python

        class Species(Carrier):
            keys = ("H", "D", "T")

        S = Splitter([0.9, 0.1], carrier=Species())

        C = Connection(S["out_1"], B["in"])


    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called
    multiple times per timestep, each time when `Simulation._update(t)` is
    called in the global simulation loop.


    Parameters
    ----------
    fractions : array_like
        fractions of the input for each output, must sum up to one
    carrier : Carrier | None
        layout of the input and output streams, single channel if 'None'


    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    input_port_labels = {"in": 0}
    output_port_labels = None

    def __init__(self, fractions=None, carrier=None):

        self.fractions = np.ones(1) if fractions is None else np.asarray(fractions, dtype=float)
        self.carrier = carrier

        #input validation
        if not np.isclose(np.sum(self.fractions), 1.0):
            raise ValueError(f"'fractions' must sum up to one but sum up to {np.sum(self.fractions)}!")

        #ports with the layout of the carrier, placed one after another
        n = 1 if carrier is None else len(carrier)

        def _port(start):
            if carrier is None:
                return start
            if carrier.keys:
                return Carrier(start, keys=carrier.keys)
            return Vector(n, start)

        self.input_port_labels = {"in": _port(0)}
        self.output_port_labels = {
            f"out_{i}": _port(i * n) for i in range(len(self.fractions))
            }

        super().__init__()

        #every channel of the input split with the same fractions
        self.op_alg = Operator(
            func=lambda u: np.kron(self.fractions, u),
            jac=lambda u: np.kron(self.fractions.reshape(-1, 1), np.eye(len(u)))
            )


    def update(self, t):
        """update system equation

        Parameters
        ----------
        t : float
            evaluation time
        """
        u = self.inputs.to_array()
        y = self.op_alg(u)
        self.outputs.update_from_array(y)
