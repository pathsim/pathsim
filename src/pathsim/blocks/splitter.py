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
from ..utils.mutable import mutable


# SIMO BLOCKS ===========================================================================

@mutable
class Splitter(Block):
    """Splits the input into multiple outputs, weighted with fractions (SIMO).

    .. math::

        y_i(t) = f_i \\cdot u(t)

    The fractions have to sum up to one, so the splitter conserves what flows
    through it, for example a mass or molar flow.


    Example
    -------
    Split a flow into two outputs with 90 and 10 percent:

    .. code-block:: python

        S = Splitter([0.9, 0.1])

        C = Connection(S["out_1"], B)


    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called
    multiple times per timestep, each time when `Simulation._update(t)` is
    called in the global simulation loop.


    Parameters
    ----------
    fractions : array_like
        fractions of the input for each output, must sum up to one


    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    input_port_labels = {"in": 0}
    output_port_labels = None

    def __init__(self, fractions=None):

        self.fractions = np.ones(1) if fractions is None else np.asarray(fractions, dtype=float)

        #input validation
        if not np.isclose(np.sum(self.fractions), 1.0):
            raise ValueError(f"'fractions' must sum up to one but sum up to {np.sum(self.fractions)}!")

        #one output per fraction
        self.output_port_labels = {f"out_{i}": i for i in range(len(self.fractions))}

        super().__init__()

        self.op_alg = Operator(
            func=lambda u: self.fractions * u[0],
            jac=lambda u: self.fractions.reshape(-1, 1)
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
