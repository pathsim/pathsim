#########################################################################################
##
##                                      ADDER BLOCK 
##                                   (blocks/adder.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..optim.operator import Operator


# MISO BLOCKS ===========================================================================

class Adder(Block):
    """Summs / adds up all input signals to a single output signal (MISO)
    
    This is how it works in the default case

    .. math::
        
        y(t) = \\sum_i u_i(t)

    and like this when additional operations are defined

    .. math::
        
        y(t) = \\sum_i \\mathrm{op}_i \\cdot u_i(t)

    
    Example
    -------
    This is the default initialization that just adds up all the inputs:

    .. code-block:: python

        A = Adder()

    and this is the initialization with specific operations that subtracts 
    the second from first input and neglects all others:

    .. code-block:: python
    
        A = Adder('+-')

    
    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called 
    multiple times per timestep, each time when `Simulation._update(t)` is 
    called in the global simulation loop.

    
    Parameters
    ----------
    operations : str, optional
        optional string of operations to be applied before 
        summation, i.e. '+-' will compute the difference, 
        'None' will just perform regular sum
    

    Attributes
    ----------
    _ops : dict
        dict that maps string operations to numerical
    _ops_array : array_like
        operations converted to array
    op_alg : Operator
        internal algebraic operator
    """

    input_port_labels = None
    output_port_labels = {"out":0}

    def __init__(self, operations=None):
        super().__init__()

        #allowed arithmetic operations
        self._ops = {"+":1.0, "-":-1.0, "0":0.0}
        self.operations = operations    
    
        #are special operations defined?
        if self.operations is None:

            #create internal algebraic operator
            self.op_alg = Operator(
                func=sum, 
                jac=lambda x: np.ones_like(x)
                )

        else:

            #input validation
            if not isinstance(self.operations, str):
                raise ValueError("'operations' must be string or 'None'")
            for op in self.operations:
                if op not in self._ops:
                    raise ValueError(f"operation '{op}' not in {self._ops}")

            #construct array from operations
            self._ops_array = np.array([self._ops[op] for op in self.operations])

            def sum_ops(X):
                return sum(x*op for x, op in zip(X, self._ops_array))
            def jac_ops(X):
                nx, no = len(X), len(self._ops_array)
                if nx < no: return self._ops_array[:nx]
                return np.pad(self._ops_array, nx-no, mode="constant")

            #create internal algebraic operator
            self.op_alg = Operator(
                func=sum_ops, 
                jac=jac_ops
                )


    def __len__(self):
        """Purely algebraic block"""
        return 1


    def update(self, t):
        """update system equation in fixed point loop for 
        algebraic loops, with optional error control

        Parameters
        ----------
        t : float
            evaluation time
        """
        u = self.inputs.to_array()
        y = self.op_alg(u)
        self.outputs.update_from_array(y)