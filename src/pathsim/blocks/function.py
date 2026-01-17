#########################################################################################
##
##                                  MIMO FUNCTION BLOCK 
##                              (pathsim/blocks/function.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..optim.operator import Operator, DynamicOperator


# MIMO BLOCKS ===========================================================================

class Function(Block):
    """Arbitrary MIMO function block, defined by a function or `lambda` expression.

    The function can have multiple arguments that are then provided 
    by the input channels of the function block.

    Form multi input, the function has to specify multiple arguments
    and for multi output, the aoutputs have to be provided as a 
    tuple or list. 

    In the context of the global system, this block implements algebraic 
    components of the global system ODE/DAE.

    .. math::

        \\vec{y} = \\mathrm{func}(\\vec{u})
    

    Note
    ----
    This block is purely algebraic and its operation (`op_alg`) will be called 
    multiple times per timestep, each time when `Simulation._update(t)` is 
    called in the global simulation loop.
    Therefore `func` must be purely algebraic and not introduce states, 
    delay, etc. For interfacing with external stateful APIs, use the 
    `Wrapper` block.


    Note
    -----
    If the outputs are provided as a single numpy array, they are 
    considered a single output. For MIMO, output has to be tuple.

    
    Example
    -------
    consider the function: 

    .. code-block:: python
    
        from pathsim.blocks import Function

        def f(a, b, c):
            return a**2, a*b, b/c

        fn = Function(f)
        

    then, when the block is uldated, the input channels of the block are 
    assigned to the function arguments following this scheme:

    .. code-block::

        inputs[0] -> a
        inputs[1] -> b
        inputs[2] -> c

    and the function outputs are assigned to the 
    output channels of the block in the same way:

    .. code-block::

        a**2 -> outputs[0]
        a*b  -> outputs[1]
        b/c  -> outputs[2]

    Because the `Function` block only has a single argument, it can be 
    used to decorate a function and make it a `PathSim` block. This might 
    be handy in some cases to keep definitions concise and localized 
    in the code:

    .. code-block:: python

        from pathsim.blocks import Function

        #does the same as the definition above
            
        @Function
        def fn(a, b, c):
            return a**2, a*b, b/c

        #'fn' is now a PathSim block


    Parameters
    ---------- 
    func : callable
        MIMO function that defines algebraic block IO behaviour, signature `func(*tuple)`


    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator that wraps `func`
    
    """

    def __init__(self, func=lambda x: x):
        super().__init__()

        #some checks to ensure that function works correctly
        if not callable(func):  
            raise ValueError(f"'{func}' is not callable")
        
        #function defining the block update
        self.func = func
        self.op_alg = Operator(func=lambda x: func(*x))


    def update(self, t):
        """Evaluate function block as part of algebraic component 
        of global system DAE. 

        Parameters
        ----------
        t : float
            evaluation time
        """
                
        #apply operator to get output
        y = self.op_alg(self.inputs.to_array())
        self.outputs.update_from_array(y)



class DynamicalFunction(Block):
    """Arbitrary MIMO time and input dependent function block.

    The function signature needs two arguments `f(u, t)` where `u` is 
    the (possibly vectorial) block input and `t` is a time dependency.

    .. math::

        \\vec{y} = \\mathrm{func}(\\vec{u}, t)

    
    Note
    ----
    This block does essentially the same as `Function` but with different 
    requirements for the signature of the function to be wrapped. 
    Block inputs are packed into an array `u` and this block additionally 
    accepts time dependency in the function provided. 
    Thats where the prefix `Dynamical..` comes from.


    Example
    -------
    Lets say we want to implement a super simple model for a voltage controlled 
    oscillator (VCO), where the block input controls the frequency of a sine wave 
    at the output. 

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import DynamicalFunction
        
        f_0 = 100

        def f_vco(u, t):
            return np.sin(2*np.pi*f_0*u*t)

        vco = DynamicalFunction(f_vco)        
    
    
    Using it as a decorator also works:

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import DynamicalFunction
        
        f_0 = 100
        
        @DynamicalFunction
        def vco(u, t):
            return np.sin(2*np.pi*f_0*u*t)

        #'vco' is now a PathSim block 


    Parameters
    ----------
    func : callable
        function that defines algebraic block IO behaviour with time dependency, 
        signature `func(u, t)` where `u` is `numpy.ndarray` and `t` is `float`


    Attributes
    ----------
    op_alg : DynamicOperator
        internal operator that wraps `func`

    """
    
    def __init__(self, func=lambda u, t: u):
        super().__init__()

        #some checks to ensure that function works correctly
        if not callable(func):  
            raise ValueError(f"'{func}' is not callable")
        
        #function defining the block update
        self.func = func
        self.op_alg = DynamicOperator(lambda x, u, t: func(u, t))


    def update(self, t):
        """Evaluate function with time dependency as part of algebraic 
        component of global system DAE. 

        Parameters
        ----------
        t : float
            evaluation time
        """
                
        #apply operator to get output
        y = self.op_alg(None, self.inputs.to_array(), t)
        self.outputs.update_from_array(y)