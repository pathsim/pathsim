#########################################################################################
##
##                               DIFFERENTIATOR BLOCK 
##                            (blocks/differentiator.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..optim.operator import DynamicOperator


# BLOCKS ================================================================================

class Differentiator(Block):
    """Differentiates the input signal. 

    Uses a first order transfer function with a pole at the origin which implements 
    a high pass filter. Supports vector input. 
        
    .. math::
        
        H_\\mathrm{diff}(s) = \\frac{s}{1 + s / f_\\mathrm{max}} 

    The approximation holds for signals up to a frequency of approximately f_max.

    Note
    -----
    Depending on `f_max`, the resulting system might become stiff or ill conditioned!
    As a practical choice set `f_max` to 3x the highest expected signal frequency.

    Note
    ----
    Since this is an approximation of real differentiation, the approximation will not hold 
    if there are high frequency components present in the signal. For example if you have 
    discontinuities such as steps or squere waves.

    Example
    -------
    The block is initialized like this:

    .. code-block:: python
        
        #cutoff at 1kHz
        D = Differentiator(f_max=1e3)

    Parameters
    ----------
    f_max : float
        highest expected signal frequency

    Attributes
    ----------
    op_dyn : DynamicOperator
        internal dynamic operator for ODE component
    op_alg : DynamicOperator
        internal algebraic operator

    """

    def __init__(self, f_max=1e2):
        super().__init__()

        #maximum frequency for differentiator approximation
        self.f_max = f_max

        #initial state for integration engine
        self.initial_value = 0.0

        self.op_dyn = DynamicOperator(
            func=lambda x, u, t: self.f_max * (u - x),
            jac_x=lambda x, u, t: -self.f_max*np.eye(len(u))
            )
        self.op_alg = DynamicOperator(
            func=lambda x, u, t: self.f_max * (u - x),
            jac_x=lambda x, u, t: -self.f_max*np.eye(len(u)),
            jac_u=lambda x, u, t: self.f_max*np.eye(len(u)),
            )

    def __len__(self):
        return 1 if self._active else 0


    def update(self, t):
        """update system equation fixed point loop,
        with convergence control
    
        Parameters
        ----------
        t : float
            evaluation time
        """
        x, u = self.engine.state, self.inputs.to_array()
        y = self.op_alg(x, u, t)
        self.outputs.update_from_array(y)


    def solve(self, t, dt):
        """advance solution of implicit update equation

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        ------- 
        error : float
            solver residual norm
        """
        x, u = self.engine.state, self.inputs.to_array()
        f, J = self.op_dyn(x, u, t), self.op_dyn.jac_x(x, u, t)
        return self.engine.solve(f, J, dt)


    def step(self, t, dt):
        """compute update step with integration engine
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """
        x, u = self.engine.state, self.inputs.to_array()
        f = self.op_dyn(x, u, t)
        return self.engine.step(f, dt)