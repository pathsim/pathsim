#########################################################################################
##
##               LINEAR TIME INVARIANT DYNAMICAL BLOCKS (blocks/lti.py)
##
##             This module defines linear time invariant dynamical blocks
##
##                                 Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from scipy.signal import ZerosPolesGain
from scipy.signal import TransferFunction as _TransferFunction

from ._block import Block

from ..utils.register import Register
from ..utils.gilbert import gilbert_realization
from ..utils.deprecation import deprecated

from ..optim.operator import DynamicOperator


# LTI BLOCKS ============================================================================

class StateSpace(Block):
    """
    This block defines a linear time invariant (LTI) multi input multi output (MIMO) 
    state space model with the structure

    .. math::

        \\begin{eqnarray}
            \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
                   y &= \\mathbf{C} x + \\mathbf{D} u 
        \\end{eqnarray}

    where `A`, `B`, `C` and `D` are the state space matrices, `x` is the state, 
    `u` the input and `y` the output vector.
    
    Example
    -------
    A SISO state space block with two internal states can be initialized 
    like this:

    .. code-block:: python

        S = StateSpace(
            A=-np.eye(2), 
            B=np.ones((2, 1)), 
            C=np.ones((1, 2)), 
            D=1.0
            )

    and a MIMO (2 in, 2 out) state space block with three internal states 
    can be initialized like this:

    .. code-block:: python

        S = StateSpace(
            A=-np.eye(3), 
            B=np.ones((3, 2)), 
            C=np.ones((2, 3)), 
            D=np.ones((2, 2))
            )

    Parameters
    ----------
    A, B, C, D : array_like
        real valued state space matrices
    initial_value : array_like, None
        initial state / initial condition

    Attributes
    ----------
    op_dyn : DynamicOperator
        internal dynamic operator for state equation
    op_alg : DynamicOperator
        internal algebraic operator for mapping to outputs
    """

    def __init__(self, 
                 A=-1.0, B=1.0, C=-1.0, D=1.0, 
                 initial_value=None):
        super().__init__()

        #statespace matrices with input shape validation
        self.A = np.atleast_2d(A)
        self.B = np.atleast_1d(B)
        self.C = np.atleast_1d(C)
        self.D = np.atleast_1d(D)

        #get statespace dimensions
        n, _ = self.A.shape 
        if self.B.ndim == 1: n_in = 1 
        else: _, n_in = self.B.shape 
        if self.C.ndim == 1: n_out = 1 
        else: n_out, _ = self.C.shape

        #set io channels
        self.inputs = Register(n_in)
        self.outputs = Register(n_out)

        #initial condition and shape validation
        if initial_value is None:
            self.initial_value = np.zeros(n)
        else: 
            self.initial_value = np.atleast_1d(initial_value)

        #operators
        self.op_dyn = DynamicOperator(
            func=lambda x, u, t: np.dot(self.A, x) + np.dot(self.B, u),
            jac_x=lambda x, u, t: self.A,
            jac_u=lambda x, u, t: self.B
            )
        self.op_alg = DynamicOperator(
            func=lambda x, u, t: np.dot(self.C, x) + np.dot(self.D, u),
            jac_x=lambda x, u, t: self.C,
            jac_u=lambda x, u, t: self.D
            )


    def __len__(self):
        #check if direct passthrough exists
        return int(np.any(self.D)) if self._active else 0

    def solve(self, t, dt):
        """advance solution of implicit update equation of the solver

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
        """compute timestep update with integration engine
        
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


class TransferFunctionPRC(StateSpace):
    """This block defines a LTI (MIMO for pole residue) transfer function.

    The transfer function is defined in pole-residue-constant (PRC) form

    .. math::
        
        \\mathbf{H}(s) = \\mathbf{C} + \\sum_n^N \\frac{\\mathbf{R}_n}{s - p_n}

    where 'Poles' are the scalar (possibly complex conjugate) poles of the 
    transfer function and 'Residues' are the possibly matrix valued (in MIMO case) 
    and complex conjugate residues of the transfer function. 'Const' has same 
    shape as 'Residues'.

    Upon initialization, the state space realization of the transfer 
    function is computed using a minimal gilbert realization.

    The resulting state space model of the form

    .. math::
        
        \\begin{eqnarray}
            \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
                   y &= \\mathbf{C} x + \\mathbf{D} u 
        \\end{eqnarray}

    is handled the same as the 'StateSpace' block, where `A`, `B`, `C` and `D` 
    are the state space matrices, `x` is the internal state, `u` the input and 
    `y` the output vector.
        
    Parameters
    ----------
    Poles : array
        transfer function poles
    Residues : array
        transfer function residues
    Const : array, float
        constant term of transfer function
    """

    def __init__(self, Poles=[], Residues=[], Const=0.0):

        #parameters of transfer function in pole-residue-const form
        self.Const, self.Poles, self.Residues = Const, Poles, Residues

        #Statespace realization of transfer function
        A, B, C, D = gilbert_realization(Poles, Residues, Const)

        #initialize statespace model
        super().__init__(A, B, C, D)


@deprecated(version="1.0.0", replacement="TransferFunctionPRC")
class TransferFunction(TransferFunctionPRC):
    """Alias for TransferFunctionPRC."""
    pass


class TransferFunctionZPG(StateSpace):
    """This block defines a LTI (SISO) transfer function.

    The transfer function is defined in zeros-poles-gain (ZPG) form

    .. math::
        
        \\mathbf{H}(s) = k \\frac{(s - z_1)(s - z_2)\\cdots(s - z_m)}{(s - p_1)(s - p_2)\\cdots(s - p_n)}

    where `Zeros` are the scalar (possibly complex conjugate) zeros of the 
    transfer function, and `Poles` are the poles (denominator zeros) of the 
    transfer function. `Gain` is the scalar factor `k`.
    
    Upon initialization, the state space realization of the transfer function is 
    computed using `scipy.signal.ZerosPolesGain(Zeros, Poles, Gain).to_ss()`.

    The resulting state space model of the form

    .. math::
        
        \\begin{eqnarray}
            \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
                   y &= \\mathbf{C} x + \\mathbf{D} u 
        \\end{eqnarray}

    is handled the same as the 'StateSpace' block, where `A`, `B`, `C` and `D` 
    are the state space matrices, `x` is the internal state, `u` the input and 
    `y` the output vector.
        
    Parameters
    ----------
    Poles : array_like
        transfer function poles
    Zeros : array_like
        transfer function zeros
    Gain : float
        gain term of transfer function 
    """

    def __init__(self, Zeros=[], Poles=[-1], Gain=1.0):

        #parameters of transfer function in zeros-poles-gain form
        self.Zeros, self.Poles, self.Gain = Zeros, Poles, Gain

        #build scipy object -> convert to statespace
        sp_SS = ZerosPolesGain(Zeros, Poles, Gain).to_ss()

        #initialize statespace model
        super().__init__(sp_SS.A, sp_SS.B, sp_SS.C, sp_SS.D)


class TransferFunctionNumDen(StateSpace):
    """This block defines a LTI (SISO) transfer function.

    The transfer function is defined in polynomial (numerator-denominator) form

    .. math::
        
        \\mathbf{H}(s) = \\frac{b_n + b_{n-1} s + \\dots + b_{0} s^n}{a_m + a_{m-1} s + \\dots + a_{0} s^m}

    where `Num` is the list of numerator polynomial coefficients and `Den` the 
    list of denominator coefficients.
    
    Upon initialization, the state space realization of the transfer function is 
    computed using `scipy.signal.TransferFunction(Num, Den).to_ss()`.

    The resulting state space model of the form

    .. math::
        
        \\begin{eqnarray}
            \\dot{x} &= \\mathbf{A} x + \\mathbf{B} u \\\\
                   y &= \\mathbf{C} x + \\mathbf{D} u 
        \\end{eqnarray}

    is handled the same as the 'StateSpace' block, where `A`, `B`, `C` and `D` 
    are the state space matrices, `x` is the internal state, `u` the input and 
    `y` the output vector.
        
    Parameters
    ----------
    Num : array_like
        numerator polynomial coefficients
    Den : array_like
        denominator polynomial coefficients
    """

    def __init__(self, Num=[1], Den=[1, 1]):

        #parameters of transfer function in numerator-denominator
        self.Num, self.Den = Num, Den

        #build scipy object -> convert to statespace
        sp_SS = _TransferFunction(Num, Den).to_ss()

        #initialize statespace model
        super().__init__(sp_SS.A, sp_SS.B, sp_SS.C, sp_SS.D)

