#########################################################################################
##
##                                 CONTROL BLOCKS
##                                (blocks/ctrl.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.register import Register
from ..optim.operator import DynamicOperator


# SISO BLOCKS ===========================================================================

class PID(Block):
    """Proportional-Integral-Differntiation (PID) controller.

    The transfer function is defined as

    .. math::
        
        H_\\mathrm{diff}(s) = K_p + K_i \\frac{1}{s} + K_d \\frac{s}{1 + s / f_\\mathrm{max}} 

    where the differentiation is approximated by a high pass filter that holds 
    for signals up to a frequency of approximately `f_max`.


    Note
    ----
    Depending on `f_max`, the resulting system might become stiff or ill conditioned!
    As a practical choice set `f_max` to 3x the highest expected signal frequency.
    Since this block uses an approximation of real differentiation, the approximation will 
    not hold if there are high frequency components present in the signal. For example if 
    you have discontinuities such as steps or square waves.


    Note
    ----
    This block supports vector input, meaning we can have multiple parallel 
    PID paths through this block.


    Example
    -------
    The block is initialized like this:

    .. code-block:: python
        
        #cutoff at 1kHz
        pid = PID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3)


    Parameters
    ----------
    Kp : float
        poroportional controller coefficient
    Ki : float
        integral controller coefficient
    Kd : float
        differentiator controller coefficient
    f_max : float
        highest expected signal frequency


    Attributes
    ----------
    op_dyn : DynamicOperator
        internal dynamic operator for ODE component
    op_alg : DynamicOperator
        internal algebraic operator

    """
    
    def __init__(self, Kp=0, Ki=0, Kd=0, f_max=100):
        super().__init__()

        #block io with port labels
        self.inputs = Register(mapping={"in": 0})
        self.outputs = Register(mapping={"out": 0})

        #pid controller coefficients
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        #maximum frequency for differentiator approximation
        self.f_max = f_max

        #initial state for integration engine (differentiator + integrator states)
        self.initial_value = np.zeros(2)

        def _g_pid(x, u, t):
            x1, x2 = x
            yp = self.Kp * u
            yi = self.Ki * x2
            yd = self.Kd * self.f_max * (u - x1)
            return yp + yi + yd

        def _jac_x_g_pid(x, u, t):
            return np.array([-self.Kd * self.f_max, self.Ki])

        def _jac_u_g_pid(x, u, t):
            return self.Kd * self.f_max + self.Kp

        def _f_pid(x, u, t):
            x1, x2 = x
            dx1, dx2 = self.f_max * (u - x1), u
            return np.array([dx1, dx2])

        #internal operators
        self.op_dyn = DynamicOperator(
            func=_f_pid,
            )
        self.op_alg = DynamicOperator(
            func=_g_pid,
            jac_x=_jac_x_g_pid,
            jac_u=_jac_u_g_pid,
            )


    def __len__(self):
        return 1 if self._active and (self.Kp or self.Kd) else 0


    def update(self, t):
        """update system equation fixed point loop, with convergence control
    
        Parameters
        ----------
        t : float
            evaluation time
        """
        x, u = self.engine.state, self.inputs[0]
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
        x, u = self.engine.state, self.inputs[0]
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
        x, u = self.engine.state, self.inputs[0]
        f = self.op_dyn(x, u, t)
        return self.engine.step(f, dt)


class AntiWindupPID(PID):
    """Proportional-Integral-Differntiation (PID) controller with tracking 
    anti-windup mechanism (back-calculation).
    
    Anti-windup mechanisms are needed when the magnitude of the control signal 
    from the PID controller is limited by some real world saturation. In these cases, 
    the integrator will continue to acumulate the control error and "wind itself up". 
    Once the setpoint is reached, this can result in significant overshoots. This 
    implementation adds a conditional feedback term to the internal integrator that 
    "unwinds" it when the PID output crosses some limits. This is pretty much a 
    deadzone feedback element for the integrator.
            
    Mathematically, this block implements the following set of ODEs 

    .. math::
    
        \\begin{eqnarray}    
        \\dot{x}_1 =& f_\\mathrm{max} (u - x_1) \\\\
        \\dot{x}_2 =& u - w \\\\
        \\end{eqnarray}
    
    with the anti-windup feedback (depending on the pid output)

    .. math::
    
        w = K_s (y - \\min(\\max(y, y_\\mathrm{min}), y_\\mathrm{max}))

    and the output itself

    .. math::

        y = K_p u - K_d f_\\mathrm{max} x_1 + K_i x_2
    

    Note
    ----
    Depending on `f_max`, the resulting system might become stiff or ill conditioned!
    As a practical choice set `f_max` to 3x the highest expected signal frequency.
    Since this block uses an approximation of real differentiation, the approximation will 
    not hold if there are high frequency components present in the signal. For example if 
    you have discontinuities such as steps or squere waves.

    
    Example
    -------
    The block is initialized like this:

    .. code-block:: python
        
        #cutoff at 1kHz, windup limits at [-5, 5]
        pid = AntiWindupPID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3, limits=[-5, 5])


    Parameters
    ----------
    Kp : float
        poroportional controller coefficient
    Ki : float
        integral controller coefficient
    Kd : float
        differentiator controller coefficient
    f_max : float
        highest expected signal frequency
    Ks : float
        feedback term for back calculation for anti-windup control of integrator
    limits : array_like[float]
        lower and upper limit for PID output that triggers anti-windup of integrator


    Attributes
    ----------
    op_dyn : DynamicOperator
        internal dynamic operator for ODE component
    op_alg : DynamicOperator
        internal algebraic operator

    """

    def __init__(self, Kp=0, Ki=0, Kd=0, f_max=100, Ks=10, limits=[-10, 10]):
        super().__init__(Kp, Ki, Kd, f_max)

        #anti-windup control
        self.Ks = Ks
        self.limits = limits

        def _g_pid(x, u, t):
            x1, x2 = x
            yp = self.Kp * u
            yi = self.Ki * x2
            yd = self.Kd * self.f_max * (u - x1)
            return yp + yi + yd

        def _jac_x_g_pid(x, u, t):
            return np.array([-self.Kd * self.f_max, self.Ki])

        def _jac_u_g_pid(x, u, t):
            return self.Kd * self.f_max + self.Kp

        def _f_pid(x, u, t):
            x1, x2 = x

            #differentiator state
            dx1 = self.f_max * (u - x1) 
            
            #integrator state with windup control
            y = _g_pid(x, u, t) #pid output
            w = self.Ks * (y - np.clip(y, *self.limits)) #anti-windup feedback
            dx2 = u - w

            return np.array([dx1, dx2])

        #internal operators
        self.op_dyn = DynamicOperator(
            func=_f_pid,
            )
        self.op_alg = DynamicOperator(
            func=_g_pid,
            jac_x=_jac_x_g_pid,
            jac_u=_jac_u_g_pid,
            )
