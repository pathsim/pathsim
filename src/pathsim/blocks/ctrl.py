#########################################################################################
##
##                                 CONTROL BLOCKS
##                                (blocks/ctrl.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .lti import StateSpace
from ._block import Block

from ..optim.operator import DynamicOperator


# LTI CONTROL BLOCKS (StateSpace subclasses) ============================================

class PT1(StateSpace):
    """First-order lag element (PT1).

    The transfer function is defined as

    .. math::

        H(s) = \\frac{K}{1 + T s}

    where `K` is the static gain and `T` is the time constant.


    Example
    -------
    The block is initialized like this:

    .. code-block:: python

        pt1 = PT1(K=2.0, T=0.5)


    Parameters
    ----------
    K : float
        static gain
    T : float
        time constant in seconds (must be > 0)
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}

    def __init__(self, K=1.0, T=1.0):

        #element parameters
        self.K = K
        self.T = T

        #statespace realization
        super().__init__(
            A=np.array([[-1.0 / T]]),
            B=np.array([[K / T]]),
            C=np.array([[1.0]]),
            D=np.array([[0.0]])
            )


class PT2(StateSpace):
    """Second-order lag element (PT2).

    The transfer function is defined as

    .. math::

        H(s) = \\frac{K}{1 + 2 d T s + T^2 s^2}

    where `K` is the static gain, `T` is the time constant
    (related to the natural frequency by :math:`\\omega_n = 1/T`)
    and `d` is the damping ratio.

    The damping ratio `d` controls the transient behavior:

    - :math:`d < 1`: underdamped (oscillatory)
    - :math:`d = 1`: critically damped
    - :math:`d > 1`: overdamped


    Example
    -------
    The block is initialized like this:

    .. code-block:: python

        #underdamped second-order system
        pt2 = PT2(K=1.0, T=0.1, d=0.3)


    Parameters
    ----------
    K : float
        static gain
    T : float
        time constant in seconds (must be > 0)
    d : float
        damping ratio (must be >= 0)
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}

    def __init__(self, K=1.0, T=1.0, d=1.0):

        #element parameters
        self.K = K
        self.T = T
        self.d = d

        #statespace realization (controllable canonical form)
        super().__init__(
            A=np.array([[0.0, 1.0], [-1.0 / T**2, -2.0 * d / T]]),
            B=np.array([[0.0], [1.0]]),
            C=np.array([[K / T**2, 0.0]]),
            D=np.array([[0.0]])
            )


class LeadLag(StateSpace):
    """Lead-Lag compensator.

    The transfer function is defined as

    .. math::

        H(s) = K \\frac{T_1 s + 1}{T_2 s + 1}

    where `K` is the static gain, `T1` is the lead time constant
    and `T2` is the lag time constant.

    - :math:`T_1 > T_2`: lead compensator (phase advance)
    - :math:`T_1 < T_2`: lag compensator (phase lag)
    - :math:`T_1 = T_2`: pure gain


    Example
    -------
    The block is initialized like this:

    .. code-block:: python

        #lead compensator
        ll = LeadLag(K=1.0, T1=0.5, T2=0.1)


    Parameters
    ----------
    K : float
        static gain
    T1 : float
        lead (numerator) time constant in seconds
    T2 : float
        lag (denominator) time constant in seconds (must be > 0)
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}

    def __init__(self, K=1.0, T1=1.0, T2=1.0):

        #compensator parameters
        self.K = K
        self.T1 = T1
        self.T2 = T2

        #statespace realization
        super().__init__(
            A=np.array([[-1.0 / T2]]),
            B=np.array([[1.0 / T2]]),
            C=np.array([[K * (T2 - T1) / T2]]),
            D=np.array([[K * T1 / T2]])
            )


class PID(StateSpace):
    """Proportional-Integral-Differentiation (PID) controller.

    The transfer function is defined as

    .. math::

        H(s) = K_p + K_i \\frac{1}{s} + K_d \\frac{s}{1 + s / f_\\mathrm{max}}

    where the differentiation is approximated by a high pass filter that holds
    for signals up to a frequency of approximately `f_max`.

    Internally realized as a linear state space model with two states
    (differentiator filter state and integrator state).


    Note
    ----
    Depending on `f_max`, the resulting system might become stiff or ill conditioned!
    As a practical choice set `f_max` to 3x the highest expected signal frequency.
    Since this block uses an approximation of real differentiation, the approximation will
    not hold if there are high frequency components present in the signal. For example if
    you have discontinuities such as steps or square waves.


    Example
    -------
    The block is initialized like this:

    .. code-block:: python

        #cutoff at 1kHz
        pid = PID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3)


    Parameters
    ----------
    Kp : float
        proportional controller coefficient
    Ki : float
        integral controller coefficient
    Kd : float
        differentiator controller coefficient
    f_max : float
        highest expected signal frequency
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}

    def __init__(self, Kp=0, Ki=0, Kd=0, f_max=100):

        #pid controller coefficients
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        #maximum frequency for differentiator approximation
        self.f_max = f_max

        #statespace realization
        #  states: x1 = differentiator filter, x2 = integrator
        #  dx1/dt = f_max * (u - x1)
        #  dx2/dt = u
        #  y = Kp*u + Ki*x2 + Kd*f_max*(u - x1)
        super().__init__(
            A=np.array([[-f_max, 0.0], [0.0, 0.0]]),
            B=np.array([[f_max], [1.0]]),
            C=np.array([[-Kd * f_max, Ki]]),
            D=np.array([[Kd * f_max + Kp]])
            )


class AntiWindupPID(PID):
    """Proportional-Integral-Differentiation (PID) controller with anti-windup mechanism (back-calculation).

    Anti-windup mechanisms are needed when the magnitude of the control signal
    from the PID controller is limited by some real world saturation. In these cases,
    the integrator will continue to accumulate the control error and "wind itself up".
    Once the setpoint is reached, this can result in significant overshoots. This
    implementation adds a conditional feedback term to the internal integrator that
    "unwinds" it when the PID output crosses some limits. This is pretty much a
    deadzone feedback element for the integrator.

    Mathematically, this block implements the following set of ODEs

    .. math::

        \\begin{align}
        \\dot{x}_1 &= f_\\mathrm{max} (u - x_1) \\\\
        \\dot{x}_2 &= u - w
        \\end{align}

    with the anti-windup feedback (depending on the pid output)

    .. math::

        w = K_s (y - \\min(\\max(y, y_\\mathrm{min}), y_\\mathrm{max}))

    and the output itself

    .. math::

        y = K_p u + K_d f_\\mathrm{max} (u - x_1) + K_i x_2


    Note
    ----
    Depending on `f_max`, the resulting system might become stiff or ill conditioned!
    As a practical choice set `f_max` to 3x the highest expected signal frequency.
    Since this block uses an approximation of real differentiation, the approximation will
    not hold if there are high frequency components present in the signal. For example if
    you have discontinuities such as steps or square waves.


    Example
    -------
    The block is initialized like this:

    .. code-block:: python

        #cutoff at 1kHz, windup limits at [-5, 5]
        pid = AntiWindupPID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3, limits=[-5, 5])


    Parameters
    ----------
    Kp : float
        proportional controller coefficient
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
    """

    def __init__(self, Kp=0, Ki=0, Kd=0, f_max=100, Ks=10, limits=[-10, 10]):
        super().__init__(Kp, Ki, Kd, f_max)

        #anti-windup control
        self.Ks = Ks
        self.limits = limits

        #override dynamic operator with nonlinear anti-windup feedback
        def _f_pid(x, u, t):
            x1, x2 = x
            u0 = u[0]

            #differentiator state
            dx1 = self.f_max * (u0 - x1)

            #integrator state with windup control
            y = self.Kp * u0 + self.Ki * x2 + self.Kd * self.f_max * (u0 - x1)
            w = self.Ks * (y - np.clip(y, *self.limits))
            dx2 = u0 - w

            return np.array([dx1, dx2])

        self.op_dyn = DynamicOperator(func=_f_pid)


# NONLINEAR CONTROL BLOCKS ==============================================================

class RateLimiter(Block):
    """Rate limiter block that limits the rate of change of a signal.

    Implements a continuous-time rate limiter as a first-order tracking system
    with clipped rate of change:

    .. math::

        \\dot{x} = \\mathrm{clip}\\left(f_\\mathrm{max} (u - x),\\; -r,\\; r\\right)

    where `r` is the maximum allowed rate and `f_max` controls the tracking
    bandwidth when the signal is not rate-limited. The output is the state
    :math:`y = x`.


    Note
    ----
    The parameter `f_max` should be set high enough that the output tracks
    the input without lag when the rate is within limits.


    Example
    -------
    The block is initialized like this:

    .. code-block:: python

        #max rate of 10 units/s
        rl = RateLimiter(rate=10.0, f_max=1e3)


    Parameters
    ----------
    rate : float
        maximum rate of change (positive value)
    f_max : float
        tracking bandwidth parameter

    Attributes
    ----------
    op_dyn : DynamicOperator
        internal dynamic operator for rate-limited ODE
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}

    def __init__(self, rate=1.0, f_max=100):
        super().__init__()

        #rate limiter parameters
        self.rate = rate
        self.f_max = f_max

        #initial state for integration engine
        self.initial_value = 0.0

        #dynamic operator with clipped rate
        self.op_dyn = DynamicOperator(
            func=lambda x, u, t: np.clip(self.f_max * (u - x), -self.rate, self.rate)
            )


    def __len__(self):
        return 0


    def update(self, t):
        """update system equation for fixed point loop

        Note
        ----
        Rate limiter does not have passthrough, therefore this
        method is performance optimized for this case.

        Parameters
        ----------
        t : float
            evaluation time
        """
        self.outputs.update_from_array(self.engine.state)


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
