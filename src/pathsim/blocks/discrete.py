#########################################################################################
##
##                              DISCRETE-TIME BLOCKS
##                              (blocks/discrete.py)
##
##         Periodically sampled blocks: hold, FIR/IIR filters, integrator,
##         derivative, state-space, tapped delay line, etc.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
from collections import deque

from scipy.signal import tf2ss

from ._block import Block
from ..utils.register import Register
from ..events.schedule import Schedule
from ..utils.mutable import mutable


# SAMPLE AND HOLD =======================================================================

@mutable
class SampleHold(Block):
    """Zero-order hold: samples the input periodically and holds it at the output.

    .. math::

        y(t) = u(k T + \\tau), \\quad k T + \\tau \\leq t < (k+1) T + \\tau

    Note
    ----
    Supports vector input — each channel is sampled independently.

    Parameters
    ----------
    T : float
        sampling period
    tau : float
        delay before first sample

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic sampling
    """

    def __init__(self, T=1.0, tau=0.0):
        super().__init__()

        self.T = T
        self.tau = tau

        def _sample(t):
            self.outputs.update_from_array(self.inputs.to_array())

        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_sample
                )
            ]


    def __len__(self):
        return 0


#alias matching the Simulink terminology
ZeroOrderHold = SampleHold


# FIRST-ORDER HOLD ======================================================================

@mutable
class FirstOrderHold(Block):
    """First-order hold reconstructor.

    Reconstructs a continuous signal from periodic samples using linear
    extrapolation across one sampling interval. Causal (one-sample-lag)
    variant matching the Simulink ``First-Order Hold`` block.

    Between two consecutive sample times :math:`t_{k-1}` and :math:`t_k`,
    the output is

    .. math::

        y(t) = u_{k-1} + \\frac{u_{k-1} - u_{k-2}}{T} (t - t_{k-1})

    During the very first interval (only one sample captured) the output
    is held at the most recent sample.

    Note
    ----
    Supports vector input — each channel is extrapolated independently.

    Parameters
    ----------
    T : float
        sampling period
    tau : float
        delay before first sample

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic sampling
    """

    def __init__(self, T=1.0, tau=0.0):
        super().__init__()

        self.T = T
        self.tau = tau

        #last two samples and time of latest sample
        self._u_prev = 0.0
        self._u_curr = 0.0
        self._t_curr = tau
        self._n_samples = 0

        def _sample(t):
            self._u_prev = self._u_curr
            self._u_curr = self.inputs.to_array()
            self._t_curr = t
            self._n_samples += 1

        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_sample
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()
        self._u_prev = 0.0
        self._u_curr = 0.0
        self._t_curr = self.tau
        self._n_samples = 0


    def update(self, t):
        if self._n_samples < 2:
            #not enough history yet, hold last sample
            self.outputs.update_from_array(np.atleast_1d(self._u_curr))
            return
        slope = (self._u_curr - self._u_prev) / self.T
        self.outputs.update_from_array(self._u_curr + slope * (t - self._t_curr))


# FIR FILTER ============================================================================

@mutable
class FIR(Block):
    """Discrete-time Finite-Impulse-Response (FIR) filter.

    Applies an FIR filter to a periodically sampled input signal.

    .. math::

        y[n] = b_0 x[n] + b_1 x[n-1] + \\dots + b_N x[n-N]

    where ``b`` are the filter coefficients and ``N`` is the filter order
    (number of coefficients minus one). The output is held constant
    between sample times.

    Note
    ----
    Supports vector input — the same coefficients are applied to each
    channel in parallel.

    Parameters
    ----------
    coeffs : array_like
        FIR filter coefficients ``[b0, b1, ..., bN]``
    T : float
        sampling period
    tau : float
        delay before first sample

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic filter evaluation
    """

    def __init__(self, coeffs=[1.0], T=1.0, tau=0.0):
        super().__init__()

        self.coeffs = np.asarray(coeffs, dtype=float)
        self.T = T
        self.tau = tau

        n = len(self.coeffs)
        self._buffer = deque([0.0] * n, maxlen=n)

        def _update_fir(t):
            self._buffer.appendleft(self.inputs.to_array())
            #weighted sum across taps; broadcasting handles scalar zero pads
            y = sum(c * b for c, b in zip(self.coeffs, self._buffer))
            self.outputs.update_from_array(np.atleast_1d(y))

        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_update_fir
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()
        n = len(self.coeffs)
        self._buffer = deque([0.0] * n, maxlen=n)


# DISCRETE INTEGRATOR ===================================================================

@mutable
class DiscreteIntegrator(Block):
    """Discrete-time integrator (forward Euler).

    .. math::

        y[k+1] = y[k] + T \\, u[k]

    The output at sample ``k`` is the accumulated sum of past inputs;
    the current input ``u[k]`` only enters the next sample.

    Note
    ----
    Supports vector input — each channel is integrated independently.
    Pass an array as ``initial_value`` to set per-channel initial values.

    Parameters
    ----------
    T : float
        sampling period
    tau : float
        delay before first sample
    initial_value : float, array_like
        initial integrator output ``y[0]``

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic update
    """

    def __init__(self, T=1.0, tau=0.0, initial_value=0.0):
        super().__init__()

        self.T = T
        self.tau = tau
        self.initial_value = np.atleast_1d(np.asarray(initial_value, dtype=float))

        self._state = self.initial_value.copy()
        self.outputs.update_from_array(self._state)

        def _update(t):
            self.outputs.update_from_array(self._state)
            self._state = self._state + self.T * self.inputs.to_array()

        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_update
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()
        self._state = self.initial_value.copy()
        self.outputs.update_from_array(self._state)


# DISCRETE DERIVATIVE ===================================================================

@mutable
class DiscreteDerivative(Block):
    """Discrete-time backward-difference derivative.

    .. math::

        y[k] = \\frac{u[k] - u[k-1]}{T}

    Note
    ----
    Supports vector input — each channel is differentiated independently.

    Parameters
    ----------
    T : float
        sampling period
    tau : float
        delay before first sample

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic update
    """

    def __init__(self, T=1.0, tau=0.0):
        super().__init__()

        self.T = T
        self.tau = tau

        self._prev = 0.0

        def _update(t):
            u = self.inputs.to_array()
            self.outputs.update_from_array((u - self._prev) / self.T)
            self._prev = u

        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_update
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()
        self._prev = 0.0


# DISCRETE STATE SPACE ==================================================================

@mutable
class DiscreteStateSpace(Block):
    """Discrete-time MIMO state space block.

    .. math::

        \\begin{align}
            x[k+1] &= \\mathbf{A}\\, x[k] + \\mathbf{B}\\, u[k] \\\\
            y[k]   &= \\mathbf{C}\\, x[k] + \\mathbf{D}\\, u[k]
        \\end{align}

    Note
    ----
    The output port reflects ``y[k]`` for the duration of the current
    sample interval (zero-order hold between updates). The direct
    feedthrough term ``D u[k]`` is computed at the sample event, so the
    block has no algebraic passthrough between updates.

    Parameters
    ----------
    A, B, C, D : array_like
        discrete state space matrices
    T : float
        sampling period
    tau : float
        delay before first sample
    initial_value : array_like, None
        initial state ``x[0]``

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic update
    """

    def __init__(self, A=0.0, B=1.0, C=1.0, D=0.0, T=1.0, tau=0.0, initial_value=None):
        super().__init__()

        self.A = np.atleast_2d(A)
        self.B = np.atleast_1d(B)
        self.C = np.atleast_1d(C)
        self.D = np.atleast_1d(D)
        self.T = T
        self.tau = tau

        n, _ = self.A.shape
        if self.B.ndim == 1:
            n_in = 1
            self._B = self.B.reshape(n, 1) if self.B.size == n else self.B
        else:
            _, n_in = self.B.shape
            self._B = self.B
        if self.C.ndim == 1:
            n_out = 1
            self._C = self.C.reshape(1, n) if self.C.size == n else self.C
        else:
            n_out, _ = self.C.shape
            self._C = self.C
        if self.D.ndim == 1:
            self._D = self.D.reshape(n_out, n_in) if self.D.size == n_out * n_in else np.atleast_2d(self.D)
        else:
            self._D = self.D

        self.inputs = Register(n_in)
        self.outputs = Register(n_out)

        if initial_value is None:
            self.initial_value = np.zeros(n)
        else:
            self.initial_value = np.atleast_1d(initial_value).astype(float)

        self._x = self.initial_value.copy()

        def _update(t):
            u = self.inputs.to_array()
            y = self._C @ self._x + self._D @ u
            self.outputs.update_from_array(np.atleast_1d(y))
            self._x = self.A @ self._x + self._B @ u

        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_update
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()
        self._x = self.initial_value.copy()


# DISCRETE TRANSFER FUNCTION ============================================================

@mutable
class DiscreteTransferFunction(DiscreteStateSpace):
    """Discrete-time SISO transfer function in numerator/denominator form.

    .. math::

        H(z) = \\frac{b_0 z^M + b_1 z^{M-1} + \\dots + b_M}{a_0 z^N + a_1 z^{N-1} + \\dots + a_N}

    Realized internally as a ``DiscreteStateSpace`` via the controllable
    canonical form returned by ``scipy.signal.tf2ss``.

    Parameters
    ----------
    Num : array_like
        numerator polynomial coefficients (highest power of z first)
    Den : array_like
        denominator polynomial coefficients (highest power of z first)
    T : float
        sampling period
    tau : float
        delay before first sample
    """

    input_port_labels = {"in": 0}
    output_port_labels = {"out": 0}

    def __init__(self, Num=[1.0], Den=[1.0, 0.0], T=1.0, tau=0.0):
        self.Num = Num
        self.Den = Den
        A, B, C, D = tf2ss(Num, Den)
        super().__init__(A=A, B=B, C=C, D=D, T=T, tau=tau)


# TAPPED DELAY LINE =====================================================================

@mutable
class TappedDelay(Block):
    """Tapped delay line.

    Outputs the current and ``N-1`` past samples of the input as parallel
    signals. The block has ``N`` outputs:

    .. math::

        y_i[k] = u[k - i], \\quad i = 0, 1, \\dots, N-1

    Parameters
    ----------
    N : int
        number of taps (output ports)
    T : float
        sampling period
    tau : float
        delay before first sample

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event for periodic shift
    """

    def __init__(self, N=2, T=1.0, tau=0.0):
        super().__init__()

        self.N = int(N)
        self.T = T
        self.tau = tau

        self.inputs = Register(1)
        self.outputs = Register(self.N)

        self._buffer = deque([0.0] * self.N, maxlen=self.N)

        def _update(t):
            self._buffer.appendleft(self.inputs[0])
            for i in range(self.N):
                self.outputs[i] = self._buffer[i]

        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=_update
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        super().reset()
        self._buffer = deque([0.0] * self.N, maxlen=self.N)
