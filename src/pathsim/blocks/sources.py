#########################################################################################
##
##                            SOURCE BLOCKS (blocks/sources.py)
##
##           This module defines blocks that serve purely as inputs / sources 
##                for the simulation such as the generic 'Source' block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from ..utils.register import Register
from ..utils.deprecation import deprecated
from ..utils.mutable import mutable
from ..events.schedule import Schedule, ScheduleList
from .._constants import TOLERANCE


# GENERIC SOURCE BLOCKS =================================================================

class Constant(Block):
    """Produces a constant output signal (SISO).

    .. math::

        y(t) = const.

        
    Parameters
    ----------
    value : float
        constant defining block output
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(self, value=1):
        super().__init__()

        self.value = value


    def __len__(self):
        """No algebraic passthrough"""
        return 0
        

    def update(self, t):
        """update system equation fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence 
            control (always '0.0' because source-type)
        """
        self.outputs[0] = self.value
        return 0.0


class Source(Block):
    """Source that produces an arbitrary time dependent output defined by `func` (callable).

    .. math::
    
        y(t) = \\mathrm{func}(t)


    Note
    ----
    This block is purely algebraic and its internal function (`func`) will 
    be called multiple times per timestep, each time when `Simulation._update(t)` 
    is called in the global simulation loop.


    Example
    -------
    For example a ramp:

    .. code-block:: python

        from pathsim.blocks import Source

        src = Source(lambda t : t)
    
    or a simple sinusoid with some frequency:

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import Source
    
        #some parameter
        omega = 100
    
        #the function that gets evaluated
        def f(t):
            return np.sin(omega * t)

        src = Source(f)
     
    Because the `Source` block only has a single argument, it can be 
    used to decorate a function and make it a `PathSim` block. This might 
    be handy in some cases to keep definitions concise and localized 
    in the code:

    .. code-block:: python
        
        import numpy as np
        from pathsim.blocks import Source

        #does the same as the definition above
            
        @Source
        def src(t):
            omega = 100
            return np.sin(omega * t)

        #'src' is now a PathSim block


    Parameters
    ---------- 
    func : callable
        function defining time dependent block output
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(self, func=lambda t: 1):
        super().__init__()

        if not callable(func):
            raise ValueError(f"'{func}' is not callable")

        self.func = func


    def __len__(self):
        """No algebraic passthrough"""
        return 0


    def update(self, t):
        """update system equation fixed point loop 
        by evaluating the internal function 'func'

        Note
        ----
        No direct passthrough, so the `update` method 
        is optimized and has no convergence check

        Parameters
        ----------
        t : float
            evaluation time
        """
        self.outputs[0] = self.func(t)


# SPECIAL CONTINUOUS SOURCE BLOCKS ======================================================

@mutable
class TriangleWaveSource(Source):
    """Source block that generates an analog triangle wave
        
    Parameters
    ----------
    frequency : float
        frequency of the triangle wave
    amplitude : float
        amplitude of the triangle wave
    phase : float
        phase of the triangle wave
    """

    def __init__(self, frequency=1, amplitude=1, phase=0):

        #specific params
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

        #phase induced delay
        self._tau = self.phase/(2*np.pi*self.frequency)

        super().__init__(
            func= lambda t: self.amplitude * self._triangle_wave(t + self._tau, self.frequency)
            )

    def _triangle_wave(self, t, f):
        """triangle wave with amplitude '1' and frequency 'f'

        Parameters
        ----------
        t : float
            evaluation time
        f : float
            trig wave frequency

        Returns
        -------
        out : float
            trig wave value
        """
        return 2 * abs(t*f - np.floor(t*f + 0.5)) - 1


@mutable
class SinusoidalSource(Source):
    """Source block that generates a sinusoid wave
        
    Parameters
    ----------
    frequency : float
        frequency of the sinusoid
    amplitude : float
        amplitude of the sinusoid
    phase : float
        phase of the sinusoid
    """

    def __init__(self, frequency=1, amplitude=1, phase=0):

        #block params
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

        #angular frequency
        self._omega = 2*np.pi*self.frequency

        super().__init__(
            func=lambda t: self.amplitude * np.sin(self._omega*t + self.phase)
            )


class GaussianPulseSource(Source):
    """Source block that generates a gaussian pulse
        
    Parameters
    ----------
    amplitude : float
        amplitude of the gaussian pulse
    f_max : float
        maximum frequency component of the gaussian pulse (steepness)
    tau : float
        time delay of the gaussian pulse 
    """

    def __init__(self, amplitude=1, f_max=1e3, tau=0.0):

        #block outputs with port labels
        self.outputs = Register(mapping={"out": 0})

        #block params
        self.amplitude = amplitude
        self.f_max = f_max
        self.tau = tau

        super().__init__(
            func=lambda t: self.amplitude * self._gaussian(t-self.tau, self.f_max)
            )


    def _gaussian(self, t, f_max):
        """gaussian pulse with its maximum at t=0
        
        Parameters
        ----------
        t : float
            evaluation time
        f_max : float
            maximum frequency component of gaussian

        Returns
        -------
        out : float
            gaussian value
        """
        tau = 0.5 / f_max
        return np.exp(-(t/tau)**2)


@mutable
class SinusoidalPhaseNoiseSource(Block):
    """Sinusoidal source with cumulative and white phase noise.
    
    Generates a sinusoid with additive phase noise from two components:
    
    - White phase noise: sampled from a normal distribution at each sample
    - Cumulative phase noise: integrated random walk process
    
    The output is given by:
    
    .. math::
    
        y(t) = A \\sin\\left(\\omega t + \\varphi_0 + \\sigma_w n_w(t) + \\sigma_c \\int_0^t n_c(\\tau) d\\tau\\right)
    
    where :math:`A` is amplitude, :math:`\\omega = 2\\pi f` is angular frequency,
    :math:`\\varphi_0` is initial phase, :math:`\\sigma_w` and :math:`\\sigma_c` are
    the white and cumulative noise weights, and :math:`n_w(t)` and :math:`n_c(t)` are
    independent standard normal random processes sampled at the specified sampling period.
    
    Parameters
    ----------
    frequency : float
        frequency of the sinusoid
    amplitude : float
        amplitude of the sinusoid
    phase : float
        initial phase of the sinusoid (radians)
    sig_cum : float
        weight for cumulative phase noise contribution
    sig_white : float
        weight for white phase noise contribution
    sampling_period : float, None
        time between phase noise samples. If None,
        noise is sampled every timestep (default is 0.1)

    Attributes
    ----------
    omega : float
        angular frequency of the sinusoid, derived from `frequency`
    noise_1 : float
        internal noise value for white phase noise
    noise_2 : float
        internal noise value for cumulative phase noise
    events : list[Schedule]
        scheduled event for periodic sampling (only if sampling_period is set)
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(
        self,
        frequency=1,
        amplitude=1,
        phase=0,
        sig_cum=0,
        sig_white=0,
        sampling_period=0.1
        ):
        super().__init__()

        #block params
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.sampling_period = sampling_period

        self.omega = 2 * np.pi * self.frequency

        #parameters for phase noise
        self.sig_cum = sig_cum
        self.sig_white = sig_white

        #initial noise sampling
        self.noise_1 = np.random.normal()
        self.noise_2 = np.random.normal()

        #initial state for integration engine
        self.initial_value = 0.0

        #sampling produces discrete time behavior for noise
        if sampling_period is None:
            pass  # sample every timestep
        else:
            #internal scheduled event for noise sampling
            def _sample_noise(t):
                self.noise_1 = np.random.normal()
                self.noise_2 = np.random.normal()

            self.events = [
                Schedule(
                    t_start=0,
                    t_period=sampling_period,
                    func_act=_sample_noise
                )
            ]


    def __len__(self):
        return 0


    def reset(self):
        """Reset block state including noise samples."""
        super().reset()

        #reset noise samples
        self.noise_1 = np.random.normal()
        self.noise_2 = np.random.normal()


    def update(self, t):
        """Update system equation for fixed point loop, evaluating the 
        sinusoid with phase noise.
    
        Parameters
        ----------
        t : float
            evaluation time
        """
        #compute phase error from white and cumulative noise
        phase_error = self.sig_white * self.noise_1 + self.sig_cum * self.engine.state

        #set output
        self.outputs[0] = self.amplitude * np.sin(self.omega*t + self.phase + phase_error)


    def sample(self, t, dt):
        """Sample from a normal distribution after successful timestep.

        Only used when sampling_period is None (continuous sampling).

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
        """
        if self.sampling_period is None:
            self.noise_1 = np.random.normal()
            self.noise_2 = np.random.normal()


    def solve(self, t, dt):
        """Advance solution of implicit update equation for cumulative noise integration.
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
            
        Returns
        -------
        float
            error estimate (always 0.0 for noise source)
        """
        #advance solution of implicit update equation (no jacobian)
        f = self.noise_2
        self.engine.solve(f, None, dt)
        return 0.0


    def step(self, t, dt):
        """Compute update step with integration engine for cumulative noise.
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
            
        Returns
        -------
        tuple
            (accepted, error, scale_factor) - always (True, 0.0, None) for noise
        """
        #compute update step with integration engine
        f = self.noise_2
        self.engine.step(f, dt)

        #no error control for noise source
        return True, 0.0, None


class ChirpPhaseNoiseSource(Block):
    """Chirp source, sinusoid with frequency ramp up and ramp down, plus phase noise.

    This works by using a time dependent triangle wave for the frequency 
    and integrating it with a numerical integration engine to get a 
    continuous phase. This phase is then used to evaluate a sinusoid.

    Additionally the chirp source can have white and cumulative phase noise. 
    Mathematically it looks like this for the contributions to the phase from 
    the triangular wave:

    .. math::

        \\varphi_t(t) = \\int_0^t \\mathrm{tri}_{f_0, B, T}(\\tau) \\, d\\tau
    
    And from the white (w) and cumulative (c) noise:

    .. math::

        \\varphi_n(t) = \\sigma_w \\, n_w(t) + \\sigma_c \\int_0^t n_c(\\tau) \\, d\\tau
    
    The phase contributions are then used to evaluate a sinusoid to get the final chirp signal:

    .. math::

        y(t) = A \\sin(\\varphi_t(t) + \\varphi_n(t) + \\varphi_0)

    Parameters
    ----------
    amplitude : float
        amplitude of the chirp signal
    f0 : float
        start frequency of the chirp signal
    BW : float
        bandwidth of the frequency ramp of the chirp signal
    T : float
        period of the frequency ramp of the chirp signal
    phase : float
        phase of sinusoid (initial, radians)
    sig_cum : float
        weight for cumulative phase noise contribution
    sig_white : float
        weight for white phase noise contribution
    sampling_period : float, None
        time between phase noise samples. If None,
        noise is sampled every timestep (default is 0.1)

    Attributes
    ----------
    noise_1 : float
        internal noise value for white phase noise
    noise_2 : float
        internal noise value for cumulative phase noise
    events : list[Schedule]
        scheduled event for periodic sampling (only if sampling_period is set)
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(
        self,
        amplitude=1,
        f0=1,
        BW=1,
        T=1,
        phase=0,
        sig_cum=0,
        sig_white=0,
        sampling_period=0.1
        ):
        super().__init__()

        #parameters of chirp signal
        self.amplitude = amplitude
        self.phase = phase
        self.f0 = f0
        self.BW = BW
        self.T = T

        #parameters for phase noise
        self.sig_cum = sig_cum
        self.sig_white = sig_white
        self.sampling_period = sampling_period

        #initial noise sampling
        self.noise_1 = np.random.normal()
        self.noise_2 = np.random.normal()

        #initial state for integration engine
        self.initial_value = 0.0

        #sampling produces discrete time behavior for noise
        if sampling_period is None:
            pass  # sample every timestep
        else:
            #internal scheduled event for noise sampling
            def _sample_noise(t):
                self.noise_1 = np.random.normal()
                self.noise_2 = np.random.normal()

            self.events = [
                Schedule(
                    t_start=0,
                    t_period=sampling_period,
                    func_act=_sample_noise
                )
            ]


    def __len__(self):
        return 0


    def _triangle_wave(self, t, f):
        """Triangle wave from -1 to +1 with frequency f.

        Parameters
        ----------
        t : float
            evaluation time
        f : float
            triangle wave frequency

        Returns
        -------
        out : float
            triangle wave value
        """
        return 2 * abs(2 * ((t * f) % 1.0) - 1) - 1


    def reset(self):
        """Reset block state including noise samples."""
        super().reset()

        #reset noise samples
        self.noise_1 = np.random.normal()
        self.noise_2 = np.random.normal()


    def sample(self, t, dt):
        """Sample from a normal distribution after successful timestep
        to update internal noise samples.

        Only used when sampling_period is None (continuous sampling).

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
        """
        if self.sampling_period is None:
            self.noise_1 = np.random.normal()
            self.noise_2 = np.random.normal()


    def update(self, t):
        """Update the block output, assemble phase and evaluate the sinusoid.
        
        Parameters
        ----------
        t : float
            evaluation time
        """
        _phase = 2 * np.pi * (self.engine.state + self.sig_white * self.noise_1) + self.phase
        self.outputs[0] = self.amplitude * np.sin(_phase)


    def solve(self, t, dt):
        """Advance implicit solver of implicit integration engine, evaluate 
        the triangle wave and cumulative noise RNG.
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
            
        Returns
        -------
        float
            error estimate (always 0.0 for chirp source)
        """
        f = self.f0 + self.BW * (1 + self._triangle_wave(t, 1/self.T))/2 + self.sig_cum * self.noise_2
        self.engine.solve(f, None, dt)

        #no error for chirp source
        return 0.0


    def step(self, t, dt):
        """Compute update step with integration engine, evaluate the triangle wave 
        and cumulative noise RNG.
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
            
        Returns
        -------
        tuple
            (accepted, error, scale_factor) - always (True, 0.0, None) for chirp
        """
        f = self.f0 + self.BW * (1 + self._triangle_wave(t, 1/self.T))/2 + self.sig_cum * self.noise_2
        self.engine.step(f, dt)

        #no error control for chirp source
        return True, 0.0, None
        

@deprecated(version="1.0.0", replacement="ChirpPhaseNoiseSource")
class ChirpSource(ChirpPhaseNoiseSource):
    """Alias for ChirpPhaseNoiseSource."""
    pass



# SPECIAL DISCRETE SOURCE BLOCKS ========================================================

@mutable
class PulseSource(Block):
    """Generates a periodic pulse waveform with defined rise and fall times.

    Scheduled events trigger phase changes (low, rising, high, falling),
    and the `update` method calculates the output value based on the
    current phase, performing linear interpolation during rise and fall.

    Parameters
    ----------
    amplitude : float, optional
        Peak amplitude of the pulse. Default is 1.0.
    T : float, optional
        Period of the pulse train. Must be positive. Default is 1.0.
    t_rise : float, optional
        Duration of the rising edge. Default is 0.0.
    t_fall : float, optional
        Duration of the falling edge. Default is 0.0.
    tau : float, optional
        Initial delay before the first pulse cycle begins. Default is 0.0.
    duty : float, optional
        Duty cycle, ratio of the pulse ON duration (plateau time only)
        to the total period T (must be between 0 and 1). Default is 0.5.
        The high plateau duration is `T * duty`.

    Attributes
    ----------
    events : list[Schedule]
        Internal scheduled events triggering phase transitions.
    _phase : str
        Current phase of the pulse ('low', 'rising', 'high', 'falling').
    _phase_start_time : float
        Simulation time when the current phase began.
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(
        self, 
        amplitude=1.0, 
        T=1.0, 
        t_rise=0.0, 
        t_fall=0.0, 
        tau=0.0, 
        duty=0.5
        ):
        super().__init__()

        #input validation
        if not (T > 0):
            raise ValueError("Period T must be positive.")
        if not (0 <= t_rise):
            raise ValueError("Rise time t_rise cannot be negative.")
        if not (0 <= t_fall):
            raise ValueError("Fall time t_fall cannot be negative.")
        if not (0 <= duty <= 1):
            raise ValueError("Duty cycle must be between 0 and 1.")

        #ensure rise + high plateau + fall fits within a period
        t_plateau = T * duty
        if t_rise + t_plateau + t_fall > T:
            raise ValueError("Total pulse time (rise+plateau+fall) exceeds period T")

        #parameters
        self.amplitude = amplitude
        self.T = T
        self.t_rise = max(TOLERANCE, t_rise)
        self.t_fall = max(TOLERANCE, t_fall)
        self.tau = tau
        self.duty = duty # Duty cycle now refers to the high plateau time

        #internal state
        self._phase = 'low'
        self._phase_start_time = self.tau 

        #event timings relative to start of cycle (tau)
        t_start_rise = self.tau
        t_start_high = t_start_rise + self.t_rise 
        t_start_fall = t_start_high + t_plateau 
        t_start_low  = t_start_fall + self.t_fall 

        #define event actions (update phase and start time) 
        def _set_phase_rising(t):
            self._phase = 'rising'
            self._phase_start_time = t
            self.outputs[0] = 0.0    

        def _set_phase_high(t):
            self._phase = 'high'
            self._phase_start_time = t
            self.outputs[0] = self.amplitude

        def _set_phase_falling(t):
            self._phase = 'falling'
            self._phase_start_time = t
            self.outputs[0] = self.amplitude

        def _set_phase_low(t):
            self._phase = 'low'
            self._phase_start_time = t
            self.outputs[0] = 0.0    

        #start rising
        _E_rising = Schedule( 
            t_start=max(0.0, t_start_rise), 
            t_period=self.T, 
            func_act=_set_phase_rising
            )

        #start high plateau (end rising)
        _E_high = Schedule(
            t_start=max(0.0, t_start_high), 
            t_period=self.T, 
            func_act=_set_phase_high
            )

        #start falling
        _E_falling = Schedule( 
            t_start=max(0.0, t_start_fall), 
            t_period=self.T, 
            func_act=_set_phase_falling
            )

        #start low (end falling)
        _E_low = Schedule( 
            t_start=max(0.0, t_start_low), 
            t_period=self.T, 
            func_act=_set_phase_low
            )
        
        #scheduled events for state transitions
        self.events = [_E_rising, _E_high, _E_falling, _E_low]


    def reset(self, t: float=None):
        """
        Resets the block state.
        
        Note
        ----
            This block has a special implementation of reset where ``t`` can be provided
            to reset the block's state to the specified time.
            This is done by changing the phase of the pulse + resetting all the internal events.

        Parameters
        ----------
            t: float, optional
                Time to reset the block state at. If None, resets to initial state.

        """
        if t:
            self._phase_start_time = t

            # event timings relative to start of cycle (tau)
            new_t_start_rise = t
            new_t_start_high = new_t_start_rise + self.t_rise
            t_plateau = self.T * self.duty
            new_t_start_fall = new_t_start_high + t_plateau
            new_t_start_low = new_t_start_fall + self.t_fall

            self.events[0].t_start = max(0.0, new_t_start_rise)
            self.events[1].t_start = max(0.0, new_t_start_high)
            self.events[2].t_start = max(0.0, new_t_start_fall)
            self.events[3].t_start = max(0.0, new_t_start_low)

            for e in self.events:
                e.reset()
        else:
            super().reset()
            self._phase = 'low'
            self._phase_start_time = self.tau

    def update(self, t):
        """Calculate the pulse output value based on the current phase.
        Performs linear interpolation during 'rising' and 'falling' phases.

        Parameters
        ----------
        t : float
            evaluation time
        """

        #calculate output based on phase
        if self._phase == 'rising':
            _val = self.amplitude * (t - self._phase_start_time) / self.t_rise
            self.outputs[0] = np.clip(_val, 0.0, self.amplitude)
        elif self._phase == 'high':
            self.outputs[0] = self.amplitude
        elif self._phase == 'falling':
            _val = self.amplitude * (1.0 - (t - self._phase_start_time) / self.t_fall)
            self.outputs[0] = np.clip(_val, 0.0, self.amplitude)
        elif self._phase == 'low':
            self.outputs[0] = 0.0


    def __len__(self):
        #no algebraic passthrough
        return 0


@deprecated(version="1.0.0", replacement="PulseSource")
class Pulse(PulseSource):
    """Alias for PulseSource."""
    pass


@mutable
class ClockSource(Block):
    """Discrete time clock source block.
    
    Utilizes scheduled events to periodically set 
    the block output to 0 or 1 at discrete times.

    Parameters
    ----------
    T : float
        period of the clock
    tau : float
        clock delay

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled event list 
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(self, T=1, tau=0):
        super().__init__()

        #block params
        self.T   = T
        self.tau = tau

        def clk_up(t):
            self.outputs[0] = 1

        def clk_down(t):
            self.outputs[0] = 0

        #internal scheduled events
        self.events = [
            Schedule(
                t_start=tau,
                t_period=T,
                func_act=clk_up
                ),
            Schedule(
                t_start=tau+T/2,
                t_period=T,
                func_act=clk_down
                )
            ]

    def __len__(self):
        #no algebraic passthrough
        return 0


@deprecated(version="1.0.0", replacement="ClockSource")
class Clock(ClockSource):
    """Alias for ClockSource."""
    pass



@mutable
class SquareWaveSource(Block):
    """Discrete time square wave source.
    
    Utilizes scheduled events to periodically set 
    the block output at discrete times.

    Parameters
    ----------
    amplitude : float
        amplitude of the square wave signal
    frequency : float
        frequency of the square wave signal
    phase : float
        phase of the square wave signal

    Attributes
    ----------
    events : list[Schedule]
        internal scheduled events 
    """

    input_port_labels = {}
    output_port_labels = {"out":0}

    def __init__(self, amplitude=1, frequency=1, phase=0):
        super().__init__()

        #block params
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

        def sqw_up(t):
            self.outputs[0] = self.amplitude

        def sqw_down(t):
            self.outputs[0] = -self.amplitude

        #internal scheduled events
        self.events = [
            Schedule(
                t_start=1/frequency * phase/360,
                t_period=1/frequency,
                func_act=sqw_up
                ),
            Schedule(
                t_start=1/frequency * (phase/360 + 0.5),
                t_period=1/frequency,
                func_act=sqw_down
                )
            ]

    def __len__(self):
        #no algebraic passthrough
        return 0


class StepSource(Block):
    """Discrete time unit step (or multi step) source block.
    
    Utilizes a scheduled event to set the block output 
    to the specified output levels at the defined event times.

    The arguments can be vectorial and in that case, the output is set to the 
    amplitude that corresponds to the defined delay like a zero-order-hold stage. 
    This functionality enables adding external or time series measurement data 
    into the system.


    Examples
    --------

    This is how to use the source as a unit step source:

    .. code-block:: python

        from pathsim.blocks import StepSource
        
        #default, starts at 0, jumps to 1
        stp = StepSource()


    And this is how to configure it with multiple consecutive steps:

    .. code-block:: python

        from pathsim.blocks import StepSource
        
        #starts at 0, jumps to 1 at 1, jumps to -1 at 2 and jumps back to 0 at 3
        stp = StepSource(amplitude=[1, -1, 0], tau=[1, 2, 3])


    Similarly implementing measured time series data via zoh:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import StepSource
        
        #some random time series arrays
        times, data = np.linspace(0, 100, 1000), np.random.rand(1000)
        
        #pass them to the block
        stp = StepSource(amplitude=data, tau=times)


    Parameters
    ----------
    amplitude : float | list[float]
        amplitude of the step signal, or amplitudes / output 
        levels of the multiple steps
    tau : float | list[float]
        delay of the step, or delays of the different steps

    Attributes
    ----------
    Evt : ScheduleList
        internal scheduled event directly accessible
    events : list[ScheduleList]
        list of interna events
    """
    input_port_labels = {}
    output_port_labels = {"out":0}
    
    def __init__(self, amplitude=1, tau=0.0):
        super().__init__()

        #input type validation
        if not isinstance(amplitude, (int, float, list, np.ndarray)):
            raise ValueError(f"'amplitude' has to be float, or array of floarts, but is {type(amplitude)}")
        if not isinstance(tau, (int, float, list, np.ndarray)):
            raise ValueError(f"'tau' has to be float, or array of floarts, but is {type(tau)}!") 

        self.amplitude = amplitude if isinstance(amplitude, (list, np.ndarray)) else [amplitude]
        self.tau = tau if isinstance(tau, (list, np.ndarray)) else [tau]

        #input shape validation
        if len(self.amplitude) != len(self.tau):
            raise ValueError("'amplitude' and 'tau' must have same dimensions!")

        #internal scheduled list event
        def stp_set(t):
            idx = len(self.Evt) - 1
            self.outputs[0] = self.amplitude[idx]

        self.Evt = ScheduleList(
            times_evt=self.tau,
            func_act=stp_set
            )
        self.events = [self.Evt]

    def __len__(self):
        #no algebraic passthrough
        return 0


@deprecated(version="1.0.0", replacement="StepSource")
class Step(StepSource):
    """Alias for StepSource."""
    pass