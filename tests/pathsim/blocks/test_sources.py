########################################################################################
##
##                                  TESTS FOR 
##                             'blocks.sources.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
from unittest.mock import Mock, patch

from pathsim.blocks.sources import (
    Constant, Source, TriangleWaveSource, SinusoidalSource, GaussianPulseSource,
    SinusoidalPhaseNoiseSource, ChirpPhaseNoiseSource, ChirpSource,
    PulseSource, Pulse, ClockSource, Clock, SquareWaveSource, StepSource, Step
)
from pathsim.events.schedule import Schedule, ScheduleList
from pathsim.solvers import EUF


# TESTS ================================================================================

class TestConstant(unittest.TestCase):
    """Test the implementation of the 'Constant' block class"""

    def test_init(self):
        C = Constant(value=5)
        self.assertEqual(C.value, 5)
        self.assertEqual(C.outputs[0], 0)

    def test_update(self):
        C = Constant(value=5)
        self.assertEqual(C.outputs[0], 0)
        
        err = C.update(0)
        self.assertEqual(C.outputs[0], 5)
        self.assertEqual(err, 0)

    def test_reset(self):
        C = Constant(value=5)
        C.update(0)
        self.assertEqual(C.outputs[0], 5)
        
        C.reset()
        self.assertEqual(C.outputs[0], 0)

    def test_len(self):
        C = Constant()
        self.assertEqual(len(C), 0)  # No algebraic passthrough


class TestSource(unittest.TestCase):
    """Test the implementation of the 'Source' block class"""

    def test_init(self):
        def f(t):
            return np.sin(t)

        S = Source(func=f)
        self.assertEqual(S.func(1), f(1))
        self.assertEqual(S.func(2), f(2))
        self.assertEqual(S.func(3), f(3))

        # Test input validation
        with self.assertRaises(ValueError): 
            S = Source(func=2)

    def test_update(self):
        def f(t):
            return np.sin(t)

        S = Source(func=f)

        # Update block at different times
        for t in [1, 2, 3]:
            S.update(t)
            self.assertEqual(S.outputs[0], f(t))

    def test_decorator_usage(self):
        @Source
        def my_source(t):
            return t**2
        
        # Check that my_source is now a Source block
        self.assertIsInstance(my_source, Source)
        
        my_source.update(3)
        self.assertEqual(my_source.outputs[0], 9)

    def test_len(self):
        S = Source()
        self.assertEqual(len(S), 0)  # No algebraic passthrough


class TestTriangleWaveSource(unittest.TestCase):
    """Test the implementation of the 'TriangleWaveSource' block class"""
    
    def test_init(self):
        # Default
        S = TriangleWaveSource()
        self.assertEqual(S.amplitude, 1)
        self.assertEqual(S.frequency, 1)
        self.assertEqual(S.phase, 0)

        # Specific
        S = TriangleWaveSource(frequency=12, amplitude=0.01, phase=np.pi/2)
        self.assertEqual(S.amplitude, 0.01)
        self.assertEqual(S.frequency, 12)
        self.assertEqual(S.phase, np.pi/2)

    def test_update(self):
        S = TriangleWaveSource(frequency=1, amplitude=2, phase=0)
        
        # Test at t=0 (should be -2)
        S.update(0)
        self.assertAlmostEqual(S.outputs[0], -2, places=10)
        
    def test_len(self):
        S = TriangleWaveSource()
        self.assertEqual(len(S), 0)


class TestSinusoidalSource(unittest.TestCase):
    """Test the implementation of the 'SinusoidalSource' block class"""
    
    def test_init(self):
        # Default
        S = SinusoidalSource()
        self.assertEqual(S.amplitude, 1)
        self.assertEqual(S.frequency, 1)
        self.assertEqual(S.phase, 0)

        # Specific
        S = SinusoidalSource(frequency=50, amplitude=10, phase=np.pi/4)
        self.assertEqual(S.amplitude, 10)
        self.assertEqual(S.frequency, 50)
        self.assertEqual(S.phase, np.pi/4)

    def test_update(self):
        S = SinusoidalSource(frequency=1, amplitude=2, phase=0)
        
        # Test at t=0
        S.update(0)
        self.assertAlmostEqual(S.outputs[0], 0, places=10)
        
        # Test at t=0.25 (quarter period)
        S.update(0.25)
        self.assertAlmostEqual(S.outputs[0], 2, places=10)
        
        # Test with phase
        S2 = SinusoidalSource(frequency=1, amplitude=1, phase=np.pi/2)
        S2.update(0)
        self.assertAlmostEqual(S2.outputs[0], 1, places=10)

    def test_len(self):
        S = SinusoidalSource()
        self.assertEqual(len(S), 0)


class TestGaussianPulseSource(unittest.TestCase):
    """Test the implementation of the 'GaussianPulseSource' block class"""
    
    def test_init(self):
        # Default
        S = GaussianPulseSource()
        self.assertEqual(S.amplitude, 1)
        self.assertEqual(S.f_max, 1e3)
        self.assertEqual(S.tau, 0.0)

        # Specific
        S = GaussianPulseSource(amplitude=5, f_max=2e3, tau=1.0)
        self.assertEqual(S.amplitude, 5)
        self.assertEqual(S.f_max, 2e3)
        self.assertEqual(S.tau, 1.0)

    def test_update(self):
        S = GaussianPulseSource(amplitude=2, f_max=100, tau=0)
        
        # Test at peak (t=0)
        S.update(0)
        self.assertAlmostEqual(S.outputs[0], 2, places=10)
        
        # Test with tau
        S2 = GaussianPulseSource(amplitude=1, f_max=1000, tau=5)
        S2.update(5)  # Peak should be at t=5
        self.assertAlmostEqual(S2.outputs[0], 1, places=10)

    def test_len(self):
        S = GaussianPulseSource()
        self.assertEqual(len(S), 0)


class TestSinusoidalPhaseNoiseSource(unittest.TestCase):
    """Test the implementation of the 'SinusoidalPhaseNoiseSource' block class"""
    
    def test_init(self):
        S = SinusoidalPhaseNoiseSource(
            frequency=100,
            amplitude=2,
            phase=np.pi/4,
            sig_cum=0.1,
            sig_white=0.05,
            sampling_period=0.001
        )
        self.assertEqual(S.frequency, 100)
        self.assertEqual(S.amplitude, 2)
        self.assertEqual(S.phase, np.pi/4)
        self.assertEqual(S.sig_cum, 0.1)
        self.assertEqual(S.sig_white, 0.05)
        self.assertEqual(S.sampling_period, 0.001)
        self.assertEqual(S.omega, 2 * np.pi * 100)

    def test_set_solver(self):
        S = SinusoidalPhaseNoiseSource()
        
        # Mock solver
        MockSolver = Mock(return_value=Mock())
        S.set_solver(EUF, None)
        
        self.assertIsNotNone(S.engine)

    def test_update_and_sample(self):
        S = SinusoidalPhaseNoiseSource(
            frequency=1,
            amplitude=1,
            phase=0,
            sig_cum=0,
            sig_white=0,
            sampling_period=0.1
        )
        
        # Mock the engine
        S.engine = Mock()
        S.engine.get.return_value = 0.0
        S.engine.state = 0.0

        # Test update without noise
        S.update(0)
        self.assertAlmostEqual(S.outputs[0], 0, places=10)
        
        S.update(0.25)  # Quarter period
        self.assertAlmostEqual(S.outputs[0], 1, places=10)

    def test_reset(self):
        S = SinusoidalPhaseNoiseSource()
        
        n1 = S.noise_1
        n2 = S.noise_2
        
        S.reset()
        
        self.assertTrue(S.noise_1 != n1)
        self.assertTrue(S.noise_2 != n2)

    def test_len(self):
        S = SinusoidalPhaseNoiseSource()
        self.assertEqual(len(S), 0)


class TestChirpPhaseNoiseSource(unittest.TestCase):
    """Test the implementation of the 'ChirpPhaseNoiseSource' block class"""
    
    def test_init(self):
        C = ChirpPhaseNoiseSource(
            amplitude=2,
            f0=100,
            BW=200,
            T=1,
            phase=np.pi/2,
            sig_cum=0.1,
            sig_white=0.05,
            sampling_period=0.001
        )
        self.assertEqual(C.amplitude, 2)
        self.assertEqual(C.f0, 100)
        self.assertEqual(C.BW, 200)
        self.assertEqual(C.T, 1)
        self.assertEqual(C.phase, np.pi/2)

    def test_set_solver(self):
        C = ChirpPhaseNoiseSource()
        
        # Mock solver
        C.set_solver(EUF, None)
        
        self.assertIsNotNone(C.engine)

    def test_triangle_wave(self):
        C = ChirpPhaseNoiseSource()
        
        # Test triangle wave at different points
        self.assertAlmostEqual(C._triangle_wave(0, 1), 1, places=10)
        self.assertAlmostEqual(C._triangle_wave(1, 1), 1, places=10)

    def test_update(self):
        C = ChirpPhaseNoiseSource(amplitude=1, phase=0, sig_white=0)
        
        # Mock the engine
        C.engine = Mock()
        C.engine.get.return_value = 0.0
        C.engine.state = 0.0

        C.update(0)
        # Without frequency sweep, should be 0
        self.assertAlmostEqual(C.outputs[0], 0, places=10)

    def test_len(self):
        C = ChirpPhaseNoiseSource()
        self.assertEqual(len(C), 0)


class TestChirpSource(unittest.TestCase):
    """Test the deprecated ChirpSource alias"""
    
    def test_deprecation_warning(self):
        with self.assertWarns(Warning):
            C = ChirpSource()
        
        # Should still work as ChirpPhaseNoiseSource
        self.assertIsInstance(C, ChirpPhaseNoiseSource)


class TestPulseSource(unittest.TestCase):
    """Test the implementation of the 'PulseSource' block class"""
    
    def test_init_default(self):
        P = PulseSource()
        self.assertEqual(P.amplitude, 1.0)
        self.assertEqual(P.T, 1.0)
        self.assertEqual(P.tau, 0.0)
        self.assertEqual(P.duty, 0.5)

    def test_init_validation(self):
        # Test invalid period
        with self.assertRaises(ValueError):
            PulseSource(T=-1)
        
        # Test invalid rise time
        with self.assertRaises(ValueError):
            PulseSource(t_rise=-1)
        
        # Test invalid fall time
        with self.assertRaises(ValueError):
            PulseSource(t_fall=-1)
        
        # Test invalid duty cycle
        with self.assertRaises(ValueError):
            PulseSource(duty=1.5)
        
        # Test total time exceeds period
        with self.assertRaises(ValueError):
            PulseSource(T=1, t_rise=0.4, t_fall=0.4, duty=0.5)

    def test_update_phases(self):
        P = PulseSource(amplitude=2, T=1, t_rise=0.1, t_fall=0.1, duty=0.5)
        
        # Test low phase
        P._phase = 'low'
        P.update(0)
        self.assertEqual(P.outputs[0], 0.0)
        
        # Test high phase
        P._phase = 'high'
        P.update(0.5)
        self.assertEqual(P.outputs[0], 2.0)
        
        # Test rising phase (halfway)
        P._phase = 'rising'
        P._phase_start_time = 0
        P.update(0.05)  # Halfway through rise
        self.assertEqual(P.outputs[0], 1.0)
        
        # Test falling phase (halfway)
        P._phase = 'falling'
        P._phase_start_time = 0.6
        P.update(0.65)  # Halfway through fall
        self.assertAlmostEqual(P.outputs[0], 1.0, places=10)

    def test_events(self):
        P = PulseSource(T=1, t_rise=0.1, t_fall=0.1, duty=0.5)
        
        # Should have 4 scheduled events
        self.assertEqual(len(P.events), 4)
        
        # All should be Schedule events
        for event in P.events:
            self.assertIsInstance(event, Schedule)

    def test_reset(self):
        P = PulseSource()
        P._phase = 'high'
        P._phase_start_time = 5.0
        
        P.reset()
        
        self.assertEqual(P._phase, 'low')
        self.assertEqual(P._phase_start_time, P.tau)

    def test_len(self):
        P = PulseSource()
        self.assertEqual(len(P), 0)
    
    def test_reset_with_time(self):
        """Test the special reset with time functionality"""
        P = PulseSource()

        # set the phase to high and check that the output
        # corresponds to the plateau value
        P._phase = 'high'
        P.update(t=0.1)
        self.assertEqual(P.outputs[0], 1.0)

        # reset the pulse to t=0.7
        # if not resetted correctly, this should output zero (low phase)
        # but here we reset with the `t` argument
        P.reset(t=0.7)
        P.update(t=0.7)
        self.assertEqual(P.outputs[0], 1.0)



class TestPulse(unittest.TestCase):
    """Test the deprecated Pulse alias"""
    
    def test_deprecation_warning(self):
        with self.assertWarns(Warning):
            P = Pulse()
        
        # Should still work as PulseSource
        self.assertIsInstance(P, PulseSource)


class TestClockSource(unittest.TestCase):
    """Test the implementation of the 'ClockSource' block class"""
    
    def test_init(self):
        # Default
        C = ClockSource()
        self.assertEqual(C.T, 1)
        self.assertEqual(C.tau, 0)
        
        # Specific
        C = ClockSource(T=0.1, tau=0.05)
        self.assertEqual(C.T, 0.1)
        self.assertEqual(C.tau, 0.05)

    def test_events(self):
        C = ClockSource(T=1, tau=0.1)
        
        # Should have 2 scheduled events (up and down)
        self.assertEqual(len(C.events), 2)
        
        # Check event timings
        self.assertEqual(C.events[0].t_start, 0.1)  # tau
        self.assertEqual(C.events[0].t_period, 1)    # T
        self.assertEqual(C.events[1].t_start, 0.6)  # tau + T/2
        self.assertEqual(C.events[1].t_period, 1)    # T

    def test_len(self):
        C = ClockSource()
        self.assertEqual(len(C), 0)


class TestClock(unittest.TestCase):
    """Test the deprecated Clock alias"""
    
    def test_deprecation_warning(self):
        with self.assertWarns(Warning):
            C = Clock()
        
        # Should still work as ClockSource
        self.assertIsInstance(C, ClockSource)


class TestSquareWaveSource(unittest.TestCase):
    """Test the implementation of the 'SquareWaveSource' block class"""
    
    def test_init(self):
        # Default
        S = SquareWaveSource()
        self.assertEqual(S.amplitude, 1)
        self.assertEqual(S.frequency, 1)
        self.assertEqual(S.phase, 0)
        
        # Specific
        S = SquareWaveSource(amplitude=5, frequency=50, phase=90)
        self.assertEqual(S.amplitude, 5)
        self.assertEqual(S.frequency, 50)
        self.assertEqual(S.phase, 90)

    def test_events(self):
        S = SquareWaveSource(amplitude=2, frequency=10, phase=0)
        
        # Should have 2 scheduled events
        self.assertEqual(len(S.events), 2)
        
        # Check event timings
        self.assertEqual(S.events[0].t_start, 0)      # phase/360 * period
        self.assertEqual(S.events[0].t_period, 0.1)   # 1/frequency
        self.assertEqual(S.events[1].t_start, 0.05)   # (phase/360 + 0.5) * period
        self.assertEqual(S.events[1].t_period, 0.1)   # 1/frequency

    def test_len(self):
        S = SquareWaveSource()
        self.assertEqual(len(S), 0)


class TestStepSource(unittest.TestCase):
    """Test the implementation of the 'StepSource' block class"""
    
    def test_init(self):
        # Default
        S = StepSource()
        self.assertEqual(S.amplitude, [1])
        self.assertEqual(S.tau, [0.0])
        
        # Specific
        S = StepSource(amplitude=10, tau=5.0)
        self.assertEqual(S.amplitude, [10])
        self.assertEqual(S.tau, [5.0])

        #specific vectorial
        S = StepSource(amplitude=[1, 2, -1, 0], tau=[1, 10, 200, 220])
        self.assertEqual(S.amplitude, [1, 2, -1, 0])
        self.assertEqual(S.tau, [1, 10, 200, 220])

        #input validation, dimension mismatch
        with self.assertRaises(ValueError):
            S = StepSource(amplitude=[1, 2, -1, 0], tau=[1, 10, 200])

        #input validation, wrong type
        with self.assertRaises(ValueError):
            S = StepSource(amplitude="3", tau=2)
        with self.assertRaises(ValueError):
            S = StepSource(amplitude=3, tau="2")

        #validation wrong order of delays (indirectly through `ScheduleList`)
        with self.assertRaises(ValueError):
            S = StepSource(amplitude=[1, 2, 0], tau=[1, 20, 10])


    def test_event(self):
        S = StepSource(amplitude=5, tau=2.0)
        
        # Should have 1 scheduled event
        self.assertEqual(len(S.events), 1)
        self.assertIsInstance(S.Evt, ScheduleList)
        

    def test_len(self):
        S = StepSource()
        self.assertEqual(len(S), 0)


class TestStep(unittest.TestCase):
    """Test the deprecated Step alias"""
    
    def test_deprecation_warning(self):
        with self.assertWarns(Warning):
            S = Step()
        
        # Should still work as StepSource
        self.assertIsInstance(S, StepSource)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)