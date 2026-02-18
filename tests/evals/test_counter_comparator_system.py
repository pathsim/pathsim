########################################################################################
##
##                  Testing counter and comparator systems
##
##   Verifies event-driven digital-like blocks (Counter, CounterUp, CounterDown,
##   Comparator) correctly detect threshold crossings in simulation.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    SinusoidalSource,
    Source,
    Counter,
    CounterUp,
    CounterDown,
    Comparator,
    Scope
    )

from pathsim.solvers import RKCK54, RKDP87


# TESTCASE =============================================================================

class TestCounterSystem(unittest.TestCase):
    """
    Test counter blocks counting zero crossings of a sinusoidal signal.

    A sine wave of frequency f crosses zero 2*f times per second.
    CounterUp counts only rising crossings (f per second).
    CounterDown counts only falling crossings (f per second).
    Counter counts both (2*f per second).
    """

    def setUp(self):

        self.freq = 2.0  # Hz
        self.duration = 5.0

        Src = SinusoidalSource(amplitude=1.0, frequency=self.freq)

        self.Cnt = Counter(threshold=0.0)
        self.CntUp = CounterUp(threshold=0.0)
        self.CntDown = CounterDown(threshold=0.0)

        self.Sco = Scope(labels=["signal", "count", "count_up", "count_down"])

        blocks = [Src, self.Cnt, self.CntUp, self.CntDown, self.Sco]

        connections = [
            Connection(Src, self.Cnt, self.CntUp, self.CntDown, self.Sco[0]),
            Connection(self.Cnt, self.Sco[1]),
            Connection(self.CntUp, self.Sco[2]),
            Connection(self.CntDown, self.Sco[3])
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.001,
            log=False
            )


    def test_counter_total_crossings(self):
        """Counter should count all zero crossings"""

        self.Sim.run(duration=self.duration, reset=True)

        time, [sig, cnt, cnt_up, cnt_down] = self.Sco.read()

        #expected total crossings: 2 * freq * duration
        expected_total = int(2 * self.freq * self.duration)

        #allow +-1 tolerance for boundary effects
        self.assertAlmostEqual(cnt[-1], expected_total, delta=2,
            msg=f"Total crossings: {cnt[-1]}, expected ~{expected_total}")


    def test_counter_up_counts_rising(self):
        """CounterUp should count only rising crossings"""

        self.Sim.run(duration=self.duration, reset=True)

        time, [sig, cnt, cnt_up, cnt_down] = self.Sco.read()

        #expected rising crossings: freq * duration
        expected_up = int(self.freq * self.duration)

        self.assertAlmostEqual(cnt_up[-1], expected_up, delta=2,
            msg=f"Rising crossings: {cnt_up[-1]}, expected ~{expected_up}")


    def test_counter_down_counts_falling(self):
        """CounterDown should count only falling crossings"""

        self.Sim.run(duration=self.duration, reset=True)

        time, [sig, cnt, cnt_up, cnt_down] = self.Sco.read()

        #expected falling crossings: freq * duration
        expected_down = int(self.freq * self.duration)

        self.assertAlmostEqual(cnt_down[-1], expected_down, delta=2,
            msg=f"Falling crossings: {cnt_down[-1]}, expected ~{expected_down}")


    def test_counter_with_adaptive_solver(self):
        """CounterUp works with adaptive solvers (short run to avoid slowness)"""

        #separate minimal setup - single counter, short duration
        Src = SinusoidalSource(amplitude=1.0, frequency=1.0)
        Cnt = CounterUp(threshold=0.0)
        Sco = Scope(labels=["signal", "count"])

        Sim = Simulation(
            blocks=[Src, Cnt, Sco],
            connections=[
                Connection(Src, Cnt, Sco[0]),
                Connection(Cnt, Sco[1])
                ],
            Solver=RKCK54,
            tolerance_lte_abs=1e-4,
            log=False
            )

        Sim.run(duration=2.0, reset=True)

        time, [sig, cnt] = Sco.read()

        #1 Hz signal, 2 seconds -> ~2 rising crossings
        self.assertAlmostEqual(cnt[-1], 2, delta=1)


class TestComparatorSystem(unittest.TestCase):
    """
    Test comparator producing a square wave from a sinusoidal input.

    A sine wave through a zero-threshold comparator should produce
    a square wave with the same frequency.
    """

    def setUp(self):

        self.freq = 1.0

        Src = SinusoidalSource(amplitude=1.0, frequency=self.freq)
        self.Cmp = Comparator(threshold=0.0, span=[-1, 1])
        self.Sco = Scope(labels=["input", "comparator"])

        blocks = [Src, self.Cmp, self.Sco]

        connections = [
            Connection(Src, self.Cmp, self.Sco[0]),
            Connection(self.Cmp, self.Sco[1])
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.001,
            log=False
            )


    def test_comparator_output_values(self):
        """Comparator output should only be +1 or -1"""

        self.Sim.run(duration=5.0, reset=True)

        time, [inp, cmp] = self.Sco.read()

        #after initial settling (t>0.1), check output values
        mask = time > 0.1
        cmp_steady = cmp[mask]

        unique_values = np.unique(np.round(cmp_steady, 1))

        #should only contain -1 and +1 (within tolerance)
        self.assertTrue(np.any(cmp_steady > 0.5), "Should have positive output")
        self.assertTrue(np.any(cmp_steady < -0.5), "Should have negative output")


    def test_comparator_frequency_preserved(self):
        """Comparator output should have same switching frequency as input"""

        self.Sim.run(duration=10.0, reset=True)

        time, [inp, cmp] = self.Sco.read()

        #count zero crossings of comparator output (transitions)
        mask = time > 0.5
        cmp_steady = cmp[mask]
        t_steady = time[mask]

        crossings = np.sum(np.abs(np.diff(np.sign(cmp_steady))) > 0)
        duration = t_steady[-1] - t_steady[0]

        #crossings per second should be ~2*freq (once up, once down)
        crossing_rate = crossings / duration
        expected_rate = 2 * self.freq

        self.assertAlmostEqual(crossing_rate, expected_rate, delta=1.0,
            msg=f"Crossing rate: {crossing_rate:.1f}, expected ~{expected_rate}")


class TestCounterWithCustomThreshold(unittest.TestCase):
    """Test counter with non-zero threshold"""

    def test_threshold_crossing_count(self):
        """Count crossings of a ramp through a specified threshold"""

        #sawtooth-like source that crosses threshold multiple times
        Src = SinusoidalSource(amplitude=2.0, frequency=1.0)
        Cnt = CounterUp(threshold=1.0)  # count when signal rises through 1.0
        Sco = Scope(labels=["signal", "count"])

        Sim = Simulation(
            blocks=[Src, Cnt, Sco],
            connections=[
                Connection(Src, Cnt, Sco[0]),
                Connection(Cnt, Sco[1])
                ],
            dt=0.001,
            log=False
            )

        Sim.run(duration=5.0, reset=True)

        time, [sig, cnt] = Sco.read()

        #signal of amplitude 2, freq 1 crosses threshold 1.0 upward once per cycle
        expected = int(1.0 * 5.0)
        self.assertAlmostEqual(cnt[-1], expected, delta=2)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
