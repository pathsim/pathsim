"""Tests for simulation diagnostics."""

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope
from pathsim.utils.diagnostics import Diagnostics, ConvergenceTracker, StepTracker


class TestDiagnosticsOff(unittest.TestCase):
    """Verify diagnostics=False (default) has no side effects."""

    def test_diagnostics_none_by_default(self):
        src = Source(lambda t: 1.0)
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01
        )
        self.assertIsNone(sim.diagnostics)
        sim.run(0.1)
        self.assertIsNone(sim.diagnostics)


class TestDiagnosticsExplicitSolver(unittest.TestCase):
    """Diagnostics with an explicit solver (step errors only)."""

    def test_snapshot_after_run(self):
        src = Source(lambda t: np.sin(t))
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            diagnostics=True
        )
        sim.run(0.1)

        diag = sim.diagnostics
        self.assertIsInstance(diag, Diagnostics)
        self.assertAlmostEqual(diag.time, sim.time, places=6)

        #explicit solver: step errors should be populated
        self.assertGreater(len(diag.step_errors), 0)
        first_key = list(diag.step_errors.keys())[0]
        self.assertEqual(first_key.__class__.__name__, "Integrator")

        #no implicit solver or algebraic loops
        self.assertEqual(len(diag.solve_residuals), 0)
        self.assertEqual(len(diag.loop_residuals), 0)

    def test_worst_block(self):
        src = Source(lambda t: 1.0)
        i1 = Integrator()
        i2 = Integrator()
        sim = Simulation(
            blocks=[src, i1, i2],
            connections=[Connection(src, i1), Connection(i1, i2)],
            dt=0.01,
            diagnostics=True
        )
        sim.run(0.1)

        result = sim.diagnostics.worst_block()
        self.assertIsNotNone(result)
        label, err = result
        self.assertIn("Integrator", label)

    def test_summary_string(self):
        src = Source(lambda t: 1.0)
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            diagnostics=True
        )
        sim.run(0.1)

        summary = sim.diagnostics.summary()
        self.assertIn("Diagnostics at t", summary)
        self.assertIn("Integrator", summary)

    def test_reset_clears_diagnostics(self):
        src = Source(lambda t: 1.0)
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            diagnostics=True
        )
        sim.run(0.1)
        self.assertGreater(sim.diagnostics.time, 0)

        sim.reset()
        self.assertEqual(sim.diagnostics.time, 0.0)


class TestDiagnosticsAdaptiveSolver(unittest.TestCase):
    """Diagnostics with an adaptive solver."""

    def test_adaptive_step_errors(self):
        from pathsim.solvers import RKCK54

        src = Source(lambda t: np.sin(10 * t))
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.1,
            Solver=RKCK54,
            tolerance_lte_abs=1e-6,
            tolerance_lte_rel=1e-4,
            diagnostics=True
        )
        sim.run(1.0)

        diag = sim.diagnostics
        self.assertIsInstance(diag, Diagnostics)
        self.assertGreater(len(diag.step_errors), 0)


class TestDiagnosticsImplicitSolver(unittest.TestCase):
    """Diagnostics with an implicit solver (solve residuals)."""

    def test_implicit_solve_residuals(self):
        from pathsim.solvers import ESDIRK32

        src = Source(lambda t: np.sin(t))
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            Solver=ESDIRK32,
            diagnostics=True
        )
        sim.run(0.1)

        diag = sim.diagnostics
        self.assertIsInstance(diag, Diagnostics)
        self.assertGreater(len(diag.solve_residuals), 0)
        self.assertGreater(diag.solve_iterations, 0)

        #worst_block should find the block from solve residuals
        result = diag.worst_block()
        self.assertIsNotNone(result)

        #summary should include implicit solver section
        summary = diag.summary()
        self.assertIn("Implicit solver residuals", summary)


class TestDiagnosticsAlgebraicLoop(unittest.TestCase):
    """Diagnostics with algebraic loops (loop residuals)."""

    def test_algebraic_loop_residuals(self):
        src = Source(lambda t: 1.0)
        P1 = Adder()
        A1 = Amplifier(0.5)
        sco = Scope()

        sim = Simulation(
            blocks=[src, P1, A1, sco],
            connections=[
                Connection(src, P1),
                Connection(P1, A1, sco),
                Connection(A1, P1[1]),
            ],
            dt=0.01,
            diagnostics=True
        )

        self.assertTrue(sim.graph.has_loops)
        sim.run(0.05)

        diag = sim.diagnostics
        self.assertGreater(len(diag.loop_residuals), 0)

        result = diag.worst_booster()
        self.assertIsNotNone(result)

        #summary should include algebraic loop section
        summary = diag.summary()
        self.assertIn("Algebraic loop residuals", summary)


class TestDiagnosticsHistory(unittest.TestCase):
    """Diagnostics history recording."""

    def test_no_history_by_default(self):
        src = Source(lambda t: 1.0)
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            diagnostics=True
        )
        sim.run(0.1)

        self.assertIsNone(sim._diagnostics_history)
        self.assertIsInstance(sim.diagnostics, Diagnostics)

    def test_history_enabled(self):
        src = Source(lambda t: 1.0)
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            diagnostics="history"
        )
        sim.run(0.1)

        #should have ~10 snapshots (0.1s / 0.01 dt)
        self.assertGreater(len(sim._diagnostics_history), 5)

        #each snapshot should have a time
        times = [s.time for s in sim._diagnostics_history]
        self.assertEqual(times, sorted(times))

    def test_history_reset(self):
        src = Source(lambda t: 1.0)
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            diagnostics="history"
        )
        sim.run(0.05)
        self.assertGreater(len(sim._diagnostics_history), 0)

        sim.reset()
        self.assertEqual(len(sim._diagnostics_history), 0)


class TestDiagnosticsUnit(unittest.TestCase):
    """Unit tests for the Diagnostics dataclass."""

    def test_defaults(self):
        d = Diagnostics()
        self.assertEqual(d.time, 0.0)
        self.assertIsNone(d.worst_block())
        self.assertIsNone(d.worst_booster())

    def test_worst_block_from_step_errors(self):

        class FakeBlock:
            pass

        b1, b2 = FakeBlock(), FakeBlock()
        d = Diagnostics(step_errors={b1: (True, 1e-3, 0.9), b2: (True, 5e-3, 0.7)})

        label, err = d.worst_block()
        self.assertAlmostEqual(err, 5e-3)

    def test_worst_block_from_solve_residuals(self):

        class FakeBlock:
            pass

        b1, b2 = FakeBlock(), FakeBlock()
        d = Diagnostics(solve_residuals={b1: 1e-4, b2: 3e-3})

        label, err = d.worst_block()
        self.assertAlmostEqual(err, 3e-3)

    def test_summary_with_all_data(self):

        class FakeBlock:
            pass

        class FakeBooster:
            class connection:
                def __str__(self):
                    return "A -> B"
            connection = connection()

        b = FakeBlock()
        bst = FakeBooster()
        d = Diagnostics(
            time=1.0,
            step_errors={b: (True, 1e-4, 0.9)},
            solve_residuals={b: 1e-8},
            solve_iterations=3,
            loop_residuals={bst: 1e-12},
            loop_iterations=2,
        )

        summary = d.summary()
        self.assertIn("Diagnostics at t", summary)
        self.assertIn("Adaptive step errors", summary)
        self.assertIn("Implicit solver residuals", summary)
        self.assertIn("Algebraic loop residuals", summary)


class TestConvergenceTrackerUnit(unittest.TestCase):
    """Unit tests for ConvergenceTracker."""

    def test_record_and_converge(self):
        t = ConvergenceTracker()
        t.record("a", 1e-5)
        t.record("b", 1e-8)
        self.assertAlmostEqual(t.max_error, 1e-5)
        self.assertTrue(t.converged(1e-4))
        self.assertFalse(t.converged(1e-6))

    def test_begin_iteration_clears(self):
        t = ConvergenceTracker()
        t.record("a", 1.0)
        t.begin_iteration()
        self.assertEqual(len(t.errors), 0)
        self.assertEqual(t.max_error, 0.0)

    def test_details(self):
        t = ConvergenceTracker()
        t.record("block_a", 1e-3)
        t.record("block_b", 2e-4)
        lines = t.details(lambda obj: f"name:{obj}")
        self.assertEqual(len(lines), 2)
        self.assertIn("name:block_a", lines[0])


class TestStepTrackerUnit(unittest.TestCase):
    """Unit tests for StepTracker."""

    def test_record_aggregation(self):
        t = StepTracker()
        t.record("a", True, 1e-4, 0.9)
        t.record("b", False, 2e-3, 0.5)
        t.record("c", True, 1e-5, None)

        self.assertFalse(t.success)
        self.assertAlmostEqual(t.max_error, 2e-3)
        self.assertAlmostEqual(t.scale, 0.5)

    def test_scale_default(self):
        t = StepTracker()
        t.record("a", True, 0.0, None)
        self.assertEqual(t.scale, 1.0)

    def test_reset(self):
        t = StepTracker()
        t.record("a", False, 1.0, 0.1)
        t.reset()
        self.assertTrue(t.success)
        self.assertEqual(t.max_error, 0.0)
        self.assertEqual(len(t.errors), 0)
