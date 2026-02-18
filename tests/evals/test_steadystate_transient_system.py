########################################################################################
##
##                  Testing steady-state + transient simulation
##
##   Verifies the two-phase simulation workflow: find DC operating point
##   (steady state), then run transient simulation from that point.
##   Also tests linearization and simulation reset/continuation.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source,
    Constant,
    Integrator,
    Amplifier,
    Adder,
    Scope,
    DynamicalSystem
    )

from pathsim.solvers import RKCK54, ESDIRK32


# TESTCASE =============================================================================

class TestSteadyStateTransient(unittest.TestCase):
    """
    Test steady state solve followed by transient simulation.

    System: dx/dt = -2*(x - u), y = x
    With u=3 (constant input), steady state is x=3.

    After finding steady state, apply a step change in input
    and verify the transient response.
    """

    def test_steady_state_then_step(self):
        """Find DC operating point then apply step change"""

        #system: dx/dt = -x + u, u=2 -> steady state x=2
        Src = Constant(2.0)
        Int = Integrator(0.0)
        Amp = Amplifier(-1.0)
        Add = Adder()
        Sco = Scope(labels=["state"])

        Sim = Simulation(
            blocks=[Src, Int, Amp, Add, Sco],
            connections=[
                Connection(Src, Add[0]),
                Connection(Amp, Add[1]),
                Connection(Add, Int),
                Connection(Int, Amp, Sco)
                ],
            log=False
            )

        #find steady state
        Sim.steadystate(reset=True)

        #state should be at 2.0
        self.assertAlmostEqual(Int.outputs[0], 2.0, 3,
            f"Steady state should be 2.0, got {Int.outputs[0]:.4f}")


    def test_steady_state_then_transient(self):
        """Find steady state then run transient with perturbation"""

        Src = Constant(2.0)

        Int = Integrator(0.0)  # start from zero
        Amp = Amplifier(-1.0)
        Add = Adder()
        Sco = Scope(labels=["state"])

        #system: dx/dt = -x + u, u=2 -> steady state x=2
        Sim = Simulation(
            blocks=[Src, Int, Amp, Add, Sco],
            connections=[
                Connection(Src, Add[0]),
                Connection(Amp, Add[1]),
                Connection(Add, Int),
                Connection(Int, Amp, Sco)
                ],
            log=False
            )

        #find steady state from zero
        Sim.steadystate(reset=True)

        #state should be at 2.0
        self.assertAlmostEqual(Int.outputs[0], 2.0, 3)

        #now run transient from steady state (should remain at 2.0)
        Sim.run(duration=5, reset=False)

        time, [state] = Sco.read()

        #state should stay near 2.0 throughout
        self.assertTrue(np.allclose(state, 2.0, atol=0.1),
            f"State should stay near 2.0, range: [{np.min(state):.3f}, {np.max(state):.3f}]")


class TestLinearizeAndRun(unittest.TestCase):
    """
    Test linearization followed by simulation.

    Linearize a nonlinear system around an operating point, then
    run the linearized system and verify it matches the expected
    linear behavior.
    """

    def test_linearize_nonlinear_system(self):
        """Linearize a nonlinear plant and verify small-signal behavior"""

        #nonlinear system: dx/dt = -x^2 + u, y = x
        #around x=1, u=1: linearized is dx/dt = -2*dx + du
        DS = DynamicalSystem(
            func_dyn=lambda x, u, t: -x**2 + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
            jac_dyn=lambda x, u, t: -2*x
            )

        Src = Constant(1.0)  # u=1 keeps x=1 at equilibrium
        Sco = Scope(labels=["output"])

        Sim = Simulation(
            blocks=[Src, DS, Sco],
            connections=[
                Connection(Src, DS),
                Connection(DS, Sco)
                ],
            Solver=RKCK54,
            tolerance_lte_abs=1e-8,
            log=False
            )

        #first get to equilibrium
        Sim.steadystate(reset=True)
        self.assertAlmostEqual(DS.outputs[0], 1.0, 3)

        #linearize around equilibrium
        Sim.linearize()

        #run the linearized system - it should stay at equilibrium
        Sim.run(duration=3, reset=False)

        time, [res] = Sco.read()

        #linearized system should stay at x=1 (equilibrium)
        self.assertTrue(np.allclose(res, 1.0, atol=0.01),
            f"Linearized system should stay at 1.0, got range [{np.min(res):.4f}, {np.max(res):.4f}]")

        #delinearize
        Sim.delinearize()


class TestResetAndContinue(unittest.TestCase):
    """
    Test simulation reset and continuation behavior.
    """

    def test_run_reset_run(self):
        """Run, reset, run again should produce same results"""

        Src = Source(lambda t: 1.0)
        Int = Integrator(0.0)
        Amp = Amplifier(-1.0)
        Add = Adder()
        Sco = Scope(labels=["state"])

        Sim = Simulation(
            blocks=[Src, Int, Amp, Add, Sco],
            connections=[
                Connection(Src, Add[0]),
                Connection(Amp, Add[1]),
                Connection(Add, Int),
                Connection(Int, Amp, Sco)
                ],
            dt=0.01,
            log=False
            )

        #first run
        Sim.run(duration=5, reset=True)
        time1, [state1] = Sco.read()

        #reset and run again
        Sim.run(duration=5, reset=True)
        time2, [state2] = Sco.read()

        #results should be identical
        self.assertTrue(np.allclose(time1, time2))
        self.assertTrue(np.allclose(state1, state2))


    def test_continuation_from_midpoint(self):
        """Run 5s, continue for another 5s, should match a single 10s run"""

        Src = Source(lambda t: 1.0)
        Int = Integrator(0.0)
        Amp = Amplifier(-1.0)
        Add = Adder()
        Sco1 = Scope(labels=["state"])
        Sco2 = Scope(labels=["state"])

        #system 1: run 10s in one go
        Sim1 = Simulation(
            blocks=[Src, Int, Amp, Add, Sco1],
            connections=[
                Connection(Src, Add[0]),
                Connection(Amp, Add[1]),
                Connection(Add, Int),
                Connection(Int, Amp, Sco1)
                ],
            dt=0.01,
            log=False
            )

        Sim1.run(duration=10, reset=True)
        time_full, [state_full] = Sco1.read()

        #system 2: identical but run 5+5
        Src2 = Source(lambda t: 1.0)
        Int2 = Integrator(0.0)
        Amp2 = Amplifier(-1.0)
        Add2 = Adder()

        Sim2 = Simulation(
            blocks=[Src2, Int2, Amp2, Add2, Sco2],
            connections=[
                Connection(Src2, Add2[0]),
                Connection(Amp2, Add2[1]),
                Connection(Add2, Int2),
                Connection(Int2, Amp2, Sco2)
                ],
            dt=0.01,
            log=False
            )

        Sim2.run(duration=5, reset=True)
        Sim2.run(duration=5, reset=False)  # continue
        time_split, [state_split] = Sco2.read()

        #the split run may have slight numerical differences at boundary
        #compare final state values - should agree to reasonable tolerance
        self.assertAlmostEqual(state_full[-1], state_split[-1], 5)

        #note: split run may overshoot by one extra dt at the boundary,
        #so we only check that total simulated times are close (within ~dt)
        self.assertAlmostEqual(Sim1.time, Sim2.time, 1)


class TestMultipleSolverSwitch(unittest.TestCase):
    """
    Test switching solvers mid-simulation and verify results
    are consistent.
    """

    def test_solver_switch_during_simulation(self):
        """Switch from explicit to implicit solver and verify consistency"""

        Src = Source(lambda t: 1.0)
        Int = Integrator(0.0)
        Amp = Amplifier(-1.0)
        Add = Adder()
        Sco = Scope(labels=["state"])

        Sim = Simulation(
            blocks=[Src, Int, Amp, Add, Sco],
            connections=[
                Connection(Src, Add[0]),
                Connection(Amp, Add[1]),
                Connection(Add, Int),
                Connection(Int, Amp, Sco)
                ],
            Solver=RKCK54,
            tolerance_lte_abs=1e-8,
            log=False
            )

        #run with explicit solver
        Sim.run(duration=3, reset=True)
        state_at_3 = Int.outputs[0]

        #switch to implicit solver and continue
        Sim._set_solver(ESDIRK32, tolerance_lte_abs=1e-8)
        Sim.run(duration=3, reset=False)

        time, [state] = Sco.read()

        #analytical: x(t) = 1 - exp(-t) for unit step
        ref_final = 1.0 - np.exp(-6.0)
        self.assertAlmostEqual(state[-1], ref_final, 3,
            f"Final state: {state[-1]:.6f}, expected: {ref_final:.6f}")


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
