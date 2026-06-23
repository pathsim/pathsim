#########################################################################################
##
##              PathSim example: parameter estimation from time series data
##
##  Model:   y(t) = gain * t + offset  (integrator with initial_value as offset)
##  Data:    Two synthetic datasets with different offsets but the same slope.
##  Fit:     gain (global, shared across experiments)
##           initial_value (local, one per experiment)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source,
    Constant,
    Multiplier,
    Integrator,
    Scope,
    Adder,
    TimeSeriesSource,
)
from pathsim.solvers import SSPRK22

from pathsim.opt import ParameterEstimator, TimeSeriesData


# DATA ==================================================================================

t_meas = np.linspace(0, 10, 21)
y_meas = np.random.uniform(0.95, 1.05, 21) * t_meas

t_meas2 = np.linspace(0, 10, 41)
y_meas2 = 5 + np.random.uniform(0.9, 1.1, 41) * t_meas2


# MODEL DEFINITION ======================================================================

source = Source(lambda t: 1.0)
gain = Constant(value=0.5)
mult = Multiplier()
integrator = Integrator()
adder = Adder()
scope = Scope()

# Optional: keep the measured signal in the sim for visual reference
tsSource = TimeSeriesSource(t=t_meas, y=y_meas)

blocks = [
    source,
    gain,
    mult,
    integrator,
    adder,
    scope,
    tsSource,
]

connections = [
    Connection(source[0], mult[0]),
    Connection(gain[0], mult[1]),
    Connection(mult[0], integrator[0]),
    Connection(integrator[0], adder[0]),
    Connection(adder[0], scope[0]),
    Connection(tsSource[0], scope[1]),
]

sim = Simulation(
    blocks,
    connections,
    Solver=SSPRK22,
    dt=0.01,
    dt_min=1e-16,
    tolerance_lte_rel=1e-4,
    tolerance_lte_abs=1e-8,
    tolerance_fpi=1e-10,
    log=False,
)


# Run Example ===========================================================================

if __name__ == '__main__':

    # Initial run (optional; the estimator resets and reruns each evaluation)
    sim.run(duration=10.0)

    # Create the estimator; experiment 0 is registered automatically
    est = ParameterEstimator(simulator=sim, adaptive=True)

    # Register experiment 1 as a deep copy so both have independent state
    est.add_experiment(sim, adaptive=True, copy_sim=True)

    # Global parameter: shared gain across both experiments
    est.add_global_block_parameter('Constant', 'value', param_id='gain', value=3.0)

    # Local parameters: one initial offset per experiment
    est.add_local_block_parameter(
        0, 'Integrator', 'initial_value',
        param_id='offset', value=0.5, bounds=(0.0, 5.0),
    )
    est.add_local_block_parameter(
        1, 'Integrator', 'initial_value',
        param_id='offset', value=4.0, bounds=(0.0, 5.0),
    )

    # Wrap measurements; pass scope[0] from experiment 0 — the estimator
    # resolves the matching deep-copied scope for experiment 1 automatically
    meas  = TimeSeriesData(time=t_meas,  data=y_meas,  name="exp0")
    meas2 = TimeSeriesData(time=t_meas2, data=y_meas2, name="exp1")

    est.add_timeseries(meas,  signal=scope[0], sigma=1.0, experiment=0)
    est.add_timeseries(meas2, signal=scope[0], sigma=1.0, experiment=1)

    # Fit
    fit = est.fit(loss='soft_l1', max_nfev=80, verbose=2)

    est.display()

    # Plot all experiments overlaid on a single axis
    fig, axes = est.plot_fit(
        fit.x,
        overlay=True,
        title="Linear fit — global gain + local offsets",
        xlabel="Time [s]",
        ylabel="Output",
    )
    plt.show()
