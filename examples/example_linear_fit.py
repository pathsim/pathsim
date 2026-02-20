#########################################################################################
##
##            PathSim example of parameter estimation using time series data
##
##  Model:   y(t) = gain * t + offset  (integrator with initial value as offset)
##  Data:    Two synthetic datasets with different offsets but the same slope.
##  Fit:     gain (global, shared) + initial_value (local, per-experiment).
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


# DATA ==================================================================================

t_meas = np.linspace(0, 10, 21)
y_meas = np.random.uniform(0.95, 1.05, 21) * t_meas

t_meas2 = np.linspace(0, 10, 41)
y_meas2 = 5 + np.random.uniform(0.9, 1.1, 41) * t_meas2

# MODEL DEFINITION ======================================================================

# One simulation model ------------------------------------------------------------------
source = Source(lambda t: 1.0)
gain = Constant(value=0.5)
mult = Multiplier()
integrator = Integrator()
adder = Adder()
scope = Scope()

# (Optional) keep timeseries sources in the sim as reference/visualization signals
# (not required for fitting, but convenient for plotting / sanity checks)
tsSource = TimeSeriesSource(t=t_meas, y=y_meas)
# tsSource2 = TimeSeriesSource(t=t_meas2, y=y_meas2)

blocks = [
    source,
    gain,
    mult,
    integrator,
    adder,
    scope,
    tsSource,
    # tsSource2,
]

connections = [
    Connection(source[0], mult[0]),
    Connection(gain[0], mult[1]),
    Connection(mult[0], integrator[0]),
    Connection(integrator[0], adder[0]),
    Connection(adder[0], scope[0]),
    Connection(tsSource[0], scope[1]),
    # Connection(tsSource2[0], scope[2]),
]

sim = Simulation(
    blocks,
    connections,
    Solver=SSPRK22,
    dt=0.01,
    dt_min=1e-16,
    tolerance_lte_rel=0.0001,
    tolerance_lte_abs=1e-08,
    tolerance_fpi=1e-10,
    log=False,
)


# Run Example ===========================================================================

if __name__ == '__main__':

    # Parameter estimation imports
    from pathsim.opt import ParameterEstimator, TimeSeriesData

    # Trigger initial run (optional; estimator will reset+run each evaluation)
    sim.run(duration=10.0)

    # create parameter estimator instance
    est = ParameterEstimator(
        simulator=sim,
        adaptive=True,
    )

    # Ensure experiment 1 exists as a deepcopy of experiment 0
    est.add_experiment(sim, adaptive=True, copy_sim=True)

    # Global parameter (shared across experiments)
    est.add_global_block_parameter('Constant', 'value', param_id='gain', value=3)

    # Local parameters (one per experiment)
    est.add_local_block_parameter(0, 'Integrator', 'initial_value', param_id='integrator', bounds=(0.0, 5), value=0.5)
    est.add_local_block_parameter(1, 'Integrator', 'initial_value', param_id='integrator', bounds=(0.0, 5), value=4.0)

    print(est.parameters)

    # create TimeSeriesData explicitly
    meas = TimeSeriesData(time=t_meas, data=y_meas, name="y_meas")
    meas2 = TimeSeriesData(time=t_meas2, data=y_meas2, name="y_meas2")

    # register measurement + model output mapping (each dataset uses its own experiment)
    # IMPORTANT: pass the scope from experiment 0; estimator resolves the corresponding
    # deep-copied scope for other experiments automatically.
    est.add_timeseries(meas, signal=scope[0], sigma=1.0, experiment=0)
    est.add_timeseries(meas2, signal=scope[0], sigma=1.0, experiment=1)

    # run the fitting routine
    fit = est.fit(loss='soft_l1', max_nfev=80, verbose=2)

    # Display the results
    est.display()

    # This will plot the fit for each experiment separately; 
    # since we have a global parameter, the fits will be the same across experiments, 
    # but this is just to illustrate the API. 
    # You can also overlay them (see below).

    # fig, axes = est.plot_fit(
    #     fit.x,
    #     title="Parameter Estimation with Global + Local Parameters",
    #     xlabel="Time [s]",
    #     ylabel="Output",
    # )
    # plt.show()

    fig, axes = est.plot_fit(
        fit.x,
        overlay=True,
        title="Fit (overlayed experiments)",
        xlabel="Time [s]",
        ylabel="Output",
    )
    plt.show()