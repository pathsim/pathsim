#########################################################################################
##
##               PathSim example: hello-world parameter estimation
##
##  Model:   y(t) = gain * t   (ramp output)
##  Fit:     Amplifier gain from noisy measurements
##
##  This is the simplest possible single-experiment, single-parameter fit.
##  Start here before looking at the more complex examples.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Amplifier, Scope
from pathsim.solvers import SSPRK22

from pathsim.opt import ParameterEstimator, TimeSeriesData


# MODEL DEFINITION ======================================================================

# Source outputs t at each timestep; Amplifier scales it by `gain`
source = Source(func=lambda t: t)
amp    = Amplifier(gain=1.0)       # gain is the parameter we want to estimate
scope  = Scope()

blocks = [source, amp, scope]

connections = [
    Connection(source[0], amp[0]),
    Connection(amp[0],    scope[0]),
]

sim = Simulation(
    blocks,
    connections,
    Solver=SSPRK22,
    dt=0.1,
    log=False,
)


# Run Example ===========================================================================

if __name__ == '__main__':

    # Synthetic noisy measurements: true gain = 3.0
    true_gain = 3.0
    t_meas = np.linspace(0.5, 10.0, 20)
    y_meas = true_gain * t_meas + 0.15 * np.random.randn(20)

    meas = TimeSeriesData(time=t_meas, data=y_meas, name="gain * t")

    # Create estimator and register the Amplifier's gain as a free parameter
    est = ParameterEstimator(simulator=sim)
    est.add_block_parameter(amp, "gain", value=1.0, bounds=(0.0, 10.0))
    est.add_timeseries(meas, signal=scope[0], sigma=0.15)

    # Fit
    result = est.fit(loss="soft_l1", f_scale=0.15, max_nfev=50, verbose=2)

    est.display()

    fig, axes = est.plot_fit(
        result.x,
        title="Hello-world fit — Amplifier gain",
        xlabel="Time [s]",
        ylabel="Output",
    )
    plt.show()
