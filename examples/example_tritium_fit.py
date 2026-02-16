
#########################################################################################
##
##        PathSim example of parameter estimation for a tritium reactor model
##                       This example was built using PathView 
##
#########################################################################################

# ────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Adder,
    Amplifier,
    Constant,
    Integrator,
    Multiplier,
    Scope,
    StepSource
)
from pathsim.solvers import SSPRK22

# ────────────────────────────────────────────────────────────────────────────
# BLOCKS
# ────────────────────────────────────────────────────────────────────────────

# Sources
p383 = Constant(
    value=4.95e8/2
)
a325 = Constant(
    value=2.13e8/2
)
tbr = Constant(
    value=5.4e-4
)
k_wall = Constant(
    value=1.2
)
k_top = Constant(
    value=4
)
stepsource = StepSource()
block_6 = StepSource(
    amplitude=-1,
    tau=12*3600
)
block_7 = StepSource(
    tau=24*3600
)
block_8 = StepSource(
    tau=36*3600,
    amplitude=-1
)
a_top = Constant(
    value=1e1
)
a_wall = Constant(
    value=1e2
)

# Dynamic
integrator = Integrator()
block_12 = Integrator()

# Algebraic
adder = Adder(
    operations="++"
)
block_14 = Adder(
    operations="+--"
)
s = Multiplier()
block_16 = Adder(
    operations="++"
)
block_17 = Adder(
    operations="++"
)
irradiations = Adder(
    operations="++"
)
multiplier = Multiplier()
block_20 = Multiplier()
block_21 = Multiplier()
amplifier = Amplifier(
    gain=1e-6
)
block_23 = Amplifier(
    gain=1e-8
)
block_24 = Amplifier(
    gain=1e-9
)

# Recording
result = Scope()
q_int = Scope()

blocks = [
    p383,
    a325,
    tbr,
    k_wall,
    k_top,
    stepsource,
    block_6,
    block_7,
    block_8,
    a_top,
    a_wall,
    integrator,
    block_12,
    adder,
    block_14,
    s,
    block_16,
    block_17,
    irradiations,
    multiplier,
    block_20,
    block_21,
    amplifier,
    block_23,
    block_24,
    result,
    q_int,
]

# ────────────────────────────────────────────────────────────────────────────
# CONNECTIONS
# ────────────────────────────────────────────────────────────────────────────

connections = [
    Connection(p383[0], adder[0]),
    Connection(a325[0], adder[1]),
    Connection(tbr[0], s[0]),
    Connection(adder[0], s[1]),
    Connection(stepsource[0], block_16[0]),
    Connection(block_6[0], block_16[1]),
    Connection(block_7[0], block_17[0]),
    Connection(block_8[0], block_17[1]),
    Connection(block_17[0], irradiations[0]),
    Connection(block_16[0], irradiations[1]),
    Connection(a_top[0], multiplier[0]),
    Connection(block_14[0], integrator[0]),
    Connection(integrator[0], multiplier[2], result[0], block_21[2]),
    Connection(multiplier[0], block_14[1], result[1], block_12[0]),
    Connection(irradiations[0], block_20[0]),
    Connection(s[0], block_20[1]),
    Connection(block_20[0], block_14[0]),
    Connection(a_wall[0], block_21[0]),
    Connection(block_21[0], block_14[2], block_12[1]),
    Connection(k_top[0], amplifier[0]),
    Connection(amplifier[0], multiplier[1]),
    Connection(k_wall[0], block_23[0]),
    Connection(block_23[0], block_21[1]),
    Connection(block_12[0], block_24[0]),
    Connection(block_12[1], block_24[1]),
    Connection(block_24[0], q_int[0]),
    Connection(block_24[1], q_int[1]),
]

# ────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ────────────────────────────────────────────────────────────────────────────

sim = Simulation(
    blocks,
    connections,
    Solver=SSPRK22,
    dt=100,
    dt_min=1e-16,
    tolerance_lte_rel=0.0001,
    tolerance_lte_abs=1e-08,
    tolerance_fpi=1e-10,
    log=False
)

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Parameter estimation imports
    from pathsim.opt import ParameterEstimator, TimeSeriesData
    from scipy.special import exp10

    # Run simulation
    sim.run(duration=7*24*3600)

    # Plot results
    # sim.plot()
    # plt.show()
    
    # Some data for parameter estimation testing
    t_meas = np.r_[0.5, 1, 2, 3, 4, 5, 6] * 24*3600
    y_meas = np.r_[4, 11, 17, 24, 24, 24, 24] / 2 * 1 +(np.random.random(7) - 0.5)
    
    # create TimeSeriesData explicitly
    meas = TimeSeriesData(time=t_meas, data=y_meas, name="y_meas")

    # Create the estimator - this needs to be done first - then add the
    est = ParameterEstimator(
        simulator=sim,
        adaptive=True,
    )

    # DAdd block parameters to estimate
    est.add_block_parameter(k_wall, 'value', id='k_wall', transform=exp10, value=0.2, bounds=(-1, 1))
    est.add_block_parameter(k_top, 'value', id='k_top', transform=exp10, value=0.3, bounds=(-1, 1))

    print(est.parameters)

    # register measurement + model output mapping
    est.add_timeseries(meas, signal=q_int[0], sigma=1.0)

    # Fit (x0 and bounds extracted from Parameters automatically)
    fit = est.fit(loss='soft_l1', max_nfev=80, verbose=2)

    # Plot
    sec2day = 1/(24*3600)
    t_pred, y_pred = est.simulate(fit.x)

    plt.figure(figsize=(8, 5))
    plt.plot(t_meas * sec2day, y_meas, 'o', ms=5, alpha=0.6, label='Measured')
    plt.plot(t_pred * sec2day, y_pred, '-', lw=2, label=f'Fit')
    plt.xlabel('Time [d]')
    plt.ylabel('Output')
    plt.title('Parameter Estimation with Parameter Objects')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    est.display()

    fig, axes = est.plot_fit(
        fit.x,
        overlay=True,
        title="Fit (overlayed experiments)",
        xlabel="Time [s]",
        ylabel="Output",
    )
    plt.show()
