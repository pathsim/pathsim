#########################################################################################
##
##     PathSim example of parameter estimation for a CSTR with consecutive reactions
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Amplifier,
    ODE,
    Scope,
    Source
)
from pathsim.opt import Parameter, ParameterEstimator, TimeSeriesData
from pathsim.solvers import SSPRK22


# ────────────────────────────────────────────────────────────────────────────
# USER-DEFINED CODE
# ────────────────────────────────────────────────────────────────────────────

Ca_0 = 1.0    # Initial concentration of A [mol/L]
Cb_0 = 0.0    # Initial concentration of B [mol/L]
T_0 = 300.0   # Initial temperature [K]

# System parameters
Tc = 280.0    # Coolant temperature [K]
tau = 1.0     # Residence time [s]
# k1_0 = 1e8    # Pre-exponential factor 1 [1/s]
# k2_0 = 6e9   # Pre-exponential factor 2 [1/s]
E1 = 5e4      # Activation energy 1 [J/mol]
E2 = 5.5e4    # Activation energy 2 [J/mol]
dH1 = -5e4    # Reaction enthalpy 1 [J/mol]
dH2 = -5.2e4  # Reaction enthalpy 2 [J/mol]
rho = 1000.0  # Density [kg/m³]
Cp = 4.184    # Heat capacity [J/(g·K)]
U = 1000.0    # Heat transfer coefficient [W/(m²·K)]
V = 0.1       # Reactor volume [m³]
A = 0.1       # Heat transfer area [m²]
R = 8.314     # Gas constant [J/(mol·K)]

# System parameters to estimate
k1_0 = Parameter("k1_0", value=1e8, bounds=(1e2, 1e10))
k2_0 = Parameter("k2_0", value=6e9, bounds=(1e2, 1e10))


def reaction_rates(x, u, t):
    """CSTR dynamics with consecutive reactions
    
    Parameters
    ----------
    x : array [Ca, Cb, T]
        State variables
    u : array [Ca_in, T_in]
        Input variables
    t : float
        Time
    
    Returns
    -------
    dx_dt : array
        Time derivatives
    """
    # Unpack states
    Ca, Cb, Cc, T = x
    
    # Unpack inputs
    Ca_in, T_in = u
    
    # Reaction rate constants (Arrhenius) - note the dependence on the parameters to estimate
    k1 = k1_0() * np.exp(-E1/(R*T))
    k2 = k2_0() * np.exp(-E2/(R*T))
    
    # Concentration dynamics
    dCa_dt = (Ca_in - Ca)/tau - k1*Ca
    dCb_dt = -Cb/tau + k1*Ca - k2*Cb
    dCc_dt = -Cc/tau + k2*Cb
    
    # Temperature dynamics
    Q_reaction1 = (-dH1/(rho*Cp)) * k1 * Ca
    Q_reaction2 = (-dH2/(rho*Cp)) * k2 * Cb
    Q_cooling = U*A*(T - Tc)/(V*rho*Cp)
    
    dT_dt = (T_in - T)/tau + Q_reaction1 + Q_reaction2 - Q_cooling
    
    return np.array([dCa_dt, dCb_dt, dCc_dt, dT_dt])


# MODEL DEFINITION =====================================================================

# Sources
source = Source(
    func=lambda t: 2.0 + 0.0*np.sin(0.5*t)
)
block_1 = Source(
    func=lambda t: 280.0 * (1 - 0.8 * np.exp(-0.6*t))
)

# Dynamic
ode = ODE(
    func=reaction_rates,
    initial_value=[Ca_0, Cb_0, 0, T_0]
)

# Algebraic
amplifier = Amplifier(
    gain=1/100
)

# Recording
scope = Scope(
    labels=['Ca', 'Cb', 'Cc', 'T']
)

blocks = [
    source,
    block_1,
    ode,
    amplifier,
    scope,
]

#the connections between the blocks
connections = [
    Connection(ode[3], amplifier[0]),
    Connection(amplifier[0], scope[3]),
    Connection(ode[2], scope[2]),
    Connection(ode[1], scope[1]),
    Connection(ode[0], scope[0]),
    Connection(source[0], ode[0]),
    Connection(block_1[0], ode[1]),
]

# initialize simulation with the blocks, connections, timestep and logging enabled
sim = Simulation(
    blocks,
    connections,
    Solver=SSPRK22,
    dt=0.01,
    dt_min=1e-16,
    tolerance_lte_rel=1e-05,
    tolerance_lte_abs=1e-08,
    tolerance_fpi=1e-10,
    log=False
)

# Run Example ===========================================================================

if __name__ == '__main__':

    # run simulation
    sim.run(duration=100)

    # sim.plot()
    
    # Create parameter estimator instance
    est = ParameterEstimator(
        simulator=sim,
        adaptive=True,
    )

    # Adds parameters to be optimized - here the pre-exponential factors of the two reactions
    est.add_parameters([k1_0, k2_0])

    # Add some test data
    t_meas = np.r_[0.5, 2, 5, 10, 20, 50, 80]
    y_meas = np.r_[0, 0, 0.01, 0.5, 0.95, 0.95, 0.95]

    # create TimeSeriesData explicitly
    meas = TimeSeriesData(time=t_meas, data=y_meas, name="y_meas")

    # register measurement + model output mapping
    est.add_timeseries(meas, signal=scope[2], sigma=1.0)

    # run the fitting routine
    fit = est.fit(loss='soft_l1', max_nfev=80, verbose=2)

    # Display the results
    est.display()

    # Plot fit results
    fig, axes = est.plot_fit(
        fit.x,
        title="CSTR Fit",
        xlabel="Time [s]",
        ylabel="Concentration of C [mol/L]",
    )
    plt.show()