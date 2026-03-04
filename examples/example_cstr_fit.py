#########################################################################################
##
##          PathSim example: parameter estimation for a CSTR with consecutive reactions
##
##  Model:   A → B → C  (consecutive reactions, temperature-dependent Arrhenius kinetics)
##  States:  [Ca, Cb, Cc, T]  concentrations [mol/L] and temperature [K]
##  Inputs:  [Ca_in, T_in]    feed concentration and feed temperature
##  Fit:     pre-exponential factors k1_0 and k2_0 in log10 space
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import exp10

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, ODE, Scope, Source
from pathsim.solvers import SSPRK22

from pathsim.opt import Parameter, ParameterEstimator, TimeSeriesData


# SYSTEM PARAMETERS =====================================================================

Ca_0 = 1.0    # Initial concentration of A  [mol/L]
Cb_0 = 0.0    # Initial concentration of B  [mol/L]
T_0  = 300.0  # Initial temperature         [K]

Tc  = 280.0   # Coolant temperature          [K]
tau = 1.0     # Residence time               [s]
E1  = 5e4     # Activation energy, rxn 1    [J/mol]
E2  = 5.5e4   # Activation energy, rxn 2    [J/mol]
dH1 = -5e4    # Reaction enthalpy, rxn 1    [J/mol]
dH2 = -5.2e4  # Reaction enthalpy, rxn 2    [J/mol]
rho = 1000.0  # Density                     [kg/m³]
Cp  = 4.184   # Heat capacity               [J/(g·K)]
U   = 1000.0  # Heat transfer coefficient   [W/(m²·K)]
V   = 0.1     # Reactor volume              [m³]
A   = 0.1     # Heat transfer area          [m²]
R   = 8.314   # Gas constant                [J/(mol·K)]

# Parameters to estimate (optimizer works in log10 space via exp10 transform)
k1_0 = Parameter("k1_0", value=8.0,  bounds=(2, 10), transform=exp10)
k2_0 = Parameter("k2_0", value=8.5,  bounds=(2, 11), transform=exp10)


# ODE RIGHT-HAND SIDE ===================================================================

def reaction_rates(x, u, t):
    """CSTR dynamics with consecutive reactions A → B → C.

    Parameters
    ----------
    x : array [Ca, Cb, Cc, T]
        State variables.
    u : array [Ca_in, T_in]
        Inputs: feed concentration and feed temperature.
    t : float
        Time.

    Returns
    -------
    np.ndarray
        Time derivatives of the state vector.
    """
    Ca, Cb, Cc, T = x
    Ca_in, T_in = u

    # Temperature-dependent Arrhenius rate constants
    k1 = k1_0() * np.exp(-E1 / (R * T))
    k2 = k2_0() * np.exp(-E2 / (R * T))

    # Concentration balances
    dCa_dt = (Ca_in - Ca) / tau - k1 * Ca
    dCb_dt = -Cb / tau + k1 * Ca - k2 * Cb
    dCc_dt = -Cc / tau + k2 * Cb

    # Energy balance
    Q_rxn1    = (-dH1 / (rho * Cp)) * k1 * Ca
    Q_rxn2    = (-dH2 / (rho * Cp)) * k2 * Cb
    Q_cooling = U * A * (T - Tc) / (V * rho * Cp)
    dT_dt = (T_in - T) / tau + Q_rxn1 + Q_rxn2 - Q_cooling

    return np.array([dCa_dt, dCb_dt, dCc_dt, dT_dt])


# MODEL DEFINITION ======================================================================

# Sources
Ca_in_source = Source(func=lambda t: 2.0 + 0.0 * np.sin(0.5 * t))
T_in_source  = Source(func=lambda t: 280.0 * (1 - 0.8 * np.exp(-0.6 * t)))

# Dynamic ODE block
ode = ODE(
    func=reaction_rates,
    initial_value=[Ca_0, Cb_0, 0, T_0],
)

# Scale temperature for recording  (T / 100 so it fits the same axis as concentrations)
T_scaler = Amplifier(gain=1 / 100)

# Recording
scope = Scope(labels=['Ca', 'Cb', 'Cc', 'T/100'])

blocks = [
    Ca_in_source,
    T_in_source,
    ode,
    T_scaler,
    scope,
]

connections = [
    Connection(Ca_in_source[0], ode[0]),
    Connection(T_in_source[0],  ode[1]),
    Connection(ode[0], scope[0]),
    Connection(ode[1], scope[1]),
    Connection(ode[2], scope[2]),
    Connection(ode[3], T_scaler[0]),
    Connection(T_scaler[0], scope[3]),
]

sim = Simulation(
    blocks,
    connections,
    Solver=SSPRK22,
    dt=0.01,
    dt_min=1e-16,
    tolerance_lte_rel=1e-5,
    tolerance_lte_abs=1e-8,
    tolerance_fpi=1e-10,
    log=False,
)


# Run Example ===========================================================================

if __name__ == '__main__':

    # Initial run to verify model behaviour
    sim.run(duration=100)

    # Create the estimator
    est = ParameterEstimator(simulator=sim, adaptive=True)

    # Register the free parameters (k1_0, k2_0 live in the ODE closure)
    est.add_parameters([k1_0, k2_0])

    # Sparse measurement data for Cc (concentration of product C)
    t_meas = np.r_[0.5, 2, 5, 10, 20, 50, 80]
    y_meas = np.r_[0,   0, 0.01, 0.5, 0.95, 0.95, 0.95]

    meas = TimeSeriesData(time=t_meas, data=y_meas, name="Cc [mol/L]")
    est.add_timeseries(meas, signal=scope[2], sigma=1.0)

    # Fit
    fit = est.fit(loss='soft_l1', max_nfev=80, verbose=2)

    est.display()

    fig, axes = est.plot_fit(
        fit.x,
        title="CSTR parameter fit — concentration of C",
        xlabel="Time [s]",
        ylabel="Cc [mol/L]",
    )
    plt.show()
