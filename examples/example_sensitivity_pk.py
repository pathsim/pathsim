#########################################################################################
##
##      PathSim example: sensitivity & identifiability — 1-compartment PK model
##
##  Model:   1-compartment oral pharmacokinetics
##
##      a'(t) = -ka * a                    (drug absorbed from GI tract)
##      c'(t) =  ka * a / V  - ke * c      (drug concentration in blood)
##
##  Dose D [mg] is a known input.  Three parameters to identify:
##
##      ka [1/h]   absorption rate constant
##      ke [1/h]   elimination rate constant
##      V  [L]     apparent volume of distribution
##
##  Synthetic measurements of c(t) are generated from the analytical solution
##  with 5% Gaussian noise, then fitted and analysed.
##
##  Sensitivity analysis reveals:
##   - ke is the most precisely identified  (terminal slope is well-sampled)
##   - V  is identified from the peak magnitude
##   - ka is the least precise             (only the rise phase pins it down)
##   - ka and ke carry moderate correlation (adjusting one shifts the peak)
##
##  This example shows how sensitivity analysis can be done in PathSim.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope
from pathsim.solvers import SSPRK22

from pathsim.opt import Parameter, ParameterEstimator, TimeSeriesData


# TRUE PARAMETER VALUES =================================================================

D       = 100.0   # dose [mg]  — known
TRUE_KA = 1.5     # absorption rate [1/h]
TRUE_KE = 0.12    # elimination rate [1/h]
TRUE_V  = 12.0    # volume of distribution [L]


# FREE PARAMETERS (estimator will update these via closure) =============================

ka = Parameter("ka", value=1.0, bounds=(0.1, 10.0))
ke = Parameter("ke", value=0.05, bounds=(0.01,  1.0))
V  = Parameter("V",  value=8.0,  bounds=(1.0,  50.0))


# ODE RIGHT-HAND SIDE ===================================================================

def pk_odes(x, u, t):
    """1-compartment oral absorption ODE.

    State vector x = [a, c]:
      a  amount of drug in GI tract [mg]
      c  drug concentration in blood [mg/L]
    """
    a, c = x
    da_dt = -ka() * a
    dc_dt =  ka() * a / V() - ke() * c
    return np.array([da_dt, dc_dt])


# MODEL DEFINITION ======================================================================

ode   = ODE(func=pk_odes, initial_value=[D, 0.0])
scope = Scope(labels=["c(t)"])
blocks = [ode, scope]
connections = [Connection(ode[1], scope[0])]

sim = Simulation(
    blocks=blocks,
    connections=connections,
    Solver=SSPRK22,
    dt=0.05, dt_min=1e-10,
    tolerance_lte_rel=1e-5,
    tolerance_lte_abs=1e-8,
    log=False,
)


# SYNTHETIC MEASUREMENTS ================================================================

def analytical_c(t, ka_, ke_, V_):
    """Exact blood concentration for 1-compartment oral PK."""
    return D * ka_ / (V_ * (ka_ - ke_)) * (np.exp(-ke_ * t) - np.exp(-ka_ * t))


# Measurement schedule: dense early (catching the peak) and sparse late (terminal slope)
t_meas = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   8.0, 10.0, 12.0, 16.0, 20.0, 24.0])

rng    = np.random.default_rng(42)
c_true = analytical_c(t_meas, TRUE_KA, TRUE_KE, TRUE_V)
c_meas = c_true * (1.0 + 0.05 * rng.standard_normal(len(t_meas)))
c_meas = np.maximum(c_meas, 0.0)   # concentrations are non-negative


# Run Example ===========================================================================

if __name__ == '__main__':

    meas = TimeSeriesData(time=t_meas, data=c_meas, name="c(t) [mg/L]")

    est = ParameterEstimator(simulator=sim, adaptive=True)
    est.add_parameters([ka, ke, V])
    est.add_timeseries(meas, signal=scope[0], sigma=0.5)

    # ── Fit ──────────────────────────────────────────────────────────────────
    print("Fitting PK parameters ...")
    fit = est.fit(loss="soft_l1", f_scale=0.5, max_nfev=120, verbose=0)
    est.display()

    # ── Sensitivity & identifiability ─────────────────────────────────────────
    #
    # sensitivity() builds the Fisher Information Matrix from the Jacobian
    # of the normalised residuals, then reports:
    #
    #   std_error   — how precisely each parameter is constrained by the data
    #   rel_error   — std_error relative to the fitted value
    #   condition # — λ_max / λ_min of the FIM (high → some direction not identified)
    #   correlation — how much the parameters trade off against each other
    #
    sens = est.sensitivity()
    sens.display()

    # True values for comparison
    print(f"\n  True values:  ka={TRUE_KA}  ke={TRUE_KE}  V={TRUE_V}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. Fit quality
    fig_fit, _ = est.plot_fit(
        fit.x,
        title="1-compartment PK fit",
        xlabel="Time [h]",
        ylabel="Concentration [mg/L]",
    )

    # 2. Sensitivity analysis
    fig_sens, _ = sens.plot()

    plt.show()
