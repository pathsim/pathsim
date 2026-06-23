#########################################################################################
##
##      PathSim example: CSTR parameter estimation with TimeSeriesSource inputs
##
##  Same consecutive-reaction CSTR as example_cstr_fit.py, but the feed
##  concentration Ca_in and feed temperature T_in are supplied as pre-recorded
##  time-series data through TimeSeriesSource blocks rather than analytical
##  Source functions.  This mirrors a real workflow where the input profile
##  comes from a sensor log or a design-of-experiment file.
##
##  Model:   A → B → C  (consecutive Arrhenius kinetics)
##  States:  [Ca, Cb, Cc, T]
##  Inputs:  Ca_in(t), T_in(t)   — replayed from sampled data
##  Fit:     pre-exponential factors k1_0, k2_0  (log10 space)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import exp10

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, ODE, Scope, TimeSeriesSource
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

# True kinetic parameters — the estimator must recover these from data
TRUE_K1_0 = 1e9    # [1/s]
TRUE_K2_0 = 1e9    # [1/s]

# Parameters to estimate (optimizer works in log10 space via exp10 transform)
k1_0 = Parameter("k1_0", value=8.0, bounds=(2, 11), transform=exp10)
k2_0 = Parameter("k2_0", value=8.5, bounds=(2, 12), transform=exp10)


# FEED PROFILES =========================================================================
#
# Sampled at 0.5 s intervals up to t=100 s.  In a real application these
# would come from a sensor log.  Here we synthesise them:
#   Ca_in — step-up at t=10, step-back at t=60
#   T_in  — smooth ramp from 280 K → 320 K then held constant

t_feed = np.arange(0, 101, 1)

Ca_in_data = np.where(t_feed < 10, 1.0, np.where(t_feed < 60, 2.5, 1.0))
T_in_data  = 280.0 + 40.0 * (1 - np.exp(-0.08 * t_feed))

Ca_in_ts = TimeSeriesData(time=t_feed, data=Ca_in_data, name="Ca_in")
T_in_ts  = TimeSeriesData(time=t_feed, data=T_in_data,  name="T_in")


# ODE RIGHT-HAND SIDE ===================================================================

def reaction_rates(x, u, t):
    Ca, Cb, Cc, T = x
    Ca_in, T_in = u

    k1 = k1_0() * np.exp(-E1 / (R * T))
    k2 = k2_0() * np.exp(-E2 / (R * T))

    dCa_dt = (Ca_in - Ca) / tau - k1 * Ca
    dCb_dt = -Cb / tau + k1 * Ca - k2 * Cb
    dCc_dt = -Cc / tau + k2 * Cb

    Q_rxn1    = (-dH1 / (rho * Cp)) * k1 * Ca
    Q_rxn2    = (-dH2 / (rho * Cp)) * k2 * Cb
    Q_cooling = U * A * (T - Tc) / (V * rho * Cp)
    dT_dt = (T_in - T) / tau + Q_rxn1 + Q_rxn2 - Q_cooling

    return np.array([dCa_dt, dCb_dt, dCc_dt, dT_dt])


# MODEL DEFINITION ======================================================================
#
# TimeSeriesSource blocks replay the pre-recorded feed profiles.  They behave
# exactly like Source blocks at the simulation level — they output an
# interpolated value at the current simulation time each timestep.

Ca_in_src = TimeSeriesSource(ts=Ca_in_ts, extrapolate="hold")
T_in_src  = TimeSeriesSource(ts=T_in_ts,  extrapolate="hold")

ode = ODE(
    func=reaction_rates,
    initial_value=[Ca_0, Cb_0, 0, T_0],
)

T_scaler = Amplifier(gain=1/100)    # T/100 so it fits same axis as concentrations
scope    = Scope(labels=["Ca", "Cb", "Cc", "T/100"])

blocks = [Ca_in_src, T_in_src, ode, T_scaler, scope]
connections = [
        Connection(Ca_in_src[0], ode[0]),
        Connection(T_in_src[0],  ode[1]),
        Connection(ode[0], scope[0]),
        Connection(ode[1], scope[1]),
        Connection(ode[2], scope[2]),
        Connection(ode[3], T_scaler[0]),
        Connection(T_scaler[0], scope[3]),
    ]

sim = Simulation(
    blocks=blocks,
    connections=connections,
    Solver=SSPRK22,
    dt=0.01,
    dt_min=1e-16,
    tolerance_lte_rel=1e-5,
    tolerance_lte_abs=1e-8,
    log=False,
)


# RUN EXAMPLE ===========================================================================

if __name__ == '__main__':

    # ── Generate "true" data with the real k values ───────────────────────────────────
    k1_0.set(np.log10(TRUE_K1_0))
    k2_0.set(np.log10(TRUE_K2_0))

    sim.run(duration=100)
    t_sim, y_sim = scope.read()

    # Cb (index 1) and Cc (index 2) are the informative outputs — pick up at sparse times
    t_meas = np.array([1, 2, 5, 10, 20, 40, 60, 80, 100], dtype=float)
    rng    = np.random.default_rng(42)

    def noisy(y_row, sigma=0.03):
        return np.maximum(np.interp(t_meas, t_sim, y_row) * (1 + sigma * rng.standard_normal(len(t_meas))), 0)

    meas_Cb = TimeSeriesData(time=t_meas, data=noisy(y_sim[1]), name="Cb [mol/L]")
    meas_Cc = TimeSeriesData(time=t_meas, data=noisy(y_sim[2]), name="Cc [mol/L]")

    print(f"True:  log10(k1_0)={np.log10(TRUE_K1_0):.2f}  log10(k2_0)={np.log10(TRUE_K2_0):.2f}")
    print()

    # ── Parameter estimation ──────────────────────────────────────────────────────────
    # Reset k to wrong initial guess before fitting
    k1_0.set(8.0)
    k2_0.set(8.5)

    est = ParameterEstimator(simulator=sim, adaptive=True)
    est.add_parameters([k1_0, k2_0])
    est.add_timeseries(meas_Cb, signal=scope[1], sigma=0.05)
    est.add_timeseries(meas_Cc, signal=scope[2], sigma=0.05)

    fit = est.fit(loss="soft_l1", f_scale=0.1, max_nfev=100, verbose=2)
    print()
    est.display()

    # ── Plot ──────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # Panel 1: feed inputs
    ax = axes[0]
    ax.plot(t_feed, Ca_in_data, "C0", label="Ca_in [mol/L]")
    ax.set_ylabel("Feed Ca_in [mol/L]")
    ax.set_title("CSTR with TimeSeriesSource inputs — parameter estimation")
    ax2 = ax.twinx()
    ax2.plot(t_feed, T_in_data, "C1--", label="T_in [K]")
    ax2.set_ylabel("Feed T_in [K]")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Cb fit
    fig2, fit_axes = est.plot_fit(
        fit.x,
        experiments=[0],
        overlay=True,
        title="",
        xlabel="",
        ylabel="",
        legend=False,
    )
    plt.close(fig2)

    # Re-apply params and run manually for the plot
    est.apply(fit.x)
    sim.reset()
    sim.run(duration=100, reset=False, adaptive=True)
    t_fit, y_fit = scope.read()

    ax = axes[1]
    ax.plot(t_fit, y_fit[1], "C0-", lw=2, label="Cb fit")
    ax.plot(t_fit, y_fit[2], "C1-", lw=2, label="Cc fit")
    ax.scatter(meas_Cb.time, meas_Cb.data, c="C0", zorder=3, label="Cb meas")
    ax.scatter(meas_Cc.time, meas_Cc.data, c="C1", marker="^", zorder=3, label="Cc meas")
    ax.set_ylabel("Concentration [mol/L]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: temperature
    ax = axes[2]
    ax.plot(t_fit, y_fit[3] * 100, "C2-", lw=2, label="T fit")
    ax.set_ylabel("Temperature [K]")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
