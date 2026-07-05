#########################################################################################
##
##          PathSim Example: Longitudinal Trim of an Aircraft Short-Period Model
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Adder, Amplifier, Integrator, Scope
from pathsim.blocks.table import LUT1D


# AIRCRAFT LONGITUDINAL SHORT-PERIOD MODEL ===============================================
#
# Simplified constant-airspeed longitudinal model with 2 states: angle of
# attack 'alpha' [rad] and pitch rate 'q' [rad/s]. The point of this example
# is to build it the way a real aircraft model gets built: the aerodynamics
# (nonlinear CL/Cm vs. alpha, including a stall-like rolloff) come from lookup
# tables that one engineer owns, the rigid-body dynamics (how forces/moments
# turn into accelerations) come from a separate set of blocks that another
# engineer owns, and they only get combined at the very end by wiring
# connections between them -- 'trim()' then has to work with whatever graph
# of blocks it's handed, not a single closed-form equation.
#
# Trim target: straight-and-level flight at the given airspeed, i.e. lift
# balances weight and the pitching moment is zero, with zero pitch rate.

#aircraft + flight condition parameters (generic light aircraft, SI units)
m, S, c, Iyy = 1200.0, 16.2, 1.49, 1346.0     # mass, wing area, chord, pitch inertia
rho, V0, g = 1.225, 60.0, 9.81                # air density, trim airspeed, gravity
qbar = 0.5 * rho * V0**2                      # dynamic pressure
W = m * g                                     # weight

CM_q, CM_de = -12.0, -1.0                     # pitch damping / elevator derivatives [1/rad]

# --- Aerodynamics (one engineer's contribution): CL(alpha), Cm0(alpha) tables ----------
#nonlinear lift/moment coefficient curves, including a stall-like rolloff in CL
#past ~15 deg -- this nonlinearity is exactly why trim needs a numerical solve
alpha_deg  = np.array([-10, -5,  0,  5,  10,  15,  20])
alpha_grid = np.radians(alpha_deg)
CL_vals    = np.array([-0.60, -0.10, 0.40, 0.90, 1.30, 1.50, 1.40])
CM0_const, CM_alpha_const = 0.02, -0.5
CM_vals    = CM0_const + CM_alpha_const * alpha_grid

CL_table = LUT1D(points=alpha_grid, values=CL_vals)   # alpha -> CL(alpha)
CM_table = LUT1D(points=alpha_grid, values=CM_vals)   # alpha -> Cm0(alpha) (excl. q, de)

# --- Rigid-body dynamics (another engineer's contribution) -----------------------------
alpha_int = Integrator(0.0)   # state: angle of attack
q_int     = Integrator(0.0)   # state: pitch rate
de_src    = Constant(0.0)     # free input: elevator deflection [rad]

#vertical force balance -> alpha_dot = q - (L - W) / (m*V0)
L_amp        = Amplifier(qbar * S)
W_const      = Constant(W)
LW_diff      = Adder("+-")
adot_from_LW = Amplifier(-1.0 / (m * V0))
adot_sum     = Adder()

#pitching moment -> q_dot = M / Iyy
M_static_amp = Amplifier(qbar * S * c)
M_q_amp      = Amplifier(qbar * S * c * CM_q * c / (2 * V0))
M_de_amp     = Amplifier(qbar * S * c * CM_de)
M_sum        = Adder()
qdot_amp     = Amplifier(1.0 / Iyy)

Sco = Scope(labels=["alpha [rad]", "q [rad/s]"])

blocks = [
    alpha_int, q_int, de_src,
    CL_table, CM_table,
    L_amp, W_const, LW_diff, adot_from_LW, adot_sum,
    M_static_amp, M_q_amp, M_de_amp, M_sum, qdot_amp,
    Sco,
    ]

connections = [
    #alpha feeds both aero tables, and is recorded on the scope
    Connection(alpha_int, CL_table, CM_table, Sco[0]),
    #lift vs weight -> contributes to alpha_dot
    Connection(CL_table, L_amp),
    Connection(L_amp, LW_diff[0]),
    Connection(W_const, LW_diff[1]),
    Connection(LW_diff, adot_from_LW),
    #q feeds the alpha_dot equation, the pitch-damping term, and the scope
    Connection(q_int, adot_sum[0], M_q_amp, Sco[1]),
    Connection(adot_from_LW, adot_sum[1]),
    Connection(adot_sum, alpha_int),
    #pitching moment -> q_dot
    Connection(CM_table, M_static_amp),
    Connection(de_src, M_de_amp),
    Connection(M_static_amp, M_sum[0]),
    Connection(M_q_amp, M_sum[1]),
    Connection(M_de_amp, M_sum[2]),
    Connection(M_sum, qdot_amp),
    Connection(qdot_amp, q_int),
    ]

Sim = Simulation(blocks, connections, dt=0.01, log=False)


# Run Example ===========================================================================

if __name__ == "__main__":

    #trim for straight-and-level flight: q=0 (output target) and moment/lift
    #balance (the implicit dx/dt=0 on alpha and q), solving for alpha and the
    #elevator deflection 'de_src' that achieve it
    result = Sim.trim(targets=[(q_int[0], 0.0)], free=[de_src])

    alpha_trim = alpha_int.state[0]
    de_trim = de_src.value

    print(f"trim success   : {result.success}")
    print(f"iterations     : {result.iterations}")
    print(f"residual       : {result.residual:.2e}")
    print(f"trim alpha     : {np.degrees(alpha_trim):.3f} deg")
    print(f"trim elevator  : {np.degrees(de_trim):.3f} deg")

    #start close to the trim point (a 5 deg angle-of-attack disturbance) and
    #watch the short-period mode settle back towards equilibrium
    alpha_int.state = alpha_int.state + np.radians(5.0)
    Sim.run(duration=8.0, reset=False)

    t, (alpha_hist, q_hist) = Sco.read()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, tight_layout=True)

    ax1.plot(t, np.degrees(alpha_hist), lw=2)
    ax1.axhline(np.degrees(alpha_trim), color="k", ls="--", lw=1, label="trim alpha")
    ax1.set_ylabel("alpha [deg]")
    ax1.set_title("Short-period response after a 5 deg angle-of-attack disturbance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, np.degrees(q_hist), lw=2)
    ax2.axhline(0.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("q [deg/s]")
    ax2.grid(True, alpha=0.3)

    plt.show()
