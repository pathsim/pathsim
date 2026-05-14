#########################################################################################
##
##  PathSim example: multi-experiment nested Schur optimization
##
##  Model:   First-order decay with per-experiment amplitude
##
##      y'(t) = -k * y(t)        (same decay rate k across all experiments)
##      y(0)  = 1                (unit initial condition — amplitude handled separately)
##      output = amp_i * y(t)    (per-experiment scale)
##
##  Analytical solution:  output_i(t) = amp_i * exp(-k * t)
##
##  Parameters
##  ----------
##  Global (shared across all experiments):
##      k      [1/s]  decay rate   — Constant block value, via add_global_block_parameter
##
##  Local (one per experiment):
##      amp_i  [ ]    output amplitude — Amplifier gain, via add_local_block_parameter
##
##
##  Part 1 — Default path: least_squares on reduced residual (outer problem)
##  ------------------------------------------------------------------------
##  fit_nested() uses scipy.optimize.least_squares as the outer optimizer.
##  It operates directly on the reduced residual vector:
##
##      Outer:  min_{k}   ½‖r_red(k)‖²     via TRF  (supports robust loss)
##      Inner:  min_{amp_i}  ‖r_i(k, amp_i)‖²   independently per experiment
##
##  The reduced residual is obtained by projecting out the local directions:
##
##      r_red_i = P_{L,i} r_i*          P_{L,i} = I - J_{L,i}(J_{L,i}^T J_{L,i})^{-1} J_{L,i}^T
##      J_red_i = P_{L,i} J_{G,i}
##
##  Computed stably via a single lstsq solve:
##
##      A = lstsq(J_Li, [r_i | J_Gi])
##      r_red_i = r_i - J_Li @ A[:,0]
##      J_red_i = J_Gi - J_Li @ A[:,1:]
##
##
##  Part 2 — Trust-region outer optimizer (outer_method=)
##  -----------------------------------------------------
##  The outer_method= parameter lets you swap the outer optimizer to any
##  scipy.optimize.minimize method.  The reduced residual problem is
##  automatically converted to a scalar form:
##
##      f(k)   =  ½‖r_red(k)‖²              scalar objective
##      ∇f(k)  =  J_red^T r_red             gradient
##      H(k)  ≈  J_red^T J_red              Gauss-Newton Hessian approximation
##
##  The Gauss-Newton Hessian is exact when residuals at the optimum are small
##  (well-fitting model) and is supplied automatically to Hessian-based methods
##  such as trust-ncg and trust-exact.
##
##  Primary motivation for switching:  outer_constraints= lets you enforce
##  nonlinear inequality or equality constraints on the global parameters —
##  something the default least_squares path does not support.
##
##
##  After fitting, sensitivity() computes:
##    · Full joint FIM over all parameters
##    · SchurResult: effective FIM for k after marginalizing the amp_i
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, Scope
from pathsim.blocks.sources import Constant
from pathsim.blocks.ode import ODE
from pathsim.solvers import SSPRK22

from pathsim.opt import ParameterEstimator, TimeSeriesData


# TRUE PARAMETER VALUES =================================================================

TRUE_K    = 0.4               # [1/s]  decay rate  — shared across experiments
TRUE_AMPS = [8.0, 3.5, 1.2]  # output amplitudes  — one per experiment


# SIMULATION MODEL ======================================================================
#
#   Constant(k) ---> ODE(y'=-k*y, y0=1) ---> Amplifier(gain=amp_i) ---> Scope
#
# The decay rate k enters the ODE as an input from the Constant block so that
# add_global_block_parameter can target it across all deep-copied experiments.
# Each experiment's Amplifier gain is the local amplitude parameter.

k_const = Constant(value=TRUE_K)
ode     = ODE(
    func=lambda x, u, t: np.array([-u[0] * x[0]]),
    initial_value=[1.0],
)
amp_out = Amplifier(gain=1.0)
scope   = Scope(labels=["y(t)"])

blocks = [k_const, ode, amp_out, scope]
connections =  [
        Connection(k_const[0], ode[0]),
        Connection(ode[0],     amp_out[0]),
        Connection(amp_out[0], scope[0]),
    ]

sim = Simulation(
    blocks=blocks,
    connections=connections,
    Solver=SSPRK22,
    dt=0.05, dt_min=1e-12,
    tolerance_lte_rel=1e-5,
    tolerance_lte_abs=1e-8,
    log=False,
)


# SYNTHETIC MEASUREMENTS ================================================================

def analytical(t, k, amp):
    return amp * np.exp(-k * t)


t_meas = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
rng    = np.random.default_rng(17)

measurements = []
for amp in TRUE_AMPS:
    y_clean = analytical(t_meas, TRUE_K, amp)
    y_noisy = y_clean * (1.0 + 0.05 * rng.standard_normal(len(t_meas)))
    measurements.append(TimeSeriesData(time=t_meas, data=np.maximum(y_noisy, 0.0),
                                       name=f"amp={amp}"))


# RUN EXAMPLE ===========================================================================

if __name__ == '__main__':

    n_exp = len(TRUE_AMPS)

    # ── Build estimator ──────────────────────────────────────────────────────
    est = ParameterEstimator(simulator=sim, adaptive=True)
    for _ in range(1, n_exp):
        est.add_experiment(sim, copy_sim=True, adaptive=True)

    for i, meas in enumerate(measurements):
        est.add_timeseries(meas, signal=scope[0], experiment=i)

    # Global: Constant.value = k  (SharedBlockParameter, writes to all copies)
    est.add_global_block_parameter(
        "Constant", "value",
        value=0.2, bounds=(0.01, 5.0),
        param_id="k",
    )

    # Local: Amplifier.gain = amp_i  (one per experiment)
    for i in range(n_exp):
        est.add_local_block_parameter(
            i, "Amplifier", "gain",
            value=2.0, bounds=(0.05, 30.0),
            param_id=f"amp{i}",
        )


    # PART 1: DEFAULT OUTER OPTIMIZER (trust-region reflective) =================
    #
    # fit_nested() defaults to outer_method="least_squares", which uses
    # scipy's TRF algorithm on the reduced residual vector r_red directly.
    # It supports robust loss functions (loss="soft_l1" here) to downweight
    # any outlier measurements.

    print("=" * 60)
    print("Part 1 — Default outer optimizer (trust-region reflective)")
    print("=" * 60)
    print()
    print("Running nested Schur fit (outer: TRF least_squares) ...")

    fit_trf = est.fit_nested(
        x0_G=[0.2],
        max_outer_nfev=30,
        max_inner_nfev=20,
        loss="soft_l1",
        f_scale=0.3,
        verbose=1,
    )
    print()
    est.display()
    print(f"\n  True values:  k={TRUE_K}  amps={TRUE_AMPS}")

    # ── Sensitivity + Schur ──────────────────────────────────────────────────
    #
    # sensitivity() uses the full Jacobian to compute:
    #   · Joint FIM over all parameters (k, amp_0, amp_1, amp_2)
    #   · SchurResult: effective FIM for k after marginalizing the amp_i
    #
    print("\nComputing post-fit sensitivity + Schur complement ...")
    sens = est.sensitivity()
    sens.display()


    # PART 2: TRUST-REGION OUTER OPTIMIZER (outer_method=) ======================
    #
    # Switching outer_method= converts the outer problem from a residual-based
    # least_squares call to a scalar scipy.optimize.minimize call:
    #
    #   f(k)  = 0.5 * ||r_red(k)||^2
    #   grad  = J_red^T r_red
    #   hess  = J_red^T J_red   (Gauss-Newton; passed to Hessian-aware methods)
    #
    # The inner loop (per-experiment local solve) is unchanged in both cases.

    print()
    print("=" * 60)
    print("Part 2 — Trust-region outer optimizer (outer_method=)")
    print("=" * 60)


    # ── 2a. trust-ncg ────────────────────────────────────────────────────────
    #
    # trust-ncg (Newton conjugate-gradient) uses the exact Gauss-Newton
    # Hessian on each outer step, which is automatically supplied by
    # fit_nested().  For well-conditioned problems it often converges in
    # fewer outer function evaluations than TRF.
    #
    # Note: trust-ncg does not support bounds natively; bounds set on the
    # global parameters are passed to minimize() but silently ignored by
    # this method.  Use trust-constr if bounds must be enforced strictly.

    print()
    print("Running nested Schur fit (outer: trust-ncg) ...")

    fit_ncg = est.fit_nested(
        x0_G=[0.2],
        max_outer_nfev=30,
        max_inner_nfev=20,
        outer_method="trust-ncg",
        verbose=1,
    )
    print(f"\n  trust-ncg result:  k={fit_ncg.x[0]:.4f}  "
          f"amps={np.round(fit_ncg.x[1:], 3)}  "
          f"nfev={fit_ncg.nfev}  success={fit_ncg.success}")
    print(f"  True values:       k={TRUE_K}  amps={TRUE_AMPS}")


    # ── 2b. trust-constr with a nonlinear constraint ─────────────────────────
    #
    # trust-constr is the main reason to reach for outer_method=.  It
    # supports nonlinear equality and inequality constraints on the global
    # parameters via outer_constraints=, which are forwarded directly to
    # scipy.optimize.minimize.
    #
    # Physical motivation: for this decay model, the half-life is
    #
    #     tau = ln(2) / k
    #
    # Suppose we have prior knowledge that the half-life must be at most
    # 2 seconds, i.e.  tau <= 2  =>  k >= ln(2)/2 ~= 0.347.
    #
    # At TRUE_K = 0.4 this constraint is inactive, so we expect to recover
    # the same answer as the unconstrained fit — confirming that the
    # constraint machinery does not distort the solution when the constraint
    # is not binding.
    #
    # Constraint dict format follows scipy:
    #   type="ineq"  means  fun(x) >= 0
    #   fun(x) = x[0] - ln(2)/2  encodes  k >= ln(2)/2

    K_MIN = np.log(2) / 2.0   # ~0.347  (half-life <= 2 s)
    # Try this with a tighter constraint (e.g. K_MIN = 0.39) to see the 
    # effect of an active constraint that distorts the solution.

    half_life_constraint = {
        "type": "ineq",
        "fun":  lambda x: x[0] - K_MIN,     # k >= ln(2)/2  =>  fun >= 0
        "jac":  lambda x: np.array([1.0]),   # d/dk (k - K_MIN) = 1  (optional but cheap)
    }

    print()
    print(f"Running nested Schur fit (outer: trust-constr,  k >= ln(2)/2 = {K_MIN:.4f}) ...")

    fit_tc = est.fit_nested(
        x0_G=[0.2],
        max_outer_nfev=50,
        max_inner_nfev=20,
        outer_method="trust-constr",
        outer_constraints=[half_life_constraint],
        verbose=1,
    )
    k_fit = fit_tc.x[0]
    print(f"\n  trust-constr result:  k={k_fit:.4f}  "
          f"amps={np.round(fit_tc.x[1:], 3)}  "
          f"nfev={fit_tc.nfev}  success={fit_tc.success}")
    print(f"  Constraint  k >= {K_MIN:.4f}:  "
          f"{'satisfied' if k_fit >= K_MIN - 1e-6 else 'VIOLATED'}  "
          f"(active: {abs(k_fit - K_MIN) < 1e-3})")
    print(f"  True values:          k={TRUE_K}  amps={TRUE_AMPS}")


    # COMPARISON PLOT ===========================================================

    methods   = ["TRF\n(default)", "trust-ncg", "trust-constr\n(constrained)"]
    k_results = [fit_trf.x[0], fit_ncg.x[0], fit_tc.x[0]]
    nfevs     = [fit_trf.nfev, fit_ncg.nfev, fit_tc.nfev]
    colors    = ["C0", "C1", "C2"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: recovered k value per method
    ax = axes[0]
    bars = ax.bar(methods, k_results, color=colors, width=0.5, zorder=3)
    ax.axhline(TRUE_K, color="k", lw=1.5, ls="--", label=f"True k = {TRUE_K}")
    ax.axhline(K_MIN,  color="red", lw=1.2, ls=":",
               label=f"Constraint k >= {K_MIN:.3f}")
    ax.set_ylabel("Recovered k  [1/s]")
    ax.set_title("Global parameter recovery")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(k_results) * 1.3)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, k_results):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: outer function evaluations per method
    ax2 = axes[1]
    ax2.bar(methods, nfevs, color=colors, width=0.5, zorder=3)
    ax2.set_ylabel("Outer function evaluations")
    ax2.set_title("Outer optimizer cost")
    ax2.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(ax2.patches, nfevs):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 str(val), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Nested Schur — outer optimizer comparison", fontweight="bold")
    plt.tight_layout()

    # ── Fit quality and sensitivity plots (from Part 1) ───────────────────────
    fig_fit, _ = est.plot_fit(
        fit_trf.x,
        title="Decay fit (nested Schur, TRF outer)",
        xlabel="Time [s]",
        ylabel="y(t)",
    )

    fig_sens, _ = sens.plot()

    if sens.schur is not None:
        fig_schur, _ = sens.schur.plot()

    plt.show()
