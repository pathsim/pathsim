#########################################################################################
##
##                    LOCAL SENSITIVITY & IDENTIFIABILITY ANALYSIS
##                                 (sensitivity.py)
##
##                                  Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np


# HELPERS ===============================================================================

def _build_stats(fim: np.ndarray) -> dict:
    """Compute covariance, std_errors, correlation, eigenvalues, condition_number
    from a Fisher Information Matrix.  Returns a dict of all derived quantities.
    """
    n_p = fim.shape[0]

    covariance = np.linalg.pinv(fim)
    std_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    corr = np.full((n_p, n_p), np.nan)
    for i in range(n_p):
        for j in range(n_p):
            denom = std_errors[i] * std_errors[j]
            if denom > 0.0:
                corr[i, j] = covariance[i, j] / denom
            elif i == j:
                corr[i, j] = 1.0          # diagonal undefined → 1.0 by convention

    eigenvalues, eigenvectors = np.linalg.eigh(fim)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    pos_ev = eigenvalues[eigenvalues > 0.0]
    if len(pos_ev) == n_p and n_p >= 2:
        condition_number = float(pos_ev[0] / pos_ev[-1])
    elif len(pos_ev) == 1 and n_p == 1:
        condition_number = 1.0
    else:
        condition_number = np.inf

    return dict(
        covariance=covariance,
        std_errors=std_errors,
        correlation=corr,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition_number,
    )


def _print_param_table(param_names, param_values, std_errors, W=72):
    """Print the standard parameter value / std-error / rel-error table."""
    dash = "-" * W
    print(f"  {'Parameter':<22} {'Value':>12} {'Std Error':>12} "
          f"{'Rel Error':>10}  {'OK?':>4}")
    print(dash)

    for i, name in enumerate(param_names):
        val = param_values[i]
        se  = std_errors[i]

        if abs(val) > 1e-15 and np.isfinite(se):
            rel = se / abs(val)
            rel_str = f"{rel * 100:.2f}%"
        else:
            rel = np.inf
            rel_str = "N/A"

        flag = "✓" if (np.isfinite(rel) and rel < 0.5) else "✗"
        print(f"  {name:<22} {val:>12.4g} {se:>12.4g} "
              f"{rel_str:>10}  {flag:>4}")

    print(dash)


def _print_condition_and_correlation(condition_number, param_names, correlation, W=72):
    """Print condition number label and highly-correlated pairs."""
    cn = condition_number
    if cn < 1e3:
        cn_label = "excellent"
    elif cn < 1e6:
        cn_label = "acceptable"
    else:
        cn_label = "POOR — parameters may not be uniquely identifiable"
    print(f"\n  FIM condition number : {cn:.3g}  ({cn_label})")

    n_p = len(param_names)
    pairs = [
        (i, j, correlation[i, j])
        for i in range(n_p)
        for j in range(i + 1, n_p)
        if abs(correlation[i, j]) > 0.90
    ]

    if pairs:
        print(f"\n  Highly correlated pairs (|r| > 0.90):")
        for i, j, r in pairs:
            print(f"    {param_names[i]} ↔ {param_names[j]}"
                  f"  :  r = {r:+.3f}")
    else:
        print("  No highly correlated parameter pairs  (|r| ≤ 0.90)")


def _plot_correlation_and_eigenvalues(
    correlation, eigenvalues, param_names, axes, title_corr, title_eig
):
    """Render correlation heatmap and eigenvalue bar chart onto *axes* (length 2)."""
    import matplotlib.colors as mcolors

    n_p = len(param_names)

    # ── Correlation heatmap ────────────────────────────────────────────
    ax = axes[0]
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im   = ax.imshow(correlation, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.figure.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(n_p))
    ax.set_yticks(range(n_p))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_title(title_corr)

    for i in range(n_p):
        for j in range(n_p):
            v = correlation[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=7, color="black")
            else:
                color = "white" if abs(v) > 0.65 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    # ── FIM eigenvalue spectrum ────────────────────────────────────────
    ax2   = axes[1]
    ev    = eigenvalues
    pos   = ev > 0.0
    colors = ["steelblue" if p else "salmon" for p in pos]
    ax2.bar(range(len(ev)), np.where(pos, ev, np.abs(ev)), color=colors)

    pos_vals = ev[pos]
    if len(pos_vals) > 1 and pos_vals.max() / pos_vals.min() > 100.0:
        ax2.set_yscale("log")

    ax2.set_xticks(range(len(ev)))
    ax2.set_xticklabels([f"λ{i + 1}" for i in range(len(ev))], fontsize=9)
    ax2.set_xlabel("Eigendirection")
    ax2.set_ylabel("Eigenvalue magnitude")
    ax2.set_title(title_eig)
    ax2.grid(True, axis="y", alpha=0.3)

    if not all(pos):
        from matplotlib.patches import Patch
        ax2.legend(handles=[
            Patch(facecolor="steelblue", label="Positive"),
            Patch(facecolor="salmon",    label="Non-positive (not identifiable)"),
        ], fontsize=8)


# CLASS: SchurResult ====================================================================

class SchurResult:
    """Effective FIM for global parameters after marginalising local uncertainty.

    In a multi-experiment problem with shared (global) parameters θ_G and
    per-experiment (local) parameters θ_{L,i}, the nested Schur complement of
    the full Fisher Information Matrix gives the information available to
    constrain θ_G after accounting for local estimation uncertainty::

        S_G = F_GG  −  Σᵢ  F_{GL_i} · pinv(F_{L_iL_i}) · F_{GL_i}ᵀ

    where

    * ``F_GG = Σᵢ J_{G,i}ᵀ J_{G,i}`` — sum of per-experiment global-block FIM
    * ``F_{L_iL_i} = J_{L_i}ᵀ J_{L_i}`` — local-block FIM for experiment *i*
    * ``F_{GL_i} = J_{G,i}ᵀ J_{L_i}`` — cross block

    Compared to using the full FIM's global diagonal block alone, *S_G* is
    smaller (less information) because it removes the information "used up"
    when estimating the local parameters.

    Parameters
    ----------
    schur_fim : np.ndarray, shape (n_G, n_G)
        Pre-computed Schur complement matrix S_G.
    param_names : list of str
        Global parameter names, length ``n_G``.
    param_values : np.ndarray, shape (n_G,)
        Model-space values of the global parameters at ``x*``.

    Attributes
    ----------
    fim : np.ndarray
        Schur complement matrix S_G.
    covariance : np.ndarray
        ``pinv(S_G)``.
    std_errors : np.ndarray
        ``√diag(covariance)``.
    correlation : np.ndarray
        Normalised covariance.
    eigenvalues : np.ndarray
        Eigenvalues of S_G, descending.
    eigenvectors : np.ndarray
        Corresponding eigenvectors, columns.
    condition_number : float
        ``λ_max / λ_min`` of S_G (positive eigenvalues only).

    Notes
    -----
    Obtain a :class:`SchurResult` automatically via
    :meth:`ParameterEstimator.sensitivity` when both global and local
    parameters are registered.  It is accessible as
    ``sens.schur`` on the returned :class:`SensitivityResult`.
    """

    def __init__(
        self,
        schur_fim: np.ndarray,
        param_names: list,
        param_values: np.ndarray,
    ):
        self.fim         = np.asarray(schur_fim, dtype=float)
        self.param_names = list(param_names)
        self.param_values = np.asarray(param_values, dtype=float)

        stats = _build_stats(self.fim)
        self.covariance      = stats["covariance"]
        self.std_errors      = stats["std_errors"]
        self.correlation     = stats["correlation"]
        self.eigenvalues     = stats["eigenvalues"]
        self.eigenvectors    = stats["eigenvectors"]
        self.condition_number = stats["condition_number"]


    # DISPLAY ===========================================================================

    def display(self) -> None:
        """Print a formatted Schur complement sensitivity summary."""
        W    = 72
        line = "=" * W

        print(line)
        print("  Schur Complement — Effective Global-Parameter Sensitivity")
        print("  (per-experiment local parameters marginalised out)")
        print(line)

        _print_param_table(
            self.param_names, self.param_values, self.std_errors, W
        )
        _print_condition_and_correlation(
            self.condition_number, self.param_names, self.correlation, W
        )
        print(line)


    # PLOT ==============================================================================

    def plot(self, *, figsize: tuple = (11, 4.5)):
        """Plot Schur correlation matrix and eigenvalue spectrum for global parameters.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : np.ndarray of matplotlib.axes.Axes, shape (2,)
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        _plot_correlation_and_eigenvalues(
            self.correlation, self.eigenvalues, self.param_names, axes,
            title_corr="Global Parameter Correlation (Schur)",
            title_eig="Schur FIM Eigenvalue Spectrum",
        )
        fig.suptitle("Schur Complement — Effective Global-Parameter Analysis",
                     fontweight="bold")
        plt.tight_layout()
        return fig, axes


# CLASS: SensitivityResult ==============================================================

class SensitivityResult:
    """Result of a local sensitivity and practical identifiability analysis.

    Computed at a specific parameter vector ``x*`` (typically the optimum
    returned by :meth:`ParameterEstimator.fit`).  All statistics derive from
    the weighted Jacobian **J** of the residual vector with respect to the
    parameters (residuals are already divided by ``sigma`` inside
    :meth:`ParameterEstimator.residuals`).

    Parameters
    ----------
    jacobian : np.ndarray, shape (n_residuals, n_params)
        Weighted Jacobian matrix ``∂r_i/∂θ_j`` at ``x*``.
    param_names : list of str
        Parameter names in the same order as columns of ``jacobian``.
    param_values : np.ndarray, shape (n_params,)
        Model-space parameter values at ``x*``.
    schur : SchurResult or None, optional
        Schur complement result for the global parameters.  Present only
        when both global and local parameters are registered in a
        multi-experiment estimator.  Access via ``result.schur``.

    Attributes
    ----------
    jacobian : np.ndarray
        Weighted Jacobian, shape ``(n_residuals, n_params)``.
    param_names : list of str
    param_values : np.ndarray
    schur : SchurResult or None
        Schur complement analysis for global parameters (multi-experiment
        problems only).  ``None`` for single-experiment problems.
    fim : np.ndarray
        Fisher Information Matrix ``Jᵀ J``, shape ``(n_params, n_params)``.
    covariance : np.ndarray
        Parameter covariance estimate ``pinv(FIM)``,
        shape ``(n_params, n_params)``.
    std_errors : np.ndarray
        Approximate standard errors ``√diag(covariance)``,
        shape ``(n_params,)``.  Expressed in optimizer space; for
        untransformed parameters this equals model space.
    correlation : np.ndarray
        Parameter correlation matrix, shape ``(n_params, n_params)``.
        Off-diagonal values near ±1 indicate strongly correlated
        (potentially unidentifiable) parameter pairs.
    eigenvalues : np.ndarray
        Eigenvalues of the FIM in descending order.  Small eigenvalues
        correspond to poorly constrained parameter directions.
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns), shape ``(n_params, n_params)``.
    condition_number : float
        Ratio of the largest to smallest *positive* FIM eigenvalue.
        Values above ~1e6 indicate practical non-identifiability.

    Notes
    -----
    The analysis is *local* — it linearises around ``x*``.  Near a flat
    optimum or with highly correlated parameters the results may be
    unreliable.  Use the condition number and eigenvalue spectrum together
    rather than relying on a single threshold.
    """

    def __init__(
        self,
        jacobian: np.ndarray,
        param_names: list,
        param_values: np.ndarray,
        schur: "SchurResult | None" = None,
    ):
        self.jacobian    = np.asarray(jacobian, dtype=float)
        self.param_names = list(param_names)
        self.param_values = np.asarray(param_values, dtype=float)
        self.schur       = schur

        # Fisher Information Matrix
        self.fim = self.jacobian.T @ self.jacobian          # (n_p, n_p)

        stats = _build_stats(self.fim)
        self.covariance      = stats["covariance"]
        self.std_errors      = stats["std_errors"]
        self.correlation     = stats["correlation"]
        self.eigenvalues     = stats["eigenvalues"]
        self.eigenvectors    = stats["eigenvectors"]
        self.condition_number = stats["condition_number"]


    # DISPLAY ===========================================================================

    def display(self) -> None:
        """Print a formatted sensitivity and identifiability summary.

        Prints a table of parameter values, standard errors, and relative
        errors, followed by the FIM condition number and any highly
        correlated parameter pairs.

        When a :class:`SchurResult` is attached (multi-experiment problems),
        the Schur complement analysis for global parameters is printed as a
        second section.
        """
        W    = 72
        line = "=" * W

        print(line)
        print("  Sensitivity & Identifiability Analysis")
        print(line)

        _print_param_table(
            self.param_names, self.param_values, self.std_errors, W
        )
        _print_condition_and_correlation(
            self.condition_number, self.param_names, self.correlation, W
        )
        print(line)

        if self.schur is not None:
            print()
            self.schur.display()


    # PLOT ==============================================================================

    def plot(self, *, figsize: tuple = (11, 4.5)):
        """Plot the correlation matrix heatmap and FIM eigenvalue spectrum.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : np.ndarray of matplotlib.axes.Axes, shape (2,)
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        _plot_correlation_and_eigenvalues(
            self.correlation, self.eigenvalues, self.param_names, axes,
            title_corr="Parameter Correlation Matrix",
            title_eig="FIM Eigenvalue Spectrum",
        )
        fig.suptitle("Sensitivity & Identifiability Analysis", fontweight="bold")
        plt.tight_layout()
        return fig, axes
