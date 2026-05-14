#########################################################################################
##
##                       TIME SERIES SOURCE BLOCK (PathSim)
##                            (timeseries_source.py)
##
##                              Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ..blocks._block import Block
from ..utils.timeseries_data import TimeSeriesData


ExtrapolationMode = Literal["hold", "nan", "error"]
InterpolationKind = Literal["linear", "zoh"]


# HELPERS ================================================================================

def _interp_at(tt: float, t_arr: np.ndarray, y_arr: np.ndarray):
    """Linear interpolation at scalar time tt.

    Handles 1D and 2D ``y_arr`` in a single binary search.

    Parameters
    ----------
    tt : float
        Query time (assumed within ``[t_arr[0], t_arr[-1]]``).
    t_arr : np.ndarray
        Strictly increasing sample times, shape ``(n,)``.
    y_arr : np.ndarray
        Sample values, shape ``(n,)`` or ``(n, m)``.

    Returns
    -------
    float or np.ndarray
        Interpolated value. Scalar ``float`` for 1D data; array of shape
        ``(m,)`` for 2D data.
    """
    idx = int(np.searchsorted(t_arr, tt, side="left"))

    if idx == 0:
        return float(y_arr[0]) if y_arr.ndim == 1 else y_arr[0, :].copy()
    if idx >= len(t_arr):
        return float(y_arr[-1]) if y_arr.ndim == 1 else y_arr[-1, :].copy()

    alpha = (tt - t_arr[idx - 1]) / (t_arr[idx] - t_arr[idx - 1])

    if y_arr.ndim == 1:
        return float(y_arr[idx - 1] + alpha * (y_arr[idx] - y_arr[idx - 1]))
    return y_arr[idx - 1, :] + alpha * (y_arr[idx, :] - y_arr[idx - 1, :])


def _zoh_at(tt: float, t_arr: np.ndarray, y_arr: np.ndarray):
    """Zero-order hold at scalar time tt.

    Returns the sample value at the latest sample time at or before ``tt``.

    Parameters
    ----------
    tt : float
        Query time (assumed within ``[t_arr[0], t_arr[-1]]``).
    t_arr : np.ndarray
        Strictly increasing sample times, shape ``(n,)``.
    y_arr : np.ndarray
        Sample values, shape ``(n,)`` or ``(n, m)``.

    Returns
    -------
    float or np.ndarray
        Sample value. Scalar ``float`` for 1D data; array of shape ``(m,)``
        for 2D data.
    """
    idx = int(np.searchsorted(t_arr, tt, side="right")) - 1
    idx = max(0, min(len(t_arr) - 1, idx))
    return float(y_arr[idx]) if y_arr.ndim == 1 else y_arr[idx, :].copy()


# CLASS ==================================================================================

class TimeSeriesSource(Block):

    """Time-dependent source defined by sampled data.

    Implements:

    .. math::
        y(t) = \\mathrm{interp}(t;\\, t_i, y_i)

    Parameters
    ----------
    ts : TimeSeriesData, optional
        Data container providing ``time`` and ``data``.  Use this **or**
        the ``t`` / ``y`` keyword pair — not both.
    t : array_like, optional
        Sample times (used when ``ts`` is not provided).
    y : array_like, optional
        Sample values (used when ``ts`` is not provided).
        Shape ``(n,)`` or ``(n, n_channels)``.
    extrapolate : {'hold', 'nan', 'error'}
        Behaviour when the simulation time falls outside
        ``[t_samples[0], t_samples[-1]]``:

        - ``'hold'``  — clamp to the nearest endpoint (default).
        - ``'nan'``   — output ``NaN``.
        - ``'error'`` — raise :class:`ValueError`.

        Ignored when ``loop=True``.
    interpolation : {'linear', 'zoh'}
        Interpolation method applied between samples:

        - ``'linear'`` — linear interpolation (default).
        - ``'zoh'``    — zero-order hold; output the value of the last
          sample at or before the query time.  Useful for data from
          discrete sensors or lookup tables.
    loop : bool
        If ``True``, wrap the simulation time modulo the signal duration
        so the data repeats cyclically.  Overrides ``extrapolate``.
        Default ``False``.
    channel : int, optional
        For multi-channel data (2D ``y``), select a single output channel
        by index.  If ``None`` (default), all channels are output as a
        vector.  Validated at construction time.

    Notes
    -----
    - ``__len__`` returns 0 because source blocks have *no algebraic
      passthrough* — their output does not depend on block inputs.
    - Interpolation uses a single ``np.searchsorted`` call per timestep;
      for multi-channel data all channels are computed in one vectorised
      numpy operation (no per-channel Python loop).
    - Use :class:`~pathsim.blocks.Source` when your signal is a callable
      ``f(t)``; use ``TimeSeriesSource`` when you have recorded sample data.
    """

    input_port_labels  = {}
    output_port_labels = {"out": 0}


    def __init__(
        self,
        ts: TimeSeriesData | None = None,
        t: Sequence[float] | np.ndarray | None = None,
        y: Sequence[float] | np.ndarray | None = None,
        *,
        extrapolate: ExtrapolationMode = "hold",
        interpolation: InterpolationKind = "linear",
        loop: bool = False,
        channel: int | None = None,
    ):
        super().__init__()

        if extrapolate not in ("hold", "nan", "error"):
            raise ValueError("extrapolate must be one of: 'hold', 'nan', 'error'")
        if interpolation not in ("linear", "zoh"):
            raise ValueError("interpolation must be one of: 'linear', 'zoh'")

        if ts is not None:
            if t is not None or y is not None:
                raise ValueError("Pass either `ts=` OR (`t=`, `y=`), not both.")
            self._series = ts
        else:
            if t is None or y is None:
                raise ValueError(
                    "You must pass either `ts=TimeSeriesData(...)` or both `t=` and `y=`."
                )
            self._series = TimeSeriesData(time=np.asarray(t), data=np.asarray(y))

        # Validate channel against the known data shape
        if channel is not None:
            ch = int(channel)
            if ch < 0:
                raise ValueError(f"channel must be non-negative, got {channel}")
            data = self._series.data
            if data.ndim == 2 and ch >= data.shape[1]:
                raise IndexError(
                    f"channel={ch} out of range for {data.shape[1]}-channel data"
                )

        self.extrapolate: ExtrapolationMode = extrapolate
        self.interpolation: InterpolationKind = interpolation
        self.loop: bool = bool(loop)
        self.channel = channel


    def __len__(self):
        """Source blocks have no algebraic passthrough; returns 0."""
        return 0


    @property
    def t0(self) -> float:
        """First sample time."""
        return float(self._series.time[0])


    @property
    def t1(self) -> float:
        """Last sample time."""
        return float(self._series.time[-1])


    def update(self, t: float):
        """Update output at time ``t``.

        Parameters
        ----------
        t : float
            Evaluation time.
        """
        tt = float(t)

        t_arr = self._series.time
        y_arr = self._series.data

        # Resolve simulation time: looping takes priority over extrapolation
        if self.loop:
            dur = t_arr[-1] - t_arr[0]
            tt  = t_arr[0] + (tt - t_arr[0]) % dur
        elif tt < self.t0 or tt > self.t1:
            if self.extrapolate == "hold":
                tt = max(self.t0, min(self.t1, tt))
            elif self.extrapolate == "nan":
                self.outputs[0] = np.nan
                return
            else:  # 'error'
                raise ValueError(
                    f"Time {t} outside TimeSeriesSource range [{self.t0}, {self.t1}]"
                )

        # Interpolate (single searchsorted; vectorised across channels)
        interp_fn = _zoh_at if self.interpolation == "zoh" else _interp_at
        val = interp_fn(tt, t_arr, y_arr)

        # Channel selection or multi-port assignment
        if self.channel is not None and y_arr.ndim == 2:
            # Single selected channel → scalar on port 0
            self.outputs[0] = float(val[self.channel])
        elif y_arr.ndim == 2:
            # All channels → one port per channel (port 0, 1, ..., m-1)
            self.outputs.update_from_array(val)
        else:
            self.outputs[0] = val


    def plot(self, **kwargs):
        """Plot the source data.

        Delegates to :meth:`TimeSeriesData.plot`. All keyword arguments are
        forwarded to that method.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        return self._series.plot(**kwargs)
