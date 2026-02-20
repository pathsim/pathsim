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

import bisect
from typing import Literal, Sequence

import numpy as np

from ..blocks._block import Block
from ..opt.timeseries_data import TimeSeriesData


ExtrapolationMode = Literal["hold", "nan", "error"]


# CLASSES =================================================================================

def _interp_scalar(t: float, t_arr: np.ndarray, y_arr: np.ndarray) -> float:
    
    """Linear interpolation for scalar output.

    Parameters
    ----------
    t : float
        Query time (assumed within range).
    t_arr : np.ndarray
        Strictly increasing sample times.
    y_arr : np.ndarray
        Sample values of shape (n,).

    Returns
    -------
    float
        Interpolated value at time `t`.
    """
    
    # Find the right interval using binary search
    idx = bisect.bisect_left(t_arr, t)

    # Handle edge cases
    if idx == 0:
        return float(y_arr[0])
    if idx >= len(t_arr):
        return float(y_arr[-1])

    # Linear interpolation between idx-1 and idx
    t0, t1 = t_arr[idx - 1], t_arr[idx]
    y0, y1 = y_arr[idx - 1], y_arr[idx]

    # Compute interpolation weight
    alpha = (t - t0) / (t1 - t0)

    return float(y0 + alpha * (y1 - y0))


class TimeSeriesSource(Block):
    
    """Time-dependent source defined by sampled data.

    Implements:

    .. math::
        y(t) = \mathrm{interp}(t; t_i, y_i)

    Parameters
    ----------
    ts : TimeSeriesData, optional
        TimeSeriesData instance providing `time` and `data`.
    t : array_like, optional
        Sample times (used if `ts` is not provided).
    y : array_like, optional
        Sample values (used if `ts` is not provided). Shape (n,) or (n, n_channels).
    extrapolate : {'hold', 'nan', 'error'}
        Extrapolation behavior outside the sample window:
        - 'hold' : clamp to endpoints
        - 'nan'  : output NaN
        - 'error': raise ValueError
    channel : int, optional
        For multi-channel input, select a single channel to output.

    Notes
    -----
    - PathSim may call `update(t)` multiple times per step.
    - The block is algebraic, therefore `__len__` returns 0.
    - Interpolation uses binary search (O(log n)) for repeated evaluations.
    """

    input_port_labels = {}
    output_port_labels = {"out": 0}
    

    def __init__(
        self,
        ts: TimeSeriesData | None = None,
        t: Sequence[float] | np.ndarray | None = None,
        y: Sequence[float] | np.ndarray | None = None,
        *,
        extrapolate: ExtrapolationMode = "hold",
        channel: int | None = None,
    ):
        super().__init__()

        if extrapolate not in ("hold", "nan", "error"):
            raise ValueError("extrapolate must be one of: 'hold', 'nan', 'error'")

        if ts is not None:
            if t is not None or y is not None:
                raise ValueError("Pass either `ts=` OR (`t=`, `y=`), not both.")
            self._series = ts
        else:
            if t is None or y is None:
                raise ValueError("You must pass either `ts=TimeSeriesData(...)` or both `t=` and `y=`.")
            self._series = TimeSeriesData(time=np.asarray(t), data=np.asarray(y))

        self.extrapolate: ExtrapolationMode = extrapolate
        self.channel = channel


    def __len__(self):
        """Return algebraic block length (no internal states)."""
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
        """Update output at time `t`.

        Parameters
        ----------
        t : float
            Evaluation time.
        """
        tt = float(t)

        # Handle extrapolation
        if tt < self.t0 or tt > self.t1:
            if self.extrapolate == "hold":
                tt = np.clip(tt, self.t0, self.t1)
            elif self.extrapolate == "nan":
                self.outputs[0] = np.nan
                return
            else:  # 'error'
                raise ValueError(
                    f"Time {t} outside TimeSeriesSource range [{self.t0}, {self.t1}]"
                )

        t_arr = self._series.time
        y_arr = self._series.data

        # 1D case: single channel
        if y_arr.ndim == 1:
            self.outputs[0] = _interp_scalar(tt, t_arr, y_arr)
            return

        # 2D case: multiple channels
        if self.channel is not None:
            ch = int(self.channel)
            if ch < 0 or ch >= y_arr.shape[1]:
                raise IndexError(f"channel={ch} out of range for y.shape={y_arr.shape}")
            self.outputs[0] = _interp_scalar(tt, t_arr, y_arr[:, ch])
        else:
            # Output vector of all channels
            out = np.array(
                [_interp_scalar(tt, t_arr, y_arr[:, j]) for j in range(y_arr.shape[1])],
                dtype=float,
            )
            self.outputs[0] = out
