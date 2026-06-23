########################################################################################
##
##                                  TESTS FOR
##                           'blocks.timeseries_source.py'
##
########################################################################################

# IMPORTS ==============================================================================

import math
import unittest

import numpy as np

from pathsim.blocks.timeseries_source import TimeSeriesSource, _interp_at, _zoh_at
from pathsim.utils.timeseries_data import TimeSeriesData


# HELPERS ==============================================================================

def _make_ramp(n=11, t_end=10.0):
    """Return (t, y) for a ramp y=t sampled at n evenly spaced points."""
    t = np.linspace(0.0, t_end, n)
    return t, t.copy()


# TESTS ================================================================================

class TestInterpAt(unittest.TestCase):
    """Unit tests for the _interp_at helper."""

    def setUp(self):
        self.t = np.array([0.0, 1.0, 2.0, 3.0])
        self.y = np.array([0.0, 2.0, 4.0, 6.0])  # y = 2t

    def test_at_sample_points(self):
        for i, (ti, yi) in enumerate(zip(self.t, self.y)):
            with self.subTest(i=i):
                self.assertAlmostEqual(_interp_at(ti, self.t, self.y), yi)

    def test_midpoint_interpolation(self):
        self.assertAlmostEqual(_interp_at(0.5, self.t, self.y), 1.0)
        self.assertAlmostEqual(_interp_at(1.5, self.t, self.y), 3.0)

    def test_at_first_sample(self):
        self.assertAlmostEqual(_interp_at(0.0, self.t, self.y), 0.0)

    def test_at_last_sample(self):
        self.assertAlmostEqual(_interp_at(3.0, self.t, self.y), 6.0)

    def test_2d_interpolation(self):
        y2d = np.column_stack([self.y, self.y * 2])  # shape (4, 2)
        result = _interp_at(0.5, self.t, y2d)
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_2d_boundary_returns_copy(self):
        y2d = np.column_stack([self.y, self.y])
        result = _interp_at(0.0, self.t, y2d)
        result[0] = 999.0  # mutate the returned array
        # The source data should be untouched
        self.assertAlmostEqual(float(y2d[0, 0]), 0.0)


class TestZohAt(unittest.TestCase):
    """Unit tests for the _zoh_at helper."""

    def setUp(self):
        self.t = np.array([0.0, 1.0, 2.0, 3.0])
        self.y = np.array([10.0, 20.0, 30.0, 40.0])

    def test_at_sample_holds_current(self):
        self.assertAlmostEqual(_zoh_at(1.0, self.t, self.y), 20.0)
        self.assertAlmostEqual(_zoh_at(2.0, self.t, self.y), 30.0)

    def test_between_samples_holds_previous(self):
        self.assertAlmostEqual(_zoh_at(0.5, self.t, self.y), 10.0)
        self.assertAlmostEqual(_zoh_at(1.9, self.t, self.y), 20.0)

    def test_at_first_sample(self):
        self.assertAlmostEqual(_zoh_at(0.0, self.t, self.y), 10.0)

    def test_at_last_sample(self):
        self.assertAlmostEqual(_zoh_at(3.0, self.t, self.y), 40.0)

    def test_2d_zoh(self):
        y2d = np.column_stack([self.y, self.y * 0.5])
        result = _zoh_at(0.5, self.t, y2d)
        np.testing.assert_allclose(result, [10.0, 5.0])


class TestTimeSeriesSourceInit(unittest.TestCase):
    """Constructor validation."""

    def test_init_with_arrays(self):
        t, y = _make_ramp()
        src = TimeSeriesSource(t=t, y=y)
        self.assertIsInstance(src._series, TimeSeriesData)

    def test_init_with_timeseries_data(self):
        t, y = _make_ramp()
        ts = TimeSeriesData(time=t, data=y)
        src = TimeSeriesSource(ts=ts)
        self.assertIs(src._series, ts)

    def test_both_ts_and_arrays_raises(self):
        t, y = _make_ramp()
        ts = TimeSeriesData(time=t, data=y)
        with self.assertRaises(ValueError):
            TimeSeriesSource(ts=ts, t=t, y=y)

    def test_neither_ts_nor_arrays_raises(self):
        with self.assertRaises(ValueError):
            TimeSeriesSource()

    def test_only_t_without_y_raises(self):
        t, _ = _make_ramp()
        with self.assertRaises(ValueError):
            TimeSeriesSource(t=t)

    def test_invalid_extrapolate_raises(self):
        t, y = _make_ramp()
        with self.assertRaises(ValueError):
            TimeSeriesSource(t=t, y=y, extrapolate="clamp")

    def test_invalid_interpolation_raises(self):
        t, y = _make_ramp()
        with self.assertRaises(ValueError):
            TimeSeriesSource(t=t, y=y, interpolation="cubic")

    def test_channel_negative_raises(self):
        t, y = _make_ramp()
        with self.assertRaises(ValueError):
            TimeSeriesSource(t=t, y=y, channel=-1)

    def test_channel_out_of_range_raises(self):
        t = np.linspace(0, 1, 5)
        y = np.column_stack([t, t])  # 2 channels
        with self.assertRaises(IndexError):
            TimeSeriesSource(t=t, y=y, channel=5)

    def test_channel_valid_2d(self):
        t = np.linspace(0, 1, 5)
        y = np.column_stack([t, t * 2])
        src = TimeSeriesSource(t=t, y=y, channel=1)
        self.assertEqual(src.channel, 1)

    def test_defaults(self):
        t, y = _make_ramp()
        src = TimeSeriesSource(t=t, y=y)
        self.assertEqual(src.extrapolate, "hold")
        self.assertEqual(src.interpolation, "linear")
        self.assertFalse(src.loop)
        self.assertIsNone(src.channel)


class TestTimeSeriesSourceLen(unittest.TestCase):
    def test_len_is_zero(self):
        t, y = _make_ramp()
        src = TimeSeriesSource(t=t, y=y)
        self.assertEqual(len(src), 0)


class TestTimeSeriesSourceProperties(unittest.TestCase):
    def test_t0_t1(self):
        t = np.array([2.0, 5.0, 8.0])
        y = np.array([1.0, 2.0, 3.0])
        src = TimeSeriesSource(t=t, y=y)
        self.assertAlmostEqual(src.t0, 2.0)
        self.assertAlmostEqual(src.t1, 8.0)


class TestUpdateLinear(unittest.TestCase):
    """update() with interpolation='linear' (default)."""

    def setUp(self):
        # y = 2t ramp on [0, 4]
        self.t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.y = self.t * 2.0
        self.src = TimeSeriesSource(t=self.t, y=self.y)

    def test_at_sample_points(self):
        for ti, yi in zip(self.t, self.y):
            self.src.update(ti)
            self.assertAlmostEqual(self.src.outputs[0], yi)

    def test_midpoint(self):
        self.src.update(0.5)
        self.assertAlmostEqual(self.src.outputs[0], 1.0)

    def test_hold_before_start(self):
        self.src.update(-1.0)
        self.assertAlmostEqual(self.src.outputs[0], 0.0)  # clamped to t0

    def test_hold_after_end(self):
        self.src.update(100.0)
        self.assertAlmostEqual(self.src.outputs[0], 8.0)  # clamped to t1 = y[-1] = 8


class TestUpdateZoh(unittest.TestCase):
    """update() with interpolation='zoh'."""

    def setUp(self):
        self.t = np.array([0.0, 1.0, 2.0, 3.0])
        self.y = np.array([10.0, 20.0, 30.0, 40.0])
        self.src = TimeSeriesSource(t=self.t, y=self.y, interpolation="zoh")

    def test_at_sample(self):
        self.src.update(1.0)
        self.assertAlmostEqual(self.src.outputs[0], 20.0)

    def test_between_samples(self):
        self.src.update(1.5)
        self.assertAlmostEqual(self.src.outputs[0], 20.0)  # holds sample at t=1

    def test_at_last_sample(self):
        self.src.update(3.0)
        self.assertAlmostEqual(self.src.outputs[0], 40.0)


class TestExtrapolate(unittest.TestCase):
    """Extrapolation modes."""

    def setUp(self):
        self.t = np.array([1.0, 2.0, 3.0])
        self.y = np.array([10.0, 20.0, 30.0])

    def test_hold_before(self):
        src = TimeSeriesSource(t=self.t, y=self.y, extrapolate="hold")
        src.update(0.0)
        self.assertAlmostEqual(src.outputs[0], 10.0)

    def test_hold_after(self):
        src = TimeSeriesSource(t=self.t, y=self.y, extrapolate="hold")
        src.update(5.0)
        self.assertAlmostEqual(src.outputs[0], 30.0)

    def test_nan_before(self):
        src = TimeSeriesSource(t=self.t, y=self.y, extrapolate="nan")
        src.update(0.0)
        self.assertTrue(math.isnan(src.outputs[0]))

    def test_nan_after(self):
        src = TimeSeriesSource(t=self.t, y=self.y, extrapolate="nan")
        src.update(5.0)
        self.assertTrue(math.isnan(src.outputs[0]))

    def test_error_before(self):
        src = TimeSeriesSource(t=self.t, y=self.y, extrapolate="error")
        with self.assertRaises(ValueError):
            src.update(0.0)

    def test_error_after(self):
        src = TimeSeriesSource(t=self.t, y=self.y, extrapolate="error")
        with self.assertRaises(ValueError):
            src.update(5.0)

    def test_within_range_no_extrapolation(self):
        src = TimeSeriesSource(t=self.t, y=self.y, extrapolate="error")
        src.update(2.0)  # should not raise
        self.assertAlmostEqual(src.outputs[0], 20.0)


class TestLoopMode(unittest.TestCase):
    """loop=True repeats the signal cyclically."""

    def setUp(self):
        # Signal on [0, 1]: y = t
        self.t = np.linspace(0.0, 1.0, 11)
        self.y = self.t.copy()
        self.src = TimeSeriesSource(t=self.t, y=self.y, loop=True)

    def test_within_first_period(self):
        self.src.update(0.5)
        self.assertAlmostEqual(self.src.outputs[0], 0.5, places=5)

    def test_second_period(self):
        self.src.update(1.5)
        self.assertAlmostEqual(self.src.outputs[0], 0.5, places=5)

    def test_third_period(self):
        self.src.update(2.75)
        self.assertAlmostEqual(self.src.outputs[0], 0.75, places=5)

    def test_at_period_boundary_wraps_to_start(self):
        # t=1.0 maps to t=0.0 (end of period = start of next period)
        self.src.update(1.0)
        self.assertAlmostEqual(self.src.outputs[0], 0.0, places=5)

    def test_before_start_wraps(self):
        # Negative time should also wrap
        self.src.update(-0.25)
        self.assertAlmostEqual(self.src.outputs[0], 0.75, places=5)

    def test_loop_overrides_extrapolate_error(self):
        src = TimeSeriesSource(t=self.t, y=self.y, loop=True, extrapolate="error")
        src.update(5.0)  # must not raise
        self.assertGreaterEqual(src.outputs[0], 0.0)


class TestMultiChannel(unittest.TestCase):
    """Multi-channel 2D data."""

    def setUp(self):
        self.t = np.linspace(0.0, 1.0, 6)
        # ch0 = t, ch1 = 2t
        self.y2d = np.column_stack([self.t, 2 * self.t])
        self.src_all = TimeSeriesSource(t=self.t, y=self.y2d)
        self.src_ch0 = TimeSeriesSource(t=self.t, y=self.y2d, channel=0)
        self.src_ch1 = TimeSeriesSource(t=self.t, y=self.y2d, channel=1)

    def test_all_channels_output_on_separate_ports(self):
        # Multi-channel data without channel= spreads across ports 0, 1, ...
        self.src_all.update(0.4)
        self.assertAlmostEqual(float(self.src_all.outputs[0]), 0.4, places=5)
        self.assertAlmostEqual(float(self.src_all.outputs[1]), 0.8, places=5)

    def test_channel_0_scalar(self):
        self.src_ch0.update(0.6)
        self.assertAlmostEqual(float(self.src_ch0.outputs[0]), 0.6, places=5)

    def test_channel_1_scalar(self):
        self.src_ch1.update(0.6)
        self.assertAlmostEqual(float(self.src_ch1.outputs[0]), 1.2, places=5)

    def test_zoh_multi_channel(self):
        src = TimeSeriesSource(t=self.t, y=self.y2d, interpolation="zoh")
        src.update(0.35)
        # nearest sample at or before 0.35: t step = 0.2, so index 1 → t=0.2
        expected_ch0 = self.t[1]       # 0.2
        expected_ch1 = 2 * self.t[1]   # 0.4
        self.assertAlmostEqual(float(src.outputs[0]), expected_ch0, places=5)
        self.assertAlmostEqual(float(src.outputs[1]), expected_ch1, places=5)


class TestPlotMethod(unittest.TestCase):
    """plot() delegates to TimeSeriesData.plot()."""

    def test_plot_returns_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t, y = _make_ramp()
        src = TimeSeriesSource(t=t, y=y)
        ax = src.plot()
        self.assertIsInstance(ax, plt.Axes)
        plt.close("all")

    def test_plot_reuses_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t, y = _make_ramp()
        src = TimeSeriesSource(t=t, y=y)
        _, ax = plt.subplots()
        returned = src.plot(ax=ax)
        self.assertIs(returned, ax)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
