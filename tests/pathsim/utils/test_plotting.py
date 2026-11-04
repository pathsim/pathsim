########################################################################################
##
##                                  TESTS FOR
##                              'utils.plotting.py'
##
##                              Milan Rother 2026
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backend_bases import FigureManagerBase

from pathsim.utils.plotting import show


# TESTS ================================================================================

class TestShow(unittest.TestCase):
    """
    Test the 'show' helper for figures
    """

    def tearDown(self):
        plt.close("all")


    def test_no_window(self):
        """backends without a window (Agg, Jupyter inline) dont show"""

        fig, _ = plt.subplots()
        self.assertIs(type(fig.canvas.manager), FigureManagerBase)

        with mock.patch("pathsim.utils.plotting.plt.show") as plt_show:
            show(fig)
            plt_show.assert_not_called()


    def test_window(self):
        """backends with a window show without blocking"""

        class WindowManager(FigureManagerBase):
            pass

        fig, _ = plt.subplots()
        fig.canvas.manager = WindowManager(fig.canvas, 1)

        with mock.patch("pathsim.utils.plotting.plt.show") as plt_show:
            show(fig)
            plt_show.assert_called_once_with(block=False)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
