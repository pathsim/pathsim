import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Function


class TestVectorAlgebraicLoop(unittest.TestCase):

    def test_coupled_ports(self):
        #a contractive two-channel loop has a unique fixed point
        A = np.array([[0.5, 0.2], [-0.1, 0.3]])
        b = np.array([1.0, 2.0])
        block = Function(lambda x, y: tuple(A @ np.array([x, y]) + b))
        sim = Simulation(
            [block], [Connection(block[0, 1], block[0, 1])], log=False
            )
        sim.run(0.1)
        np.testing.assert_allclose(
            block.outputs.to_array(), np.linalg.solve(np.eye(2) - A, b),
            atol=1e-10, rtol=0.0
            )
