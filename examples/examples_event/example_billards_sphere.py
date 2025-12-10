#########################################################################################
##
##                PathSim Example for Collisions with Circular Boundary
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Integrator, Scope
from pathsim.events import ZeroCrossingUp
from pathsim.solvers import RKBS32


# SYSTEM DEFINITION =====================================================================

# system parameters
g = 9.81
l = 1

# initial conditions
x0 = np.array([0.5, 0.5])
v0 = np.array([0, 0])

# blocks for dynamics
cn = Constant(-g)
ix = Integrator(x0)
iv = Integrator(v0)
sc = Scope(labels=["x", "y"])

# collision event functions
def bounce_detect(_):
    x = ix.engine.get()
    return np.linalg.norm(x) - l 

def bounce_act(_):
    v = iv.engine.get()
    x = ix.engine.get()
    n = x / np.linalg.norm(x)
    iv.engine.set(v - 2 * np.dot(v, n) * n)
    ix.engine.set(l * n)

# simulation definition
sim = Simulation(
    blocks=[ix, iv, sc, cn],
    connections=[
        Connection(cn, iv[1]),
        Connection(iv[0,1], ix[0,1]),
        Connection(ix[0,1], sc[0,1]),
        ],
    events=[
        ZeroCrossingUp(
            func_evt=bounce_detect, 
            func_act=bounce_act,
            ),
        ],
    Solver=RKBS32,
    dt_max=0.01
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    sim.run(7)

    # plot results
    sc.plot()

    fig, ax = sc.plot2D()
    ang = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(ang), np.sin(ang), color="k")

    plt.show()