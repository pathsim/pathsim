#########################################################################################
##
##                             PathSim Nested Subsystems Example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection, Interface, Subsystem
from pathsim.blocks import Integrator, Scope, Function, Multiplier, Adder, Amplifier, Constant
from pathsim.solvers import GEAR52A, ESDIRK43, ESDIRK32


# VAN DER POL OSCILLATOR INITIAL VALUE PROBLEM ==========================================

#initial condition
x1_0 = 2
x2_0 = 0

#van der Pol parameter
mu = 1000

#simulation timestep
dt = 0.01


# subsystem for modeling ode function ---------------------------------------------------

In = Interface()
M1 = Multiplier()
C1 = Constant(1)
A1 = Amplifier(mu)
A2 = Amplifier(-1)
A3 = Amplifier(-1)
P1 = Adder()
P2 = Adder()
F1 = Function(lambda x: x**2)

fn_blocks = [In, M1, C1, A1, A2, A3, P1, P2, F1]
fn_connections = [
    Connection(In[0], A2, F1),
    Connection(In[1], M1[0]),
    Connection(F1, A3),
    Connection(A3, P2[0]),
    Connection(C1, P2[1]),
    Connection(P2, M1[1]),
    Connection(M1, A1),
    Connection(A1, P1[0]),
    Connection(A2, P1[1]),
    Connection(P1, In)
    ]

Fn = Subsystem(fn_blocks, fn_connections)


# subsystem with two integrators emulating ODE block ------------------------------------

If = Interface()
I1 = Integrator(x1_0)
I2 = Integrator(x2_0)

vdp_blocks = [If, I1, I2, Fn]
vdp_connections = [
    Connection(I2, I1, Fn[1], If[1]), 
    Connection(I1, Fn, If), 
    Connection(Fn, I2)
    ]

VDP = Subsystem(vdp_blocks, vdp_connections)


# top level system ----------------------------------------------------------------------

Sco = Scope()

#blocks of the main system
blocks = [VDP, Sco]

#the connections between the blocks in the main system
connections = [
    Connection(VDP, Sco),
    # Connection(VDP[1], Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    dt=dt, 
    log=True, 
    Solver=GEAR52A, 
    tolerance_lte_abs=1e-5, 
    tolerance_lte_rel=1e-3, 
    tolerance_fpi=1e-8 
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run simulation for some number of seconds
    Sim.run(2*mu)

    Sco.plot(".-", lw=1.5)

    plt.show()