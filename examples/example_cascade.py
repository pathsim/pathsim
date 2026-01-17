#########################################################################################
##
##                         PathSim Example for Cascade-controller 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection, Subsystem, Interface
from pathsim.blocks import Source, TransferFunctionZPG, Adder, Scope, PID, WhiteNoise

from pathsim.solvers import RKCK54


# SYSTEM SETUP AND SIMULATION ===========================================================

# define the plant as a subsystem -------------------------------------------------------

in1 = Interface()

p1 = TransferFunctionZPG(Zeros=[], Poles=[-1, -1, -1], Gain=10)
p2 = TransferFunctionZPG(Zeros=[], Poles=[-2], Gain=3)

a1 = Adder()
a2 = Adder()

d1 = WhiteNoise(standard_deviation=0.001, sampling_period=0.1)
d2 = WhiteNoise(standard_deviation=0.001, sampling_period=0.1)

plant = Subsystem(
    blocks=[p1, p2, a1, a2, d1, d2, in1],
    connections=[
        Connection(in1, p2),
        Connection(p2, a2[0]),
        Connection(d2, a2[1]),
        Connection(a2, p1, in1[1]),
        Connection(p1, a1[0]),
        Connection(d1, a1[1]),
        Connection(a1, in1[0])
        ]
    )


# define control loops ------------------------------------------------------------------

#source function
def f_s(t):
    if t>60: return 0.5
    elif t>20: return 1
    else: return 0 

stp = Source(f_s)

c1 = PID(Kp=0.015, Ki=0.015/0.716, Kd=0.0, f_max=10.0)
c2 = PID(Kp=0.244, Ki=0.244/0.134, Kd=0.0, f_max=10.0)

e1 = Adder("+-")
e2 = Adder("+-")

sc0 = Scope(labels=["setpoint", "plant 1", "plant 2"])
sc1 = Scope(labels=["err 1", "pid 1"])
sc2 = Scope(labels=["err 2", "pid 2"])

Sim = Simulation(
    blocks=[stp, plant, c1, c2, e1, e2, sc0, sc1, sc2],
    connections=[
        Connection(stp, e1[0], sc0[0]),
        Connection(plant[0], e1[1], sc0[1]),
        Connection(e1, c1, sc1[0]),
        Connection(c1, e2[0], sc1[1]),
        Connection(plant[1], e2[1], sc0[2]),
        Connection(e2, c2, sc2[0]),
        Connection(c2, plant, sc2[1])
        ],
    Solver=RKCK54,
    tolerance_lte_rel=1e-4,
    tolerance_lte_abs=1e-6
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation
    Sim.run(100)

    #quickly plot all scopes
    Sim.plot()

    plt.show()
