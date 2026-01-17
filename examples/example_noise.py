#########################################################################################
##
##               PathSim Example for Noise Sources from the RF toolbox
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

#the standard blocks are imported like this
from pathsim.blocks import (
    Amplifier, 
    Adder, 
    Scope, 
    Spectrum,
    ButterworthBandpassFilter,
    SquareWaveSource, 
    WhiteNoise,
    PinkNoise
    )


# FILTERING A SQUAREWAVE ================================================================

#simulation timestep
dt = 0.02

#fundamental frequency of square wave
f = 0.5


#blocks that define the system
Src = SquareWaveSource(frequency=f)
Ns1 = PinkNoise(standard_deviation=0.1, sampling_period=3*dt)
Ns2 = WhiteNoise(standard_deviation=0.07, sampling_period=3*dt)
FLT = ButterworthBandpassFilter((f-f/10, f+f/10), 4)
Add = Adder()
Sco = Scope(
    labels=[
        "squarewave", 
        "filter", 
        "adder", 
        "pink noise", 
        "white noise"
        ]
    )
Spc = Spectrum(
    freq=np.linspace(0, 5, 500), 
    labels=[
        "squarewave", 
        "filter", 
        "adder", 
        "pink noise", 
        "white noise"
        ]
    )

blocks = [Src, Ns1, Ns2, Add, FLT, Sco, Spc]

#the connections between the blocks
connections = [
    Connection(Src, Add, Sco, Spc),
    Connection(Ns1, Add[1], Sco[3], Spc[3]),
    Connection(Ns2, Add[2], Sco[4], Spc[4]),
    Connection(Add, FLT, Sco[2], Spc[2]),
    Connection(FLT, Sco[1], Spc[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(100/f)

    Sco.plot()

    Spc.plot()
    Spc.ax.set_yscale("log")

    plt.show()