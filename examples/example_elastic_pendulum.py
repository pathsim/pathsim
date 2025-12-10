#########################################################################################
##
##                PathSim example of a damped elastic pendulum
##                          Using coupled ODE blocks
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Function, Scope
from pathsim.solvers import RKBS32


# DAMPED ELASTIC PENDULUM ===============================================================

# Initial conditions
r0, vr0 = 2, 0.0
phi0, omega0 = 0.3*np.pi, 0.0

# Physical parameters
g = 9.81          # gravity [m/s^2]
l0 = 1.0          # natural spring length [m]
k = 50.0          # spring constant [N/m]
m = 1.0           # mass [kg]
c_r = 0.3         # radial damping [kg/s]
c_phi = 0.1       # angular damping [N m s]


# Define the radial ODE (spring-mass-damper with coupling terms)
def rad_ode(x, u, t):
    r, vr = x
    omega, phi = u
        
    # radial acceleration terms
    centrifugal = r * omega**2
    spring = -(k/m) * (r - l0)
    gravity_rad = g * np.cos(phi)
    damping = -(c_r/m) * vr
    
    accel_r = centrifugal + spring + gravity_rad + damping
    
    return np.array([vr, accel_r])


# Define the angular ODE (pendulum with coupling terms)
def ang_ode(x, u, t):
    phi, omega = x
    r, vr = u
    
    # angular acceleration terms
    gravity_torque = -(g / r) * np.sin(phi)
    coriolis = -(2 / r) * vr * omega
    damping = -(c_phi / (m * r**2)) * omega
    
    accel_phi = gravity_torque + coriolis + damping
    
    return np.array([omega, accel_phi])


# Create the ODE blocks
rad = ODE(rad_ode, np.array([r0, vr0]))
ang = ODE(ang_ode, np.array([phi0, omega0]))

# Cartesian conversion
@Function
def crt(r, phi):
    return r*np.sin(phi), -r*np.cos(phi)

# Scope for visualization
sc1 = Scope(labels=["r [m]", "vr [m/s]", "phi [rad]", "omega [rad/s]"])
sc2 = Scope(labels=["x [m]", "y [m]"], sampling_rate=0.005)

blocks = [rad, ang, crt, sc1, sc2]

# Connect the coupled system
connections = [
    Connection(ang[1], rad[0]),           # omega -> rad input 0
    Connection(ang[0], rad[1], crt[1]),   # phi -> rad input 1
    Connection(rad[0], ang[0], crt[0]),   # r -> ang input 0
    Connection(rad[1], ang[1]),           # vr -> ang input 1
    Connection(rad[:2], sc1[:2]),         # r, vr -> scope
    Connection(ang[:2], sc1[2:4]),        # phi, omega -> scope
    Connection(crt[:2], sc2[:2])
    ]

# Simulation instance
Sim = Simulation(
    blocks,
    connections,
    Solver=RKBS32
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run the simulation
    Sim.run(10)

    # Plot state variables
    sc1.plot()

    fig, ax = sc2.plot2D()
    ax.plot(0, 0, "o", c="k")
    ax.set_aspect(1)

    # Read the data from scope
    t, (x, y) = sc2.read()

    plt.show()