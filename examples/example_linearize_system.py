#########################################################################################
##
##          PathSim Example: Global Linearization of a PID + Nonlinear Plant
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Adder, DynamicalSystem, StateSpace, Scope
from pathsim.blocks.ctrl import PID


# PID + NONLINEAR PLANT CLOSED LOOP ======================================================

#nonlinear plant: dx/dt = -x^2 + u,  y = x
ref  = Constant(1.0)
err  = Adder("+-")
ctrl = PID(Kp=1.0, Ki=2.0, Kd=0.0, f_max=50)
plant = DynamicalSystem(
    func_dyn=lambda x, u, t: -x**2 + u,
    func_alg=lambda x, u, t: x,
    initial_value=1.0,
    jac_dyn=lambda x, u, t: -2*x
    )
Sco = Scope(labels=["y"])

blocks = [ref, err, ctrl, plant, Sco]

connections = [
    Connection(ref, err[0]),      # reference -> error junction
    Connection(plant, err[1]),    # feedback  -> error junction
    Connection(err, ctrl),        # error     -> controller
    Connection(ctrl, plant),      # controller-> plant
    Connection(plant, Sco),
    ]

Sim = Simulation(blocks, connections, dt=0.01, log=False)


# Run Example ===========================================================================

if __name__ == "__main__":

    #find the closed-loop operating point (Ki != 0 -> y settles at the reference)
    Sim.trim(targets=[], free=[])
    y0 = plant.outputs[0]
    print(f"operating point: y0 = {y0:.4f}")

    #assemble the global closed-loop state-space model, breaking the loop
    #at the reference input and tapping the plant output
    result = Sim.linearize_system(inputs=[err[0]], outputs=[plant[0]], as_block=False)
    A, B, C, D = result.A, result.B, result.C, result.D
    print(f"state-space shapes: A{A.shape}  B{B.shape}  C{C.shape}  D{D.shape}")
    print(f"state labels : {result.state_labels}")
    print(f"input labels : {result.input_labels}")
    print(f"output labels: {result.output_labels}")

    poles = np.linalg.eigvals(A)
    print("closed-loop poles:", np.round(poles, 4))
    print(f"stable (all Re(poles) < 0): {np.all(poles.real < 0)}")

    #'linearize_system' leaves the touched blocks individually linearized as
    #a side effect (same as calling 'Sim.linearize()' would) -- revert them
    #so the comparison below runs the *true* nonlinear closed loop
    Sim.delinearize()

    #save the trim point so each step-response trial can restart from it
    plant_state0, ctrl_state0 = plant.state.copy(), ctrl.state.copy()

    def step_response(delta, duration=3.0):
        """Apply a reference step of size 'delta' to the nonlinear closed
        loop and to the assembled linear StateSpace model, both starting
        from the same trimmed operating point, and return both trajectories.
        """
        plant.state, ctrl.state = plant_state0.copy(), ctrl_state0.copy()
        Sco.reset()  # clear the recording, but keep the states set above
        ref.value = 1.0 + delta
        Sim.run(duration=duration, reset=False)
        t_nl, (y_nl,) = Sco.read()
        t_nl = t_nl - t_nl[0]
        ref.value = 1.0

        step_src = Constant(delta)
        ss_block = StateSpace(A=A, B=B, C=C, D=D, initial_value=np.zeros(A.shape[0]))
        Sco_lin = Scope(labels=["dy"])
        Sim_lin = Simulation(
            blocks=[step_src, ss_block, Sco_lin],
            connections=[Connection(step_src, ss_block), Connection(ss_block, Sco_lin)],
            dt=0.01,
            log=False
            )
        Sim_lin.run(duration=duration, reset=False)
        _, (dy,) = Sco_lin.read()

        return t_nl, y_nl, y0 + dy

    #small step: linearization should hold up closely
    t_small, y_nl_small, y_lin_small = step_response(delta=0.05)

    #large step: the plant's nonlinearity (-x^2) should make the linear
    #model visibly diverge from the true nonlinear response
    t_large, y_nl_large, y_lin_large = step_response(delta=1.5)

    print(f"max error, small step (0.05): {np.max(np.abs(y_nl_small - y_lin_small)):.2e}")
    print(f"max error, large step (1.5) : {np.max(np.abs(y_nl_large - y_lin_large)):.2e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), tight_layout=True)

    ax1.plot(t_small, y_nl_small, label="true nonlinear closed loop", lw=2)
    ax1.plot(t_small, y_lin_small, "--", label="linearize_system() model", lw=2)
    ax1.set_title("Small reference step (0.05): linear model matches closely")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_large, y_nl_large, label="true nonlinear closed loop", lw=2)
    ax2.plot(t_large, y_lin_large, "--", label="linearize_system() model", lw=2)
    ax2.set_title("Large reference step (1.5): linear model visibly diverges")
    ax2.set_xlabel("time [s]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.show()
