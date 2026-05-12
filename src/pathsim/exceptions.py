#########################################################################################
##
##                              PATHSIM EXCEPTIONS
##                               (exceptions.py)
##
##      This module defines custom exceptions for the PathSim simulation framework.
##
#########################################################################################


class StopSimulation(Exception):
    """Exception that can be raised by blocks or models to signal that the 
    simulation should stop immediately.

    This provides a clean mechanism for user-defined stopping conditions,
    such as reaching a target state, detecting a fault, or satisfying 
    a convergence criterion.

    When raised inside a block's update, sample, or any other method 
    called during the simulation loop, the 'Simulation' class will catch 
    it gracefully and terminate the run as if 'stop()' had been called.

    Parameters
    ----------
    message : str
        optional message describing the stopping condition

    Example
    -------

    Raise from inside a block to stop the simulation:

    .. code-block:: python

        from pathsim.exceptions import StopSimulation
        from pathsim.blocks import Function

        def check(x):
            if x > 10.0:
                raise StopSimulation(f"value exceeded threshold: {x:.4f}")
            return x

        blk = Function(check)
    """
