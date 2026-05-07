#########################################################################################
##
##                         EVENT MANAGER CLASS FOR EVENT DETECTION
##                                   (events/_event.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .. _constants import EVT_TOLERANCE


# EVENT MANAGER CLASS ===================================================================

class Event:
    """This is the base class of the event handling system.
    
    Monitors system state by evaluating an event function (func_evt) with scalar output.
    
    .. code-block::

        func_evt(time) -> event?

    If an event is detected, some action (func_act) is performed on the states of the blocks.

    .. code-block::

        func_evt(time) == True -> event -> func_act(time)

    The methods are structured such that event detection can be separated from event 
    resolution. This is required for adaptive timestep solvers to approach the event 
    and only resolve it when the event tolerance ('tolerance') is satisfied.

    If no action function (func_act) is specified, the event will only be detected but other 
    than that, no action will be triggered. For general state monitoring.    

    Parameters
    ----------
    func_evt : callable
        event function, where zeros are events
    func_act : callable
        action function for event resolution 
    tolerance : float
        tolerance to check if detection is close to actual event

    Attributes
    ----------
    _history : tuple[None, float], tuple[float, float], tuple[bool, float]
        history of event function evaluation after buffering
    _times : list[float]
        tracking the event times
    _active : bool
        flag that sets event active or inactive
    """

    def __init__(
        self, 
        func_evt=None, 
        func_act=None, 
        tolerance=EVT_TOLERANCE
        ):
            
        #event detection function
        self.func_evt = func_evt

        #event action function -> event resolution (must not be callable)
        self.func_act = func_act

        #tolerance for checking if close to actual event
        self.tolerance = tolerance

        #event function evaluation and evaluation time history (eval, time)
        self._history = None, 0.0

        #recording the event times
        self._times = []

        #flag for active event checking
        self._active = True


    def __len__(self):
        """
        Return the number of detected (or rather resolved) events.
        
        Returns
        -------
        length : int
            number of events detected

        """
        return len(self._times)


    def __iter__(self):
        """
        Yields the recorded times at which events are detected.
        """
        for t in self._times:
            yield t


    def __bool__(self):
        return self._active


    # external methods ------------------------------------------------------------------

    def on(self): self._active = True
    def off(self): self._active = False


    def reset(self):
        """
        Reset the recorded event times. Resetting the history is not 
        required because of the 'buffer' method. Reactivates event tracking.
        """
        self._history = None, 0.0
        self._times = []
        self._active = True


    def buffer(self, t):
        """Buffer the event function evaluation before the timestep is 
        taken and the evaluation time. 
        
        Parameters
        ----------
        t : float
            evaluation time for buffering history
        """
        if self.func_evt is not None:
            self._history = self.func_evt(t), t


    def estimate(self, t):
        """Estimate the time of the next event, based on history or internal schedule.

        This improves simulation performance by estimating events before the simulation 
        step such that fewer steps have to be rejected for event location. 
              
        Parameters
        ----------
        t : float 
            evaluation time for estimation 
        
        Returns
        -------
        float | None
            estimated time until next event
        """
        return None


    def detect(self, t):
        """Evaluate the event function and decide if an event has occurred. 
        Can also use the history of the event function evaluation from 
        before the timestep.

        Notes
        -----
        This does nothing and needs to be implemented for specific events!!!
    
        Parameters
        ----------
        t : float
            evaluation time for detection 
        
        Returns
        -------
        detected : bool
            was an event detected?
        close : bool
            are we close to the event?
        ratio : float
            interpolated event location as ratio of timestep
        """

        return False, False, 1.0
        

    def resolve(self, t):
        """Resolve the event and record the time (t) at which it occurs. 
        
        Resolves event using the action function (func_act) if it is defined. 

        Otherwise this just marks the location of the event in time.

        Parameters
        ----------
        t : float
            evaluation time for event resolution 
        """

        #save the time of event resolution
        self._times.append(t)

        #action function for event resolution
        if self.func_act is not None:
            self.func_act(t)


    # checkpoint methods ----------------------------------------------------------------

    def to_checkpoint(self, prefix):
        """Serialize event state for checkpointing.

        Parameters
        ----------
        prefix : str
            key prefix for NPZ arrays (assigned by simulation)

        Returns
        -------
        json_data : dict
            JSON-serializable metadata
        npz_data : dict
            numpy arrays keyed by path
        """
        #extract history eval value
        hist_eval, hist_time = self._history
        if hist_eval is not None and hasattr(hist_eval, 'item'):
            hist_eval = float(hist_eval)

        json_data = {
            "type": self.__class__.__name__,
            "active": self._active,
            "history_eval": hist_eval,
            "history_time": hist_time,
        }

        npz_data = {}
        if self._times:
            npz_data[f"{prefix}/times"] = np.array(self._times)

        return json_data, npz_data


    def load_checkpoint(self, prefix, json_data, npz):
        """Restore event state from checkpoint.

        Parameters
        ----------
        prefix : str
            key prefix for NPZ arrays (assigned by simulation)
        json_data : dict
            event metadata from checkpoint JSON
        npz : dict-like
            numpy arrays from checkpoint NPZ
        """
        self._active = json_data["active"]
        self._history = json_data["history_eval"], json_data["history_time"]

        times_key = f"{prefix}/times"
        if times_key in npz:
            self._times = npz[times_key].tolist()
        else:
            self._times = []
