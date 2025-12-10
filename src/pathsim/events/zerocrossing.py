#########################################################################################
##
##                                  ZERO CROSSING EVENTS
##                                (events/zerocrossing.py)
##
##                                    Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._event import Event

from .. _constants import TOLERANCE


# EVENT MANAGER CLASS ===================================================================

class ZeroCrossing(Event):
    """Subclass of base 'Event' that triggers if the event function crosses zero. 
    This is a bidirectional zero-crossing detector. 
    
    Monitors system state by evaluating an event function (func_evt) with scalar output and 
    testing for zero crossings (sign changes). 
    
    .. code-block::

        func_evt(time) -> event?

    If an event is detected, some action (func_act) is performed on the system state.

    .. code-block::

        func_evt(time) == 0 -> event -> func_act(time)

    Example
    -------
    Initialize a zero-crossing event handler like this:

    .. code-block:: python

        # define the event function
        def evt(t):
            # here we have a zero-crossing at 't==10'
            return t - 10
        
        # define the action function (callback)
        def act(t):
            # do something at event resolution
            pass
    
        # initialize the event manager
        E = ZeroCrossing(
            func_evt=evt,  # the event function
            func_act=act   # the action function
            )    
    
    Parameters
    ----------
    func_evt : callable
        event function, where zeros are events
    func_act : callable
        action function for event resolution 
    tolerance : float
        tolerance to check if detection is close to actual event
    """

    def detect(self, t):
        """Evaluate the event function and check for zero-crossings
        
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

        # unpack history
        _result, _t = self._history

        # no history -> no zero crossing
        if _result is None:
            return False, False, 1.0

        # evaluate event function
        result = self.func_evt(t)

        # exactly hit zero -> quit early
        if result == 0.0 and _result != 0.0:
            return True, True, 1.0

        # are we close to the actual event?
        close = abs(result) <= self.tolerance

        # check for zero crossing (sign change, + to - and - to +)
        is_event = np.sign(result * _result) < 0

        # definitely no event detected -> quit early
        if not is_event:
            return False, False, 1.0

        # linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(_result) / np.clip(abs(_result - result), TOLERANCE, None)

        # convert to scalar if needed to avoid numpy deprecation warning
        if isinstance(ratio, np.ndarray):
            ratio = ratio.item()

        return True, close, float(ratio)


class ZeroCrossingUp(Event):
    """Modification of standard 'ZeroCrossing' event where events are only triggered 
    if the event function changes sign from negative to positive (up). Also called
    unidirectional zero-crossing.
    """

    def detect(self, t):
        """Evaluate the event function and check for zero-crossings

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
            
        # evaluate event function
        result = self.func_evt(t)
            
        # unpack history
        _result, _t = self._history
        
        # no history -> no zero crossing
        if _result is None:
            return False, False, 1.0

        # exactly hit zero -> quit early
        if result == 0.0 and _result < 0.0:
            return True, True, 1.0

        # are we close to the actual event?
        close = abs(result) <= self.tolerance

        # check for zero crossing (sign change, negative to positive)
        is_event = np.sign(result * _result) < 0 and result > _result

        # no event detected or wrong direction -> quit early
        if not is_event or _result >= 0:
            return False, False, 1.0

        # linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(_result) / np.clip(abs(_result - result), TOLERANCE, None)

        # convert to scalar if needed to avoid numpy deprecation warning
        if isinstance(ratio, np.ndarray):
            ratio = ratio.item()

        return True, close, float(ratio)


class ZeroCrossingDown(Event):
    """Modification of standard 'ZeroCrossing' event where events are only triggered 
    if the event function changes sign from positive to negative (down). Also called
    unidirectional zero-crossing.
    """

    def detect(self, t):
        """Evaluate the event function and check for zero-crossings
        
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
        
        # evaluate event function
        result = self.func_evt(t)
            
        # unpack history
        _result, _t = self._history

        # no history -> no zero crossing
        if _result is None:
            return False, False, 1.0
        
        # exactly hit zero -> quit early
        if result == 0.0 and _result > 0.0:
            return True, True, 1.0

        # are we close to the actual event?
        close = abs(result) <= self.tolerance

        # check for zero crossing (sign change, positive to negative)
        is_event = np.sign(result * _result) < 0 and result < _result

        # no event detected or wrong direction -> quit early
        if not is_event or _result <= 0:
            return False, False, 1.0

        # linear interpolation to find event time ratio (secant crosses x-axis)
        ratio = abs(_result) / np.clip(abs(_result - result), TOLERANCE, None)

        # convert to scalar if needed to avoid numpy deprecation warning
        if isinstance(ratio, np.ndarray):
            ratio = ratio.item()

        return True, close, float(ratio)