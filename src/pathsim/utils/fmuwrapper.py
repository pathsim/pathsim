#########################################################################################
##
##                   FMU WRAPPER - VERSION AGNOSTIC FMI INTERFACE
##                            (pathsim/utils/fmuwrapper.py)
##
##        Provides a unified interface for FMI 2.0 and FMI 3.0 FMUs, abstracting
##        away version-specific API differences for both Co-Simulation and
##        Model Exchange modes.
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import ctypes
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


# HELPER CLASSES ========================================================================

@dataclass
class EventInfo:
    """Unified event information structure for both FMI 2.0 and 3.0.

    Attributes
    ----------
    discrete_states_need_update : bool
        whether discrete state iteration is needed
    terminate_simulation : bool
        whether FMU requests simulation termination
    nominals_changed : bool
        whether nominal values of continuous states changed
    values_changed : bool
        whether continuous state values changed
    next_event_time_defined : bool
        whether FMU has scheduled a next time event
    next_event_time : float
        time of next scheduled event (if defined)
    """
    discrete_states_need_update: bool = False
    terminate_simulation: bool = False
    nominals_changed: bool = False
    values_changed: bool = False
    next_event_time_defined: bool = False
    next_event_time: float = 0.0


@dataclass
class StepResult:
    """Result information from a co-simulation step.

    Attributes
    ----------
    event_encountered : bool
        whether an event was encountered during step (FMI 3.0 only)
    terminate_simulation : bool
        whether FMU requests simulation termination (FMI 3.0 only)
    early_return : bool
        whether step returned early (FMI 3.0 only)
    last_successful_time : float
        last time successfully reached (FMI 3.0 only)
    """
    event_encountered: bool = False
    terminate_simulation: bool = False
    early_return: bool = False
    last_successful_time: float = 0.0


# MAIN WRAPPER CLASS ====================================================================

class FMUWrapper:
    """Version-agnostic wrapper for FMI 2.0 and 3.0 FMUs.

    This class provides a unified interface for working with FMUs regardless of
    FMI version (2.0 or 3.0) or interface type (Co-Simulation or Model Exchange).
    It handles all version-specific API differences internally and provides
    automatic ctypes conversion for seamless numpy array integration.

    Parameters
    ----------
    fmu_path : str
        path to the FMU file (.fmu)
    instance_name : str, optional
        name for the FMU instance (default: 'fmu_instance')
    mode : str, optional
        FMU interface mode: 'cosimulation' or 'model_exchange' (default: 'cosimulation')

    Attributes
    ----------
    fmu_path : str
        path to the FMU file
    instance_name : str
        name of the FMU instance
    mode : str
        interface mode ('cosimulation' or 'model_exchange')
    model_description : ModelDescription
        FMI model description from FMPy
    fmu : FMU2Slave | FMU3Slave | FMU2Model | FMU3Model
        underlying FMPy FMU instance
    fmi_version : str
        detected FMI version ('2.0' or '3.0')
    unzipdir : str
        directory where FMU was extracted
    n_states : int
        number of continuous states (Model Exchange only)
    n_event_indicators : int
        number of event indicators (Model Exchange only)
    variable_map : dict
        mapping from variable names to ModelVariable objects
    input_refs : dict
        mapping from input variable names to value references
    output_refs : dict
        mapping from output variable names to value references
    """

    def __init__(self, fmu_path, instance_name="fmu_instance", mode="cosimulation"):

        # Import FMPy (lazy import to avoid dependency if not used)
        try:
            from fmpy import read_model_description, extract
            from fmpy.fmi2 import FMU2Slave, FMU2Model
            from fmpy.fmi3 import FMU3Slave, FMU3Model
        except ImportError:
            raise ImportError("FMPy is required for FMU support. Install with: pip install fmpy")

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.mode = mode.lower()

        if self.mode not in ['cosimulation', 'model_exchange']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'cosimulation' or 'model_exchange'")

        # Read model description and detect FMI version
        self.model_description = read_model_description(fmu_path)
        self.fmi_version = self.model_description.fmiVersion

        # Extract FMU
        self.unzipdir = extract(fmu_path)

        # Build variable lookup maps
        self._build_variable_maps()

        # Get state and event info for Model Exchange
        if self.mode == 'model_exchange':
            self.n_states = self.model_description.numberOfContinuousStates
            self.n_event_indicators = self.model_description.numberOfEventIndicators
        else:
            self.n_states = 0
            self.n_event_indicators = 0

        # Instantiate appropriate FMU class based on version and mode
        if self.fmi_version.startswith('2.'):
            if self.mode == 'cosimulation':
                self.fmu = FMU2Slave(
                    guid=self.model_description.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                    instanceName=self.instance_name
                )
            else:  # model_exchange
                self.fmu = FMU2Model(
                    guid=self.model_description.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=self.model_description.modelExchange.modelIdentifier,
                    instanceName=self.instance_name
                )
        elif self.fmi_version.startswith('3.'):
            if self.mode == 'cosimulation':
                self.fmu = FMU3Slave(
                    guid=self.model_description.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                    instanceName=self.instance_name
                )
            else:  # model_exchange
                self.fmu = FMU3Model(
                    guid=self.model_description.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=self.model_description.modelExchange.modelIdentifier,
                    instanceName=self.instance_name
                )
        else:
            raise ValueError(f"Unsupported FMI version: {self.fmi_version}")


    def _build_variable_maps(self):
        """Build internal variable name to reference mappings."""
        self.variable_map = {var.name: var for var in self.model_description.modelVariables}
        self.input_refs = {}
        self.output_refs = {}

        for variable in self.model_description.modelVariables:
            if variable.causality == 'input':
                self.input_refs[variable.name] = variable.valueReference
            elif variable.causality == 'output':
                self.output_refs[variable.name] = variable.valueReference


    def instantiate(self, visible=False, logging_on=False):
        """Instantiate the FMU.

        Parameters
        ----------
        visible : bool, optional
            whether to show FMU GUI (if available)
        logging_on : bool, optional
            whether to enable FMU logging
        """
        self.fmu.instantiate(visible=visible, loggingOn=logging_on)


    def setup_experiment(self, tolerance=None, start_time=0.0, stop_time=None):
        """Setup experiment parameters.

        For FMI 2.0, this calls setupExperiment(). For FMI 3.0, parameters
        are stored and used in enterInitializationMode().

        Parameters
        ----------
        tolerance : float, optional
            tolerance for integration/event detection
        start_time : float, optional
            simulation start time
        stop_time : float, optional
            simulation stop time
        """
        self._tolerance = tolerance
        self._start_time = start_time
        self._stop_time = stop_time

        if self.fmi_version.startswith('2.'):
            self.fmu.setupExperiment(
                tolerance=tolerance,
                startTime=start_time,
                stopTime=stop_time
            )


    def enter_initialization_mode(self):
        """Enter initialization mode.

        For FMI 2.0, this just calls enterInitializationMode().
        For FMI 3.0, this passes the experiment parameters.
        """
        if self.fmi_version.startswith('2.'):
            self.fmu.enterInitializationMode()
        else:  # FMI 3.0
            self.fmu.enterInitializationMode(
                tolerance=self._tolerance,
                startTime=self._start_time,
                stopTime=self._stop_time
            )


    def exit_initialization_mode(self) -> Optional[EventInfo]:
        """Exit initialization mode and return event information.

        Returns
        -------
        event_info : EventInfo or None
            event information if Model Exchange with FMI 3.0, None for FMI 2.0 and Co-Simulation
        """
        result = self.fmu.exitInitializationMode()

        # For FMI 3.0 Model Exchange, this returns event info
        # For FMI 2.0, this just returns a status code (int), no event info yet
        if self.mode == 'model_exchange' and self.fmi_version.startswith('3.'):
            # Check if result is actually an event info structure (not just int)
            if hasattr(result, 'nextEventTimeDefined'):
                return EventInfo(
                    discrete_states_need_update=bool(getattr(result, 'discreteStatesNeedUpdate', False)),
                    terminate_simulation=bool(getattr(result, 'terminateSimulation', False)),
                    nominals_changed=bool(getattr(result, 'nominalsOfContinuousStatesChanged', False)),
                    values_changed=bool(getattr(result, 'valuesOfContinuousStatesChanged', False)),
                    next_event_time_defined=bool(getattr(result, 'nextEventTimeDefined', False)),
                    next_event_time=float(getattr(result, 'nextEventTime', 0.0))
                )

        # For FMI 2.0 Model Exchange, exitInitializationMode() doesn't return event info
        # Events are only discovered during simulation via event indicators or completedIntegratorStep
        return None


    def set_real(self, names_or_refs, values):
        """Set real-valued variables.

        Parameters
        ----------
        names_or_refs : list of str or list of int
            variable names or value references
        values : array_like
            values to set
        """
        # Convert names to references if needed
        if names_or_refs and isinstance(names_or_refs[0], str):
            refs = [self.variable_map[name].valueReference for name in names_or_refs]
        else:
            refs = names_or_refs

        # Convert to numpy array if needed
        values = np.atleast_1d(values)

        if self.fmi_version.startswith('2.'):
            self.fmu.setReal(refs, values)
        else:  # FMI 3.0
            self.fmu.setFloat64(refs, values)


    def get_real(self, names_or_refs):
        """Get real-valued variables.

        Parameters
        ----------
        names_or_refs : list of str or list of int
            variable names or value references

        Returns
        -------
        values : np.ndarray
            variable values
        """
        # Convert names to references if needed
        if names_or_refs and isinstance(names_or_refs[0], str):
            refs = [self.variable_map[name].valueReference for name in names_or_refs]
        else:
            refs = names_or_refs

        if self.fmi_version.startswith('2.'):
            return np.array(self.fmu.getReal(refs))
        else:  # FMI 3.0
            return np.array(self.fmu.getFloat64(refs))


    def set_variable(self, name, value):
        """Set a single variable by name (automatically detects type).

        Parameters
        ----------
        name : str
            variable name
        value : float, int, bool
            value to set
        """
        variable = self.variable_map.get(name)
        if variable is None:
            raise ValueError(f"Variable '{name}' not found in FMU")

        vr = variable.valueReference
        var_type = variable.type

        # Map type names between FMI 2.0 and 3.0
        if var_type in ['Real', 'Float64', 'Float32']:
            if self.fmi_version.startswith('2.'):
                self.fmu.setReal([vr], [float(value)])
            else:
                self.fmu.setFloat64([vr], [float(value)])
        elif var_type in ['Integer', 'Int64', 'Int32', 'Int16', 'Int8']:
            if self.fmi_version.startswith('2.'):
                self.fmu.setInteger([vr], [int(value)])
            else:
                self.fmu.setInt64([vr], [int(value)])
        elif var_type == 'Boolean':
            self.fmu.setBoolean([vr], [bool(value)])
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")


    def do_step(self, current_time, step_size) -> StepResult:
        """Perform a co-simulation step.

        Parameters
        ----------
        current_time : float
            current communication point
        step_size : float
            communication step size

        Returns
        -------
        result : StepResult
            step result information
        """
        if self.mode != 'cosimulation':
            raise RuntimeError("do_step() is only available for Co-Simulation FMUs")

        if self.fmi_version.startswith('2.'):
            # FMI 2.0 doStep returns nothing (throws on error)
            self.fmu.doStep(current_time, step_size)
            return StepResult()
        else:  # FMI 3.0
            # FMI 3.0 returns (eventEncountered, terminateSimulation, earlyReturn, lastSuccessfulTime)
            event, terminate, early, last_time = self.fmu.doStep(current_time, step_size)
            return StepResult(
                event_encountered=event,
                terminate_simulation=terminate,
                early_return=early,
                last_successful_time=last_time
            )


    def set_time(self, time):
        """Set current time (Model Exchange only).

        Parameters
        ----------
        time : float
            current time
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("set_time() is only available for Model Exchange FMUs")

        self.fmu.setTime(time)


    def set_continuous_states(self, states):
        """Set continuous states (Model Exchange only).

        Parameters
        ----------
        states : array_like
            continuous state vector
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("set_continuous_states() is only available for Model Exchange FMUs")

        if self.n_states == 0:
            return

        # Convert to ctypes array
        states = np.atleast_1d(states)
        x_ctypes = (ctypes.c_double * self.n_states)(*states)

        if self.fmi_version.startswith('2.'):
            self.fmu.setContinuousStates(x_ctypes, self.n_states)
        else:  # FMI 3.0
            self.fmu.setContinuousStates(x_ctypes, self.n_states)


    def get_continuous_states(self):
        """Get continuous states (Model Exchange only).

        Returns
        -------
        states : np.ndarray
            continuous state vector
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("get_continuous_states() is only available for Model Exchange FMUs")

        if self.n_states == 0:
            return np.array([])

        states = (ctypes.c_double * self.n_states)()
        self.fmu.getContinuousStates(states, self.n_states)
        return np.array(states)


    def get_derivatives(self):
        """Get state derivatives (Model Exchange only).

        Returns
        -------
        derivatives : np.ndarray
            state derivative vector
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("get_derivatives() is only available for Model Exchange FMUs")

        if self.n_states == 0:
            return np.array([])

        derivatives = (ctypes.c_double * self.n_states)()

        if self.fmi_version.startswith('2.'):
            self.fmu.getDerivatives(derivatives, self.n_states)
        else:  # FMI 3.0
            self.fmu.getContinuousStateDerivatives(derivatives, self.n_states)

        return np.array(derivatives)


    def get_event_indicators(self):
        """Get event indicators (Model Exchange only).

        Returns
        -------
        indicators : np.ndarray
            event indicator vector
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("get_event_indicators() is only available for Model Exchange FMUs")

        if self.n_event_indicators == 0:
            return np.array([])

        indicators = (ctypes.c_double * self.n_event_indicators)()
        self.fmu.getEventIndicators(indicators, self.n_event_indicators)
        return np.array(indicators)


    def enter_event_mode(self):
        """Enter event mode (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("enter_event_mode() is only available for Model Exchange FMUs")

        self.fmu.enterEventMode()


    def enter_continuous_time_mode(self):
        """Enter continuous time mode (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("enter_continuous_time_mode() is only available for Model Exchange FMUs")

        self.fmu.enterContinuousTimeMode()


    def update_discrete_states(self) -> EventInfo:
        """Update discrete states during event iteration (Model Exchange only).

        Returns
        -------
        event_info : EventInfo
            event information
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("update_discrete_states() is only available for Model Exchange FMUs")

        if self.fmi_version.startswith('2.'):
            # FMI 2.0: newDiscreteStates returns tuple with 6 elements
            result = self.fmu.newDiscreteStates()
            return EventInfo(
                discrete_states_need_update=result[0],
                terminate_simulation=result[1],
                nominals_changed=result[2],
                values_changed=result[3],
                next_event_time_defined=result[4],
                next_event_time=result[5]
            )
        else:  # FMI 3.0
            # FMI 3.0: updateDiscreteStates returns similar tuple
            result = self.fmu.updateDiscreteStates()
            return EventInfo(
                discrete_states_need_update=result[0],
                terminate_simulation=result[1],
                nominals_changed=result[2],
                values_changed=result[3],
                next_event_time_defined=result[4],
                next_event_time=result[5]
            )


    def completed_integrator_step(self) -> Tuple[bool, bool]:
        """Notify FMU that integrator step completed (Model Exchange only).

        Returns
        -------
        enter_event_mode : bool
            whether FMU requests event mode
        terminate_simulation : bool
            whether FMU requests simulation termination
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("completed_integrator_step() is only available for Model Exchange FMUs")

        return self.fmu.completedIntegratorStep()


    def reset(self):
        """Reset FMU to initial state."""
        self.fmu.reset()


    def terminate(self):
        """Terminate FMU."""
        self.fmu.terminate()


    def free_instance(self):
        """Free FMU instance and resources."""
        self.fmu.freeInstance()


    def __del__(self):
        """Cleanup FMU resources on deletion."""
        try:
            self.terminate()
            self.free_instance()
        except:
            pass
