#########################################################################################
##
##                           FUNCTIONAL MOCK-UP UNIT (FMU) BLOCKS
##                                   (pathsim/blocks/fmu.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block
from .dynsys import DynamicalSystem

from ..events.schedule import Schedule, ScheduleList
from ..events.zerocrossing import ZeroCrossing


# BLOCKS ================================================================================

class CoSimulationFMU(Block):
    """Co-Simulation FMU block using FMPy with support for FMI 2.0 and FMI 3.0.

    This block wraps an FMU (Functional Mock-up Unit) for co-simulation. 
    The FMU encapsulates a simulation model that can be executed independently 
    and synchronized with the main simulation.

    Parameters
    ----------
    fmu_path : str
        path to the FMU file (.fmu)
    instance_name : str, optional
        name for the FMU instance (default: 'fmu_instance')
    start_values : dict, optional
        dictionary of variable names and their initial values
    dt : float, optional
        communication step size for co-simulation. If None, uses the FMU's 
        default experiment step size if available.

    Attributes
    ----------
    fmu_path : str
        path to the FMU file
    instance_name : str
        name of the FMU instance
    start_values : dict
        initial values for FMU variables
    model_description : ModelDescription
        FMI model description from FMPy
    fmu : FMU2Slave or FMU3Slave
        FMPy FMU instance for co-simulation
    fmi_version : str
        FMI version ('2.0' or '3.0')
    _input_refs : dict
        reference map for input variables
    _output_refs : dict
        reference map for output variables
    
    FMU Capabilities
    ----------------
    can_interpolate_inputs : bool
        whether FMU can interpolate inputs between communication points
    can_handle_variable_step : bool
        whether FMU supports variable communication step sizes
    default_step_size : float or None
        recommended default step size from FMU
    max_output_derivative_order : int
        maximum order of output derivatives available
    """

    #max number of ports (will be configured based on FMU)
    _n_in_max = None
    _n_out_max = None

    #maps for input and output port labels
    _port_map_in = {}
    _port_map_out = {}

    def __init__(self, fmu_path, instance_name="fmu_instance", start_values=None, 
                 dt=None, verbose=False):

        # Import FMPy here to avoid requiring it as a dependency if not used
        try:
            from fmpy import read_model_description, extract
            from fmpy.fmi2 import FMU2Slave
            from fmpy.fmi3 import FMU3Slave
        except ImportError:
            raise ImportError("FMPy is required for FMU blocks. Install with: pip install fmpy")

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.start_values = start_values if start_values is not None else {}
        self.verbose = verbose

        # Read model description
        self.model_description = read_model_description(fmu_path)
        
        # Detect FMI version
        self.fmi_version = self.model_description.fmiVersion
        
        # Extract FMU
        self.unzipdir = extract(fmu_path)

        # Extract metadata and capabilities
        self._extract_fmu_metadata()
        
        # Use provided dt or fall back to FMU default
        if dt is None:
            if self.default_step_size is not None:
                self.dt = self.default_step_size
            else:
                raise ValueError("No step size provided and FMU has no default experiment step size")
        else:
            self.dt = dt

        # Get input and output variable references
        self._input_refs = {}
        self._output_refs = {}

        for variable in self.model_description.modelVariables:
            if variable.causality == 'input':
                self._input_refs[variable.name] = variable.valueReference
                self._port_map_in[variable.name] = len(self._input_refs) - 1
            elif variable.causality == 'output':
                self._output_refs[variable.name] = variable.valueReference
                self._port_map_out[variable.name] = len(self._output_refs) - 1

        # Initialize base class with proper port configuration
        super().__init__()

        # Instantiate FMU based on version
        if self.fmi_version.startswith('2.'):
            self.fmu = FMU2Slave(
                guid=self.model_description.guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName=self.instance_name
            )
        elif self.fmi_version.startswith('3.'):
            self.fmu = FMU3Slave(
                guid=self.model_description.guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                instanceName=self.instance_name
            )
        else:
            raise ValueError(f"Unsupported FMI version: {self.fmi_version}")

        # Setup experiment
        self.fmu.instantiate()
        
        # FMI 3.0 has different initialization sequence
        if self.fmi_version.startswith('3.'):
            self.fmu.enterInitializationMode(
                tolerance=None,
                startTime=0.0,
                stopTime=None
            )
        else:
            self.fmu.setupExperiment(startTime=0.0)
            self.fmu.enterInitializationMode()

        # Set start values
        self._set_start_values()

        # Exit initialization mode
        self.fmu.exitInitializationMode()

        # Internal scheduled event function
        self.events = [
            Schedule(
                t_start=0,
                t_period=self.dt,
                func_act=self._step_fmu
            )
        ]

        # Read initial outputs
        self._update_outputs_from_fmu()
        

    def _extract_fmu_metadata(self):
        """Extract metadata and capabilities from FMU."""
        
        cs = self.model_description.coSimulation
        
        if cs is None:
            raise ValueError("FMU does not support Co-Simulation")
        
        # Extract capabilities (consistent across FMI 2.0 and 3.0)
        self.can_interpolate_inputs = getattr(cs, 'canInterpolateInputs', False)
        self.can_handle_variable_step = getattr(cs, 'canHandleVariableCommunicationStepSize', False)
        self.max_output_derivative_order = getattr(cs, 'maxOutputDerivativeOrder', 0)
        
        # FMI 3.0 specific capabilities
        if self.fmi_version.startswith('3.'):
            self.can_return_early = getattr(cs, 'canReturnEarlyAfterIntermediateUpdate', False)
            self.has_event_mode = getattr(cs, 'hasEventMode', False)
            self.recommended_intermediate_input_smoothness = getattr(cs, 'recommendedIntermediateInputSmoothness', 0)
        else:
            self.can_return_early = False
            self.has_event_mode = False
            self.recommended_intermediate_input_smoothness = 0
        
        # Extract default experiment settings
        default_experiment = self.model_description.defaultExperiment
        
        if default_experiment is not None:
            self.default_start_time = getattr(default_experiment, 'startTime', 0.0)
            self.default_stop_time = getattr(default_experiment, 'stopTime', None)
            self.default_step_size = getattr(default_experiment, 'stepSize', None)
            self.default_tolerance = getattr(default_experiment, 'tolerance', None)
        else:
            self.default_start_time = 0.0
            self.default_stop_time = None
            self.default_step_size = None
            self.default_tolerance = None
        
        # Model metadata
        self.model_name = self.model_description.modelName
        self.generation_tool = getattr(self.model_description, 'generationTool', 'Unknown')
        self.generation_date = getattr(self.model_description, 'generationDateAndTime', 'Unknown')
        self.description = getattr(self.model_description, 'description', '')
        self.author = getattr(self.model_description, 'author', 'Unknown')
        self.version = getattr(self.model_description, 'version', 'Unknown')


    def _set_start_values(self):
        """Set initial values for FMU variables."""
        for name, value in self.start_values.items():
            for variable in self.model_description.modelVariables:
                if variable.name == name:
                    if variable.type in ['Real', 'Float64', 'Float32']:  # FMI 3.0 uses Float64/Float32
                        self.fmu.setReal([variable.valueReference], [value])
                    elif variable.type in ['Integer', 'Int64', 'Int32', 'Int16', 'Int8']:
                        self.fmu.setInteger([variable.valueReference], [int(value)])
                    elif variable.type == 'Boolean':
                        self.fmu.setBoolean([variable.valueReference], [bool(value)])


    def _step_fmu(self, t):
        """Perform one FMU co-simulation step"""
        self._update_fmu_from_inputs()    

        # Perform co-simulation step
        self.fmu.doStep(
            currentCommunicationPoint=t, 
            communicationStepSize=self.dt
        )  

        self._update_outputs_from_fmu()


    def _update_fmu_from_inputs(self):
        """Read block inputs and update FMU inputs."""
        if len(self._input_refs) > 0:
            input_vrefs = list(self._input_refs.values())
            self.fmu.setReal(input_vrefs, self.inputs.to_array())


    def _update_outputs_from_fmu(self):
        """Read outputs from FMU and update block outputs."""
        if len(self._output_refs) > 0:
            output_vrefs = list(self._output_refs.values())
            self.outputs.update_from_array(self.fmu.getReal(output_vrefs))


    def update(self, t):
        """Update FMU inputs/outputs between scheduled steps if interpolation supported."""
        if self.can_interpolate_inputs:
            self._update_fmu_from_inputs()
            self._update_outputs_from_fmu()


    def reset(self):
        """Reset the FMU instance."""
        super().reset()
        self.fmu.reset()
        
        if self.fmi_version.startswith('3.'):
            self.fmu.enterInitializationMode(
                tolerance=None,
                startTime=0.0,
                stopTime=None
            )
        else:
            self.fmu.enterInitializationMode()
            
        self.fmu.exitInitializationMode()
        self._update_outputs_from_fmu()


    def __len__(self):
        """FMU is a discrete time source-like block without direct passthrough"""
        return 0


    def __del__(self):
        """Cleanup FMU resources."""
        try:
            self.fmu.terminate()
            self.fmu.freeInstance()
        except:
            pass


class ModelExchangeFMU(DynamicalSystem):
    """Model Exchange FMU block using FMPy with support for FMI 2.0 and FMI 3.0.

    This block wraps an FMU (Functional Mock-up Unit) for model exchange.
    The FMU provides the right-hand side of an ODE system that is integrated
    by PathSim's numerical solvers. Internal FMU events (state events, time
    events, and step completion events) are translated to PathSim events.

    Parameters
    ----------
    fmu_path : str
        path to the FMU file (.fmu)
    instance_name : str, optional
        name for the FMU instance (default: 'fmu_instance')
    start_values : dict, optional
        dictionary of variable names and their initial values
    tolerance : float, optional
        tolerance for event detection (default: 1e-10)
    verbose : bool, optional
        enable verbose output (default: False)

    Attributes
    ----------
    fmu_path : str
        path to the FMU file
    instance_name : str
        name of the FMU instance
    start_values : dict
        initial values for FMU variables
    model_description : ModelDescription
        FMI model description from FMPy
    fmu : FMU2Model or FMU3Model
        FMPy FMU instance for model exchange
    fmi_version : str
        FMI version ('2.0' or '3.0')
    _input_refs : dict
        reference map for input variables
    _output_refs : dict
        reference map for output variables
    n_states : int
        number of continuous states
    n_event_indicators : int
        number of event indicators
    time_event : ScheduleList or None
        dynamic time event for FMU-scheduled events
    """

    #max number of ports (will be configured based on FMU)
    _n_in_max = None
    _n_out_max = None

    #maps for input and output port labels
    _port_map_in = {}
    _port_map_out = {}

    def __init__(self, fmu_path, instance_name="fmu_instance", start_values=None,
                 tolerance=1e-10, verbose=False):

        # Import FMPy here to avoid requiring it as a dependency if not used
        try:
            from fmpy import read_model_description, extract
            from fmpy.fmi2 import FMU2Model
            from fmpy.fmi3 import FMU3Model
        except ImportError:
            raise ImportError("FMPy is required for FMU blocks. Install with: pip install fmpy")

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.start_values = start_values if start_values is not None else {}
        self.verbose = verbose
        self.tolerance = tolerance

        # Read model description
        self.model_description = read_model_description(fmu_path)

        # Detect FMI version
        self.fmi_version = self.model_description.fmiVersion

        # Extract FMU
        self.unzipdir = extract(fmu_path)

        # Extract metadata and capabilities
        self._extract_fmu_metadata()

        # Get input and output variable references
        self._input_refs = {}
        self._output_refs = {}

        for variable in self.model_description.modelVariables:
            if variable.causality == 'input':
                self._input_refs[variable.name] = variable.valueReference
                self._port_map_in[variable.name] = len(self._input_refs) - 1
            elif variable.causality == 'output':
                self._output_refs[variable.name] = variable.valueReference
                self._port_map_out[variable.name] = len(self._output_refs) - 1

        # Get continuous states and derivatives
        self.n_states = self.model_description.numberOfContinuousStates
        self.n_event_indicators = self.model_description.numberOfEventIndicators

        # Instantiate FMU based on version
        if self.fmi_version.startswith('2.'):
            self.fmu = FMU2Model(
                guid=self.model_description.guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=self.model_description.modelExchange.modelIdentifier,
                instanceName=self.instance_name
            )
        elif self.fmi_version.startswith('3.'):
            self.fmu = FMU3Model(
                guid=self.model_description.guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=self.model_description.modelExchange.modelIdentifier,
                instanceName=self.instance_name
            )
        else:
            raise ValueError(f"Unsupported FMI version: {self.fmi_version}")

        # Setup FMU
        self.fmu.instantiate()

        # FMI 3.0 has different initialization sequence
        if self.fmi_version.startswith('3.'):
            self.fmu.enterInitializationMode(
                tolerance=self.tolerance,
                startTime=0.0,
                stopTime=None
            )
        else:
            self.fmu.setupExperiment(tolerance=self.tolerance, startTime=0.0)
            self.fmu.enterInitializationMode()

        # Set start values
        self._set_start_values()

        # Exit initialization mode and check for initial events
        event_info = self.fmu.exitInitializationMode()

        # Store initial time event if defined
        self._initial_time_event = (
            event_info.nextEventTime
            if event_info and getattr(event_info, 'nextEventTimeDefined', False)
            else None
        )

        # Enter continuous time mode after initialization
        self.fmu.enterContinuousTimeMode()

        # Get initial continuous states
        initial_states = self._get_continuous_states()

        # Initialize parent DynamicalSystem with FMU dynamics
        super().__init__(
            func_dyn=self._get_derivatives,
            func_alg=self._get_outputs,
            initial_value=initial_states,
            jac_dyn=None  # Could add Jacobian support later via directional derivatives
        )

        # Initialize time event manager (will be populated dynamically)
        self.time_event = None

        # Create state event (zero-crossing) events for each event indicator
        for i in range(self.n_event_indicators):
            event = ZeroCrossing(
                func_evt=lambda t, idx=i: self._get_event_indicator(idx),
                func_act=self._handle_event,
                tolerance=self.tolerance
            )
            self.events.append(event)

        # Schedule initial time event if any
        if self._initial_time_event is not None:
            self._update_time_events(self._initial_time_event)


    def _extract_fmu_metadata(self):
        """Extract metadata and capabilities from FMU."""

        me = self.model_description.modelExchange

        if me is None:
            raise ValueError("FMU does not support Model Exchange")

        # Extract capabilities
        self.provides_directional_derivative = getattr(me, 'providesDirectionalDerivative', False)
        self.completed_integrator_step_not_needed = getattr(me, 'completedIntegratorStepNotNeeded', False)

        # Model metadata
        self.model_name = self.model_description.modelName
        self.generation_tool = getattr(self.model_description, 'generationTool', 'Unknown')
        self.generation_date = getattr(self.model_description, 'generationDateAndTime', 'Unknown')
        self.description = getattr(self.model_description, 'description', '')
        self.author = getattr(self.model_description, 'author', 'Unknown')
        self.version = getattr(self.model_description, 'version', 'Unknown')


    def _set_start_values(self):
        """Set initial values for FMU variables."""
        if not self.start_values:
            return

        # Build variable lookup dict for efficient access
        var_map = {v.name: v for v in self.model_description.modelVariables}

        for name, value in self.start_values.items():
            variable = var_map.get(name)
            if variable is None:
                continue

            vr = variable.valueReference
            if self.fmi_version.startswith('2'):
                # FMI 2.0 API
                if variable.type in ['Real', 'Float64', 'Float32']:
                    self.fmu.setReal([vr], [value])
                elif variable.type in ['Integer', 'Int64', 'Int32', 'Int16', 'Int8']:
                    self.fmu.setInteger([vr], [int(value)])
                elif variable.type == 'Boolean':
                    self.fmu.setBoolean([vr], [bool(value)])
            else:
                # FMI 3.0 API (FMPy provides simplified wrappers)
                if variable.type in ['Real', 'Float64', 'Float32']:
                    self.fmu.setFloat64([vr], [value])
                elif variable.type in ['Integer', 'Int64', 'Int32', 'Int16', 'Int8']:
                    self.fmu.setInt64([vr], [int(value)])
                elif variable.type == 'Boolean':
                    self.fmu.setBoolean([vr], [bool(value)])


    def _get_continuous_states(self):
        """Get continuous states from FMU (handles FMI 2.0 and 3.0 API differences).

        Returns
        -------
        states : array
            continuous state vector
        """
        if self.n_states == 0:
            return np.array([])

        if self.fmi_version.startswith('2'):
            return self.fmu.getContinuousStates()
        else:  # FMI 3.0
            import ctypes
            states = (ctypes.c_double * self.n_states)()
            self.fmu.getContinuousStates(states, self.n_states)
            return np.array(states)


    def _set_fmu_state(self, x, u, t):
        """Set FMU state (time, continuous states, inputs).

        Parameters
        ----------
        x : array
            continuous state vector
        u : array
            input vector
        t : float
            current time
        """
        self.fmu.setTime(t)

        if self.n_states > 0:
            if self.fmi_version.startswith('2'):
                self.fmu.setContinuousStates(x)
            else:  # FMI 3.0
                import ctypes
                x_ctypes = (ctypes.c_double * self.n_states)(*x)
                self.fmu.setContinuousStates(x_ctypes, self.n_states)

        if self._input_refs:
            input_vrefs = list(self._input_refs.values())
            if self.fmi_version.startswith('2'):
                self.fmu.setReal(input_vrefs, u)
            else:  # FMI 3.0
                self.fmu.setFloat64(input_vrefs, u)


    def _get_derivatives(self, x, u, t):
        """Evaluate FMU derivatives (RHS of ODE).

        Parameters
        ----------
        x : array
            continuous state vector
        u : array
            input vector
        t : float
            current time

        Returns
        -------
        dx : array
            state derivatives
        """
        if self.n_states == 0:
            return np.array([])

        self._set_fmu_state(x, u, t)
        if self.fmi_version.startswith('2'):
            return self.fmu.getDerivatives()
        else:  # FMI 3.0
            import ctypes
            derivatives = (ctypes.c_double * self.n_states)()
            self.fmu.getContinuousStateDerivatives(derivatives, self.n_states)
            return np.array(derivatives)


    def _get_outputs(self, x, u, t):
        """Evaluate FMU outputs (algebraic part).

        Parameters
        ----------
        x : array
            continuous state vector
        u : array
            input vector
        t : float
            current time

        Returns
        -------
        y : array
            output vector
        """
        self._set_fmu_state(x, u, t)
        if not self._output_refs:
            return np.array([])

        output_vrefs = list(self._output_refs.values())
        if self.fmi_version.startswith('2'):
            return self.fmu.getReal(output_vrefs)
        else:  # FMI 3.0
            return np.array(self.fmu.getFloat64(output_vrefs))


    def sample(self, t, dt):
        """Sample block after successful timestep and handle FMU step completion events.

        This is called by the simulation after a complete timestep (all RK stages finished).
        It's the proper place to call completedIntegratorStep() per FMI standard.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        dt : float
            integration timestep
        """
        # Call parent's sample method (if any)
        super().sample(t, dt)

        # If FMU requires completedIntegratorStep, call it after successful step
        if not self.completed_integrator_step_not_needed:
            # Notify FMU that integration step completed
            enter_event_mode, terminate_simulation = self.fmu.completedIntegratorStep()

            if terminate_simulation:
                if self.verbose:
                    print("FMU requested termination in completedIntegratorStep")
                raise RuntimeError("FMU requested simulation termination")

            # If FMU signals a step event
            if enter_event_mode:
                if self.verbose:
                    print(f"Step completion event at t={t}")

                # Handle the step event
                self._handle_event(t)


    def _get_event_indicator(self, idx):
        """Get value of a specific event indicator.

        Parameters
        ----------
        idx : int
            index of the event indicator

        Returns
        -------
        float
            event indicator value
        """
        if self.fmi_version.startswith('2'):
            indicators = self.fmu.getEventIndicators()
            return indicators[idx]
        else:  # FMI 3.0
            import ctypes
            indicators = (ctypes.c_double * self.n_event_indicators)()
            self.fmu.getEventIndicators(indicators, self.n_event_indicators)
            return indicators[idx]


    def _handle_event(self, t):
        """Handle FMU event with fixed-point iteration for discrete states.

        Parameters
        ----------
        t : float
            event time
        """
        if self.verbose:
            print(f"FMU event detected at t={t}")

        # Enter event mode before event iteration
        self.fmu.enterEventMode()

        # Perform event update iteration until discrete states stabilize
        if self.fmi_version.startswith('2'):
            # FMI 2.0 API - returns EventInfo object
            while True:
                event_info = self.fmu.eventUpdate()

                # Check if simulation should terminate
                if event_info.terminateSimulation:
                    if self.verbose:
                        print("FMU requested simulation termination")
                    raise RuntimeError("FMU requested simulation termination")

                # Break if no more discrete state updates needed
                if not event_info.newDiscreteStatesNeeded:
                    break
        else:
            # FMI 3.0 API - returns tuple
            # (discreteStatesNeedUpdate, terminateSimulation, nominalsChanged,
            #  valuesChanged, nextEventTimeDefined, nextEventTime)
            while True:
                result = self.fmu.updateDiscreteStates()
                discreteStatesNeedUpdate = result[0]
                terminateSimulation = result[1]
                valuesOfContinuousStatesChanged = result[3]
                nextEventTimeDefined = result[4]
                nextEventTime = result[5]

                # Check if simulation should terminate
                if terminateSimulation:
                    if self.verbose:
                        print("FMU requested simulation termination")
                    raise RuntimeError("FMU requested simulation termination")

                # Break if no more discrete state updates needed
                if not discreteStatesNeedUpdate:
                    break

            # Create event_info-like object for unified handling below
            class EventInfo:
                pass
            event_info = EventInfo()
            event_info.valuesOfContinuousStatesChanged = valuesOfContinuousStatesChanged
            event_info.nextEventTimeDefined = nextEventTimeDefined
            event_info.nextEventTime = nextEventTime

        # Re-enter continuous time mode after event handling
        self.fmu.enterContinuousTimeMode()

        # Check if continuous states changed during event
        if event_info.valuesOfContinuousStatesChanged:
            x_new = self._get_continuous_states()
            self.engine.set(x_new)
            if self.verbose:
                print(f"Continuous states updated after event: {x_new}")

        # Update time events if FMU scheduled new ones
        if event_info.nextEventTimeDefined:
            self._update_time_events(event_info.nextEventTime)
            if self.verbose:
                print(f"Next time event scheduled at t={event_info.nextEventTime}")


    def _update_time_events(self, next_time):
        """Update or create time event schedule.

        Parameters
        ----------
        next_time : float
            next scheduled event time
        """
        if self.time_event is None:
            # Create new ScheduleList with the first time event
            self.time_event = ScheduleList(
                times_evt=[next_time],
                func_act=self._handle_event,
                tolerance=self.tolerance
            )
            self.events.append(self.time_event)
        elif next_time not in self.time_event.times_evt:
            # Insert new time in sorted order
            import bisect
            bisect.insort(self.time_event.times_evt, next_time)


    def reset(self):
        """Reset the FMU instance."""
        super().reset()
        self.fmu.reset()

        # Re-initialize FMU
        if self.fmi_version.startswith('3.'):
            self.fmu.enterInitializationMode(
                tolerance=self.tolerance,
                startTime=0.0,
                stopTime=None
            )
        else:
            self.fmu.setupExperiment(tolerance=self.tolerance, startTime=0.0)
            self.fmu.enterInitializationMode()

        self._set_start_values()
        event_info = self.fmu.exitInitializationMode()
        self.fmu.enterContinuousTimeMode()

        # Reset to initial states
        self.engine.set(self._get_continuous_states())

        # Reset time events and re-schedule initial event if present
        if self.time_event is not None:
            self.time_event.times_evt.clear()
            if self._initial_time_event is not None:
                import bisect
                bisect.insort(self.time_event.times_evt, self._initial_time_event)


    def __del__(self):
        """Cleanup FMU resources."""
        try:
            self.fmu.terminate()
            self.fmu.freeInstance()
        except Exception:
            pass