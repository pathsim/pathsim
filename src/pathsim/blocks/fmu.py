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
from ..utils.fmuwrapper import FMUWrapper


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
    fmu_wrapper : FMUWrapper
        version-agnostic FMU wrapper instance
    dt : float
        communication step size
    """

    #max number of ports (will be configured based on FMU)
    _n_in_max = None
    _n_out_max = None

    #maps for input and output port labels
    _port_map_in = {}
    _port_map_out = {}

    def __init__(self, fmu_path, instance_name="fmu_instance", start_values=None,
                 dt=None, verbose=False):

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.verbose = verbose
        self.start_values = start_values if start_values is not None else {}

        # Create FMU wrapper
        self.fmu_wrapper = FMUWrapper(fmu_path, instance_name, mode='cosimulation')

        # Expose commonly used attributes for backward compatibility
        self.model_description = self.fmu_wrapper.model_description
        self.fmi_version = self.fmu_wrapper.fmi_version
        self.unzipdir = self.fmu_wrapper.unzipdir
        self.fmu = self.fmu_wrapper.fmu
        self._input_refs = self.fmu_wrapper.input_refs
        self._output_refs = self.fmu_wrapper.output_refs

        # Extract metadata
        self._extract_fmu_metadata()

        # Determine step size
        if dt is None:
            if self.default_step_size is not None:
                self.dt = self.default_step_size
            else:
                raise ValueError("No step size provided and FMU has no default experiment step size")
        else:
            self.dt = dt

        # Build port maps from FMU variables
        self._port_map_in = {name: idx for idx, name in enumerate(self.fmu_wrapper.input_refs.keys())}
        self._port_map_out = {name: idx for idx, name in enumerate(self.fmu_wrapper.output_refs.keys())}

        # Initialize base class with proper port configuration
        super().__init__()

        # Initialize FMU
        self.fmu_wrapper.instantiate()
        self.fmu_wrapper.setup_experiment(start_time=0.0)
        self.fmu_wrapper.enter_initialization_mode()

        # Set start values
        for name, value in self.start_values.items():
            self.fmu_wrapper.set_variable(name, value)

        # Exit initialization mode
        self.fmu_wrapper.exit_initialization_mode()

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

        md = self.fmu_wrapper.model_description
        cs = md.coSimulation

        if cs is None:
            raise ValueError("FMU does not support Co-Simulation")

        # Extract capabilities
        self.can_interpolate_inputs = getattr(cs, 'canInterpolateInputs', False)
        self.can_handle_variable_step = getattr(cs, 'canHandleVariableCommunicationStepSize', False)
        self.max_output_derivative_order = getattr(cs, 'maxOutputDerivativeOrder', 0)

        # Extract default experiment settings
        default_experiment = md.defaultExperiment

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
        self.model_name = md.modelName
        self.generation_tool = getattr(md, 'generationTool', 'Unknown')
        self.generation_date = getattr(md, 'generationDateAndTime', 'Unknown')
        self.description = getattr(md, 'description', '')
        self.author = getattr(md, 'author', 'Unknown')
        self.version = getattr(md, 'version', 'Unknown')


    def _step_fmu(self, t):
        """Perform one FMU co-simulation step"""
        self._update_fmu_from_inputs()

        # Perform co-simulation step
        result = self.fmu_wrapper.do_step(
            current_time=t,
            step_size=self.dt
        )

        # Handle FMI 3.0 specific results
        if result.terminate_simulation:
            raise RuntimeError("FMU requested simulation termination")

        self._update_outputs_from_fmu()


    def _update_fmu_from_inputs(self):
        """Read block inputs and update FMU inputs."""
        if len(self.fmu_wrapper.input_refs) > 0:
            input_vrefs = list(self.fmu_wrapper.input_refs.values())
            self.fmu_wrapper.set_real(input_vrefs, self.inputs.to_array())


    def _update_outputs_from_fmu(self):
        """Read outputs from FMU and update block outputs."""
        if len(self.fmu_wrapper.output_refs) > 0:
            output_vrefs = list(self.fmu_wrapper.output_refs.values())
            self.outputs.update_from_array(self.fmu_wrapper.get_real(output_vrefs))


    def update(self, t):
        """Update FMU inputs/outputs between scheduled steps if interpolation supported."""
        if self.can_interpolate_inputs:
            self._update_fmu_from_inputs()
            self._update_outputs_from_fmu()


    def reset(self):
        """Reset the FMU instance."""
        super().reset()
        self.fmu_wrapper.reset()
        self.fmu_wrapper.enter_initialization_mode()
        self.fmu_wrapper.exit_initialization_mode()
        self._update_outputs_from_fmu()


    def __len__(self):
        """FMU is a discrete time source-like block without direct passthrough"""
        return 0


    def __del__(self):
        """Cleanup FMU resources."""
        try:
            self.fmu_wrapper.terminate()
            self.fmu_wrapper.free_instance()
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
    fmu_wrapper : FMUWrapper
        version-agnostic FMU wrapper instance
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

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.verbose = verbose
        self.tolerance = tolerance
        self.start_values = start_values if start_values is not None else {}

        # Create FMU wrapper
        self.fmu_wrapper = FMUWrapper(fmu_path, instance_name, mode='model_exchange')

        # Expose commonly used attributes for backward compatibility
        self.model_description = self.fmu_wrapper.model_description
        self.fmi_version = self.fmu_wrapper.fmi_version
        self.unzipdir = self.fmu_wrapper.unzipdir
        self.fmu = self.fmu_wrapper.fmu
        self.n_states = self.fmu_wrapper.n_states
        self.n_event_indicators = self.fmu_wrapper.n_event_indicators
        self._input_refs = self.fmu_wrapper.input_refs
        self._output_refs = self.fmu_wrapper.output_refs

        # Extract metadata
        self._extract_fmu_metadata()

        # Build port maps from FMU variables
        self._port_map_in = {name: idx for idx, name in enumerate(self.fmu_wrapper.input_refs.keys())}
        self._port_map_out = {name: idx for idx, name in enumerate(self.fmu_wrapper.output_refs.keys())}

        # Setup FMU
        self.fmu_wrapper.instantiate()
        self.fmu_wrapper.setup_experiment(tolerance=self.tolerance, start_time=0.0)
        self.fmu_wrapper.enter_initialization_mode()

        # Set start values
        for name, value in self.start_values.items():
            self.fmu_wrapper.set_variable(name, value)

        # Exit initialization mode and check for initial events
        event_info = self.fmu_wrapper.exit_initialization_mode()

        # Store initial time event if defined
        self._initial_time_event = (
            event_info.next_event_time
            if event_info and event_info.next_event_time_defined
            else None
        )

        # Enter continuous time mode after initialization
        self.fmu_wrapper.enter_continuous_time_mode()

        # Get initial continuous states
        initial_states = self.fmu_wrapper.get_continuous_states()

        # Initialize parent DynamicalSystem with FMU dynamics
        super().__init__(
            func_dyn=self._get_derivatives,
            func_alg=self._get_outputs,
            initial_value=initial_states,
            jac_dyn=None
        )

        # Initialize time event manager
        self.time_event = None

        # Create state event (zero-crossing) events for each event indicator
        for i in range(self.fmu_wrapper.n_event_indicators):
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

        md = self.fmu_wrapper.model_description
        me = md.modelExchange

        if me is None:
            raise ValueError("FMU does not support Model Exchange")

        # Extract capabilities
        self.provides_directional_derivative = getattr(me, 'providesDirectionalDerivative', False)
        self.completed_integrator_step_not_needed = getattr(me, 'completedIntegratorStepNotNeeded', False)

        # Model metadata
        self.model_name = md.modelName
        self.generation_tool = getattr(md, 'generationTool', 'Unknown')
        self.generation_date = getattr(md, 'generationDateAndTime', 'Unknown')
        self.description = getattr(md, 'description', '')
        self.author = getattr(md, 'author', 'Unknown')
        self.version = getattr(md, 'version', 'Unknown')


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
        if self.fmu_wrapper.n_states == 0:
            return np.array([])

        # Set FMU state
        self.fmu_wrapper.set_time(t)
        self.fmu_wrapper.set_continuous_states(x)

        if len(self.fmu_wrapper.input_refs) > 0:
            input_vrefs = list(self.fmu_wrapper.input_refs.values())
            self.fmu_wrapper.set_real(input_vrefs, u)

        return self.fmu_wrapper.get_derivatives()


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
        # Set FMU state
        self.fmu_wrapper.set_time(t)
        self.fmu_wrapper.set_continuous_states(x)

        if len(self.fmu_wrapper.input_refs) > 0:
            input_vrefs = list(self.fmu_wrapper.input_refs.values())
            self.fmu_wrapper.set_real(input_vrefs, u)

        if len(self.fmu_wrapper.output_refs) == 0:
            return np.array([])

        output_vrefs = list(self.fmu_wrapper.output_refs.values())
        return self.fmu_wrapper.get_real(output_vrefs)


    def sample(self, t, dt):
        """Sample block after successful timestep and handle FMU step completion events.

        Parameters
        ----------
        t : float
            evaluation time for sampling
        dt : float
            integration timestep
        """
        super().sample(t, dt)

        # If FMU requires completedIntegratorStep, call it after successful step
        if not self.completed_integrator_step_not_needed:
            enter_event_mode, terminate_simulation = self.fmu_wrapper.completed_integrator_step()

            if terminate_simulation:
                if self.verbose:
                    print("FMU requested termination in completedIntegratorStep")
                raise RuntimeError("FMU requested simulation termination")

            if enter_event_mode:
                if self.verbose:
                    print(f"Step completion event at t={t}")
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
        indicators = self.fmu_wrapper.get_event_indicators()
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
        self.fmu_wrapper.enter_event_mode()

        # Perform event update iteration until discrete states stabilize
        while True:
            event_info = self.fmu_wrapper.update_discrete_states()

            # Check if simulation should terminate
            if event_info.terminate_simulation:
                if self.verbose:
                    print("FMU requested simulation termination")
                raise RuntimeError("FMU requested simulation termination")

            # Break if no more discrete state updates needed
            if not event_info.discrete_states_need_update:
                break

        # Re-enter continuous time mode after event handling
        self.fmu_wrapper.enter_continuous_time_mode()

        # Check if continuous states changed during event
        if event_info.values_changed:
            x_new = self.fmu_wrapper.get_continuous_states()
            self.engine.set(x_new)
            if self.verbose:
                print(f"Continuous states updated after event: {x_new}")

        # Update time events if FMU scheduled new ones
        if event_info.next_event_time_defined:
            self._update_time_events(event_info.next_event_time)
            if self.verbose:
                print(f"Next time event scheduled at t={event_info.next_event_time}")


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
        self.fmu_wrapper.reset()

        # Re-initialize FMU
        self.fmu_wrapper.setup_experiment(tolerance=self.tolerance, start_time=0.0)
        self.fmu_wrapper.enter_initialization_mode()

        for name, value in self.start_values.items():
            self.fmu_wrapper.set_variable(name, value)

        event_info = self.fmu_wrapper.exit_initialization_mode()
        self.fmu_wrapper.enter_continuous_time_mode()

        # Reset to initial states
        self.engine.set(self.fmu_wrapper.get_continuous_states())

        # Reset time events and re-schedule initial event if present
        if self.time_event is not None:
            self.time_event.times_evt.clear()
            if self._initial_time_event is not None:
                import bisect
                bisect.insort(self.time_event.times_evt, self._initial_time_event)


    def __del__(self):
        """Cleanup FMU resources."""
        try:
            self.fmu_wrapper.terminate()
            self.fmu_wrapper.free_instance()
        except Exception:
            pass
