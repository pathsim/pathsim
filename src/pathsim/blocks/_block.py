#########################################################################################
##
##                                     BASE BLOCK 
##                                 (blocks/_block.py)
##
##            This module defines the base 'Block' class that is the parent 
##         to all other blocks and can serve as a base for new or custom blocks
##
#########################################################################################

# IMPORTS ===============================================================================

from ..utils.register import Register
from ..utils.serialization import Serializable
from ..utils.portreference import PortReference


# BASE BLOCK CLASS ======================================================================

class Block(Serializable):
    """Base 'Block' object that defines the inputs, outputs and the connect method.

    Block interconnections are handeled via the io interface of the blocks. 
    It is realized by dicts for the 'inputs' and for the 'outputs', where the 
    key of the dict is the input/output channel and the corresponding value is 
    the input/output value. 

    The block can spawn discrete events that are handled by the main simulation 
    for triggers, discrete time blocks, etc.

    Mathematically the block behavior is defined by two operators in most cases

    .. math::
    
        \\begin{eqnarray}
        \\dot{x} &= f_\\mathrm{dyn}(x, u, t)\\\\
               y &= f_\\mathrm{alg}(x, u, t)
        \\end{eqnarray}


    they are algebraic operators for the algebraic path of the block and for the 
    dynamic path that feeds into the internal numerical integration engine.

    There are special cases where one or both of them are not defined, also for 
    purely algebraic blocks such as multipliers and functions, there exists a 
    simplified operator definition:
    
    .. math::

        y = f_\\mathrm{alg}(u)
    

    Note
    ----
    This block is not intended to be used directly and serves as a base 
    class definition for other blocks to be inherited.
    

    Attributes
    ----------
    inputs : Register
        input value register of block
    outputs : Register
        output value register of block
    engine : None | Solver
        numerical integrator instance
    events : list[Event]
        list of internal events, for mixed signal blocks
    _active : bool
        flag that sets the block active or inactive
    op_alg : Operator | DynamicOperator | None
        internal callable operator for algebraic components of block
    op_dyn : DynamicOperator | None
        internal callable operator for dynamic (ODE) components of block
    """

    def __init__(self):

        #registers to hold input and output values
        self.inputs = Register()
        self.outputs = Register()

        #initialize integration engine as 'None' by default
        self.engine = None

        #flag to set block active
        self._active = True

        #internal discrete events (for mixed signal blocks)
        self.events = []

        #operators for algebraic and dynamic components
        self.op_alg = None
        self.op_dyn = None


    def __len__(self):
        """The '__len__' method of the block is used to compute the length of the 
        algebraic path of the block. 

        For instant time blocks or blocks with purely algenbraic components 
        (adders, amplifiers, etc.) it returns 1, otherwise (integrator, delay, etc.) 
        it returns 0.

        If the block is disabled '_active == False', it returns 0 as well, since
        this breaks the signal path.

        Returns
        -------
        len : int
            length of the algebraic path of the block
        """
        return 1 if self._active else 0


    def __getitem__(self, key):
        """The '__getitem__' method is intended to make connection creation more 
        convenient and therefore just returns the block itself and the key directly 
        after doing some basic checks.

        Parameters
        ----------
        key : int, str, slice, tuple[int, str], list[int, str]
            port indices or port names, or list / tuple of them

        Returns
        -------
        PortReference
            container object that hold block reference and list of ports
        """

        if isinstance(key, slice):

            #slice validation
            if key.stop is None: raise ValueError("Port slice cannot be open ended!")
            if key.stop == 0: raise ValueError("Port slice cannot end with 0!")

            #start, step handling
            start = 0 if key.start is None else key.start
            step  = 1 if key.step  is None else key.step

            #build port list
            ports = list(range(start, key.stop, step))
            return PortReference(self, ports)

        elif isinstance(key, (tuple, list)):
            
            for k in key:

                #port type validation
                if not isinstance(k, (int, str)):
                    raise ValueError(f"Port '{k}' must be (int, str) but is '{type(k)}'!")
            
            #duplicate validation
            if len(set(key)) < len(key):
                raise ValueError("Ports cannot be duplicates!")

            return PortReference(self, list(key))

        elif isinstance(key, (int, str)):

            #standard key
            return PortReference(self, [key])

        else:
            raise ValueError(f"Port must be type (int, str, slice, tuple[int, str], list[int, str]) but is '{type(key)}'!")


    def __call__(self):
        """The '__call__' is an alias for the 'get_all' method."""
        return self.get_all()


    def __bool__(self):
        return self._active


    # methods for access to metadata ----------------------------------------------------

    @property
    def size(self):
        """Get size information from block, such as 
        number of internal states, etc.

        Returns
        -------
        sizes : tuple[int]
            size of block (default 1) and number 
            of internal states (from internal engine)
        """
        nx = len(self.engine) if self.engine else 0
        return 1, nx

    @property
    def shape(self):
        """Get the number of input and output ports of the block

        Returns
        -------
        shape : tuple[int]
            number of input and output ports
        """ 
        return len(self.inputs), len(self.outputs)


    # methods for visualization ---------------------------------------------------------

    def plot(self, *args, **kwargs):
        """Block specific visualization, enables plotting 
        access from the simulation level.

        This gets primarily used by the visualization blocks 
        such as the 'Scope' and 'Spectrum'.

        Parameters
        ----------
        args : tuple
            args for the plot methods
        kwargs : dict
            kwargs for the plot method
        """
        pass


    # methods for simulation management -------------------------------------------------

    def on(self):
        """Activate the block and all internal events, sets the boolean
        evaluation flag to 'True'.
        """
        self._active = True
        for event in self.events: 
            event.on()


    def off(self):
        """Deactivate the block and all internal events, sets the boolean 
        evaluation flag to 'False'. Also resets the block.
        """
        self._active = False
        for event in self.events: 
            event.off()
        self.reset()  


    def reset(self):
        """Reset the blocks inputs and outputs and also its internal solver, 
        if the block has a solver instance.
        """
        #reset inputs and outputs
        self.inputs.reset()
        self.outputs.reset()

        #reset engine if block has solver
        if self.engine: self.engine.reset()

        #reset operators if defined
        if self.op_alg: self.op_alg.reset()
        if self.op_dyn: self.op_dyn.reset()


    def linearize(self, t):
        """Linearize the algebraic and dynamic components of the block.

        This is done by linearizing the internal 'Operator' and 'DynamicOperator' 
        instances in the current system operating point. The operators create 
        1st order taylor approximations internally and use them on subsequent 
        calls after linarization.

        Parameters
        ----------
        t : float 
            evaluation time
        """

        #get current state
        u, _, x = self.get_all()

        #no engine -> stateless
        if not self.engine:
            #linearize only algebraic operator 
            if self.op_alg: self.op_alg.linearize(u)
        else:
            #linearize algebraic and dynamic operators
            if self.op_alg: self.op_alg.linearize(x, u, t)
            if self.op_dyn: self.op_dyn.linearize(x, u, t)


    def delinearize(self):
        """Revert the linearization of the blocks algebraic and dynamic components.
        
        This is resets the internal 'Operator' and 'DynamicOperator' instances, 
        deleting the linear surrogate model and using the original function for 
        subsequent calls.
        """
        #reset algebraic and dynamic operators
        if self.op_alg: self.op_alg.reset()
        if self.op_dyn: self.op_dyn.reset()


    # methods for blocks with integration engines ---------------------------------------

    def set_solver(self, Solver, parent, **solver_args):
        """Initialize the numerical integration engine with local truncation error 
        tolerance if required.

        If the block already has an integration engine, it is changed, 
        if it does not require an integration engine, this method just passes.

        Parameters
        ----------
        Solver : Solver
            numerical integrator
        parent : None | Solver
            numerical integrator instance
        solver_args : dict
            additional args for the solver
        """
        pass


    def revert(self):
        """Revert the block to the state of the previous timestep, if the 
        block has a solver instance indicated by the 'has_engine' flag.

        This is required for adaptive solvers to revert the state to the 
        previous timestep.
        """
        if self.engine: self.engine.revert()


    def buffer(self, dt):
        """
        Buffer current internal state of the block and the current timestep
        if the block has a solver instance (is stateful).

        This is required for multistage, multistep and adaptive integrators.

        Parameters
        ----------
        dt : float
            integration timestep
        """
        if self.engine: self.engine.buffer(dt)


    # methods for sampling data ---------------------------------------------------------
    
    def sample(self, t, dt):
        """Samples the data of the blocks inputs or internal state when called. 

        This can record block parameters after a succesful timestep such as 
        for the 'Scope' and 'Delay' blocks but also for sampling from a random 
        distribution in the 'RNG' and the noise blocks.
        
        Parameters
        ----------
        t : float
            evaluation time for sampling
        dt : float
            integration timestep
        """
        pass


    # methods for inter-block data transfer ---------------------------------------------

    def get_all(self):
        """Retrieves and returns internal states of engine (if available) 
        and the block inputs and outputs as arrays for use outside. 

        Either for monitoring, postprocessing or event detection. 
        In any case this enables easy access to the current block state.

        Returns
        -------
        inputs : array
            block input register
        outputs : array
            block output register
        states : array
            internal states of the block
        """
        _inputs  = self.inputs.to_array()
        _outputs = self.outputs.to_array()
        _states  = self.engine.get() if self.engine else []
        return _inputs, _outputs, _states


    # methods for block output and state updates ----------------------------------------

    def update(self, t):
        """The 'update' method is called iteratively for all blocks to evaluate the 
        algebraic components of the global system ode from the DAG.

        It is meant for instant time blocks (blocks that dont have a delay due to the 
        timestep, such as Amplifier, etc.) and updates the 'outputs' of the block 
        directly based on the 'inputs' and possibly internal states. 

        Note
        ----
        The implementation of the 'update' method in the base 'Block' class is intended 
        as a fallback and is not performance optimized. Special blocks might reimplement 
        this method differently for higher performance, for example SISO or MISO blocks.

        Parameters
        ----------
        t : float
            evaluation time
        """

        #no internal algebraic operator -> early exit
        if self.op_alg is None:
            return 0.0

        #block inputs 
        u = self.inputs.to_array()

        #no internal state -> standard 'Operator'
        if self.engine: 
            x = self.engine.get()
            y = self.op_alg(x, u, t)
        else: 
            y = self.op_alg(u)           

        #update register
        self.outputs.update_from_array(y)


    def solve(self, t, dt):
        """The 'solve' method performs one iterative solution step that is required
        to solve the implicit update equation of the solver if an implicit solver 
        (numerical integrator) is used.

        It returns the relative difference between the new updated solution 
        and the previous iteration of the solution to track convergence within 
        an outer loop.

        This only has to be implemented by blocks that have an internal 
        integration engine with an implicit solver.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        ------- 
        error : float
            solver residual norm
        """
        return 0.0 


    def step(self, t, dt):
        """The 'step' method is used in transient simulations and performs an action 
        (numeric integration timestep, recording data, etc.) based on the current 
        inputs and the current internal state. 

        It performs one timestep for the internal states. For instant time blocks,
        the 'step' method does not has to be implemented specifically. 
        
        The method handles timestepping for dynamic blocks with internal states
        such as 'Integrator', 'StateSpace', etc.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """

        #by default no error estimate (error norm -> 0.0)
        return True, 0.0, 1.0