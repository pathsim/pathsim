#########################################################################################
##
##                                 SUBSYSTEM DEFINITION 
##                                    (subsystem.py)
##
##              This module contains the 'Subsystem' and 'Interface' classes 
##         that manage subsystems that can be embedded within a larger simulation
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from collections import defaultdict

from .connection import Connection

from .blocks._block import Block

from .optim.booster import ConnectionBooster

from .utils.graph import Graph
from .utils.register import Register
from .utils.portreference import PortReference

from ._constants import (
    SIM_ITERATIONS_MAX,
    SIM_TOLERANCE_FPI,
    LOG_ENABLE
    )


# IO CLASS ==============================================================================

class Interface(Block):
    """Bare-bone block that serves as a data interface for the 'Subsystem' class.

    It works like this:
    
    - Internal blocks of the subsystem are connected to the inputs and outputs 
      of this Interface block via the internal connections.
    - It behaves like a normal block (inherits the main 'Block' class methods).
    - It implements some special methods to get and set the inputs and outputs 
      of the blocks, that are used to translate between the internal blocks of the 
      subsystem and the inputs and outputs of the subsystem.
    - Handles data transfer to and from the internal subsystem blocks 
      to and from the inputs and outputs of the subsystem.
    """
    def __len__(self):
        return 0


    def register_port_map(self, port_map_in, port_map_out):
        """Update the input and output registers of the interface block with port mappings

        Parameters
        ----------
        port_map_in : dict[str: int]
            port alias mapping for block inputs
        port_map_out : dict[str: int]
            port alias mapping for block outputs
        """

        #build registers with mappings
        self.inputs = Register(mapping=port_map_in)
        self.outputs = Register(mapping=port_map_out)


# MAIN SUBSYSTEM CLASS ==================================================================

class Subsystem(Block):
    """Subsystem class that holds its own blocks and connecions and 
    can natively interface with the main simulation loop. 

    IO interface is realized by a special 'Interface' block, that has extra 
    methods for setting and getting inputs and outputs and serves 
    as the interface of the internal blocks to the outside. 

    The subsystem doesnt use its 'inputs' and 'outputs' dicts directly. 
    It exclusively handles data transfer via the 'Interface' block. 

    This class can be used just like any other block during the simulation, 
    since it implements the required methods 'update' for the fixed-point 
    iteration (resolving algebraic loops with instant time blocks), 
    the 'step' method that performs timestepping (especially for dynamic 
    blocks with internal states) and the 'solve' method for solving the 
    implicit update equation for implicit solvers. 


    Example
    -------
    This is how we can wrap up multiple blocks within a subsystem. 
    In this case vanderpol system built from discrete components 
    instead of using an ODE block (in practice you should use 
    a monolithic ODE whenever possible due to performance).

    .. code-block:: python
        
        from pathsim import Subsystem, Interface, Connection
        from pathsim.blocks import Integrator, Function

        #van der Pol parameter
        mu = 1000

        #blocks in the subsystem
        If = Interface() # this is the interface to the outside
        I1 = Integrator(2)
        I2 = Integrator(0)
        Fn = Function(lambda x1, x2: mu*(1 - x1**2)*x2 - x1)

        sub_blocks = [If, I1, I2, Fn]

        #connections in the subsystem
        sub_connections = [
            Connection(I2, I1, Fn[1], If[1]), 
            Connection(I1, Fn, If), 
            Connection(Fn, I2)
            ]

        #the subsystem acts just like a normal block
        vdp = Subsystem(sub_blocks, sub_connections)


    Parameters
    ----------
    blocks : list[Block] 
        internal blocks of the subsystem
    connections : list[Connection]
        internal connections of the subsystem
    tolerance_fpi : float
        absolute tolerance for convergence of algebraic loops
        default see ´SIM_TOLERANCE_FPI´ in ´_constants.py´
    iterations_max : int
        maximum allowed number of iterations for algebraic loop 
        solver, default see ´SIM_ITERATIONS_MAX´ in ´_constants.py´

    Attributes
    ----------
    interface : Interface
        internal interface block for data transfer to the outside
    graph : Graph
        internal graph representation for fast system funcion 
        evluations using DAG with algebraic depths
    boosters : None | list[ConnectionBooster]
        list of boosters (fixed point accelerators) that wrap 
        algebraic loop closing connections assembled from the 
        system graph
    """

    def __init__(self, 
        blocks=None, 
        connections=None,
        tolerance_fpi=SIM_TOLERANCE_FPI, 
        iterations_max=SIM_ITERATIONS_MAX
        ):

        #internal integration engine -> initialized later
        self.engine = None

        #flag to set block (subsystem) active
        self._active = True

        #error tolerance for alg. loop solver
        self.tolerance_fpi = tolerance_fpi

        #max iterations for internal alg. loop solver
        self.iterations_max = iterations_max

        #operators for algebraic and dynamic components (not here)
        self.op_alg = None
        self.op_dyn = None

        #internal graph representation -> initialized later
        self.graph = None

        #internal algebraic loop solvers -> initialized later
        self.boosters = None

        #internal connecions
        self.connections = set() 
        if connections:
            self.connections.update(connections)
        
        #collect and organize internal blocks
        self.blocks = set()
        self.interface = None

        if blocks:
            for block in blocks:
                if isinstance(block, Interface): 
                    
                    if self.interface is not None:
                        #interface block is already defined
                        raise ValueError("Subsystem can only have one 'Interface' block!")
                    
                    self.interface = block
                else: 
                    #regular blocks
                    self.blocks.add(block)

        #check if interface is defined
        if self.interface is None:
            raise ValueError("Subsystem 'blocks' list needs to contain 'Interface' block!")

        #assemble internal graph
        self._assemble_graph()


    def __len__(self):
        """Check if the Subsystem has algebraic passthrough by quering 
        the graph for an algebraic path from the interface to itself.
        """
        is_alg = self.graph.is_algebraic_path(self.interface, self.interface)
        return int(is_alg)


    def __call__(self):
        """Recursively get the subsystems internal states of engines 
        (if available) of all internal blocks and nested subsystems 
        and the subsystem inputs and outputs as arrays for use outside. 

        Either for monitoring, postprocessing or event detection. 
        In any case this enables easy access to the current block state.
        """
        _inputs  = self.interface.outputs.to_array()
        _outputs = self.interface.inputs.to_array()
        _states  = []
        for block in self.blocks:
            _i, _o, _s = block()
            _states.append(_s)
        return _inputs, _outputs, np.hstack(_states)


    def __contains__(self, other):
        """Check if blocks and connections are already part of the subsystem

        Paramters
        ---------
        other : obj
            object to check if its part of subsystem

        Returns
        -------
        bool
        """
        return other in self.blocks or other in self.connections


    # subsystem graph assembly --------------------------------------------------------------

    def _assemble_graph(self):
        """Assemble internal graph of subsystem for fast 
        algebraic evaluation during simulation.
        """
        self.graph = Graph({*self.blocks, self.interface}, self.connections)

        #create boosters for loop closing connections
        if self.graph.has_loops:
            self.boosters = [
                ConnectionBooster(conn) for conn in self.graph.loop_closing_connections()
            ]


    # methods for access to metadata --------------------------------------------------------

    @property
    def size(self):
        """Get size information from subsystem, recursively assembled 
        from internal blocks, including nested subsystems.

        Returns
        -------
        sizes : tuple[int]
            size of block (number of all internal blocks) 
            and number of internal states (from internal engines)
        """
        total_n, total_nx = 0, 0
        for block in self.blocks:
            n, nx = block.size
            total_n += n
            total_nx += nx
        return total_n, total_nx


    # visualization -------------------------------------------------------------------------

    def plot(self, *args, **kwargs):
        """Plot the simulation results by calling all the blocks 
        that have visualization capabilities such as the 'Scope' 
        and 'Spectrum'.

        Parameters
        ----------
        args : tuple
            args for the plot methods
        kwargs : dict
            kwargs for the plot method
        """
        for block in self.blocks:
            block.plot(*args, **kwargs)


    # system management ---------------------------------------------------------------------

    def reset(self):
        """Reset the subsystem interface and all internal blocks"""

        #reset interface
        self.interface.reset()

        #reset internal blocks
        for block in self.blocks:
            block.reset()


    def on(self):
        """Activate the subsystem and all internal blocks, sets the boolean
        evaluation flag to 'True'.
        """
        self._active = True
        for block in self.blocks: 
            block.on()
    

    def off(self):
        """Deactivate the subsystem and all internal blocks, sets the boolean
        evaluation flag to 'False'. Also resets the subsystem.
        """
        self._active = False
        for block in self.blocks: 
            block.off()
        self.reset()


    def linearize(self, t):
        """Linearize the algebraic and dynamic components of the internal blocks.

        This is done by linearizing the internal 'Operator' and 'DynamicOperator' 
        instances of all the internal blocks of the subsystem in the current system 
        operating point. The operators create 1st order tayler approximations 
        internally and use them on subsequent calls after linarization.
    
        Recursively traverses down the hierarchy for nested subsystems and linearizes 
        all of them.

        Parameters
        ----------
        t : float 
            evaluation time
        """
        for block in self.blocks: 
            block.linearize(t)


    def delinearize(self):
        """Revert the linearization of the internal blocks."""
        for block in self.blocks: 
            block.delinearize()


    # serialization / deserialization -------------------------------------------------------
    
    def to_dict(self):
        """Custom serialization for Subsystem"""
        data = super().to_dict()
        
        #serialization for internal blocks and interface
        data["params"]["blocks"] = [block.to_dict() for block in {*self.blocks, self.interface}]

        #serialize connections
        data["params"]["connections"] = [conn.to_dict() for conn in self.connections]
        
        return data

    
    @classmethod
    def from_dict(cls, data):
        """Custom deserialization for Subsystem"""
        from .connection import Connection
        
        #deserialize blocks and create block ID mapping
        blocks, id_to_block = [], {}
        for blk_data in data["params"].pop("blocks", []):
            block = Block.from_dict(blk_data)
            blocks.append(block)
            id_to_block[blk_data["id"]] = block

        #deserialize connections
        connections = []
        for conn_data in data["params"].pop("connections", []):

            #source data
            source_block = id_to_block[conn_data["source"]["block"]]
            source_ports = conn_data["source"]["ports"]
            source = PortReference(source_block, source_ports)
            
            #target data
            targets = []
            for trg in conn_data["targets"]:
                target_block = id_to_block[trg["block"]]
                target_ports = trg["ports"]
                targets.append(
                    PortReference(target_block, target_ports)
                    )
            
            #create the connection
            connections.append(
                Connection(source, *targets)
                )
        
        #finally construct the subsystem
        return cls(blocks, connections)
        

    # methods for discrete event management -------------------------------------------------

    @property
    def events(self):
        """Recursively collect and return events spawned by the 
        internal blocks of the subsystem, for discrete time 
        blocks such as triggers / comparators, clocks, etc.
        """
        _events = []
        for block in self.blocks:
            _events.extend(block.events)
        return _events


    # methods for inter-block data transfer -------------------------------------------------

    @property    
    def inputs(self):
        return self.interface.outputs

    @property
    def outputs(self):
        return self.interface.inputs


    # methods for data recording ------------------------------------------------------------

    def sample(self, t, dt):
        """Update the internal connections again and sample data from 
        the internal blocks that implement the 'sample' method.
    
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep 
        """

        #record data if required
        for block in self.blocks:
            block.sample(t, dt)


    # methods for block output and state updates --------------------------------------------

    def update(self, t):
        """Update the instant time components of the internal blocks 
        to evaluate the (distributed) system equation.

        Parameters
        ----------
        t : float
            evaluation time 
        """

        #evaluate DAG
        self._dag(t)

        #algebraic loops -> solve them
        if self.graph.has_loops:   
            self._loops(t)

        
    def _dag(self, t):
        """Update the directed acyclic graph components of the system.
        
        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #update interface outgoing connections
        for connection in self.graph.outgoing_connections(self.interface):
            if connection: connection.update()

        #perform gauss-seidel iterations without error checking
        for _, blocks_dag, connections_dag in self.graph.dag():

            #update blocks at algebraic depth
            for block in blocks_dag:
                if block: block.update(t)

            #update connenctions at algebraic depth (data transfer)
            for connection in connections_dag:
                if connection: connection.update()


    def _loops(self, t):
        """Perform the algebraic loop solve of the subsystem using accelerated 
        fixed-point iterations on the broken loop directed graph.
        
        Parameters
        ----------
        t : float
            evaluation time for system function
        """

        #reset accelerators of loop closing connections
        for con_booster in self.boosters:
            con_booster.reset()

        #perform solver iterations on algebraic loops
        for iteration in range(1, self.iterations_max):
            
            #iterate DAG depths of broken loops
            for depth, blocks_loop, connections_loop in self.graph.loop():

                #update blocks at algebraic depth
                for block in blocks_loop:
                    if block: block.update(t)

                #step accelerated connenctions at algebraic depth (data transfer)
                for connection in connections_loop:
                    if connection: connection.update()

            #step boosters of loop closing connections
            max_err = 0.0
            for con_booster in self.boosters:
                err = con_booster.update()
                if err > max_err:
                    max_err = err
                       
            #check convergence after first iteration
            if max_err <= self.tolerance_fpi:
                return

        #not converged -> error
        raise RuntimeError(
            "algebraic loop in 'Subsystem' not converged (iters: {}, err: {})".format(
                self.iterations_max, max_err)
            )
        

    # methods for blocks with integration engines -------------------------------------------

    def solve(self, t, dt):
        """Advance solution of implicit update equation 
        for internal blocks.

        Parameters
        ----------
        t : float
            evaluation time 
        dt : float
            timestep
    
        Returns
        -------
        max_error : float
            maximum error of implicit update equaiton
        """
        max_error = 0.0
        for block in self._blocks_dyn:
            if not block: continue
            err = block.solve(t, dt)
            if err > max_error:
                max_error = err
        return max_error


    def step(self, t, dt):
        """Explicit component of timestep for internal blocks 
        including error propagation.

        Notes
        ----- 
        This is pretty much an exact copy of the '_step' method 
        from the 'Simulation' class.

        Parameters
        ---------- 
        t : float
            evaluation time 
        dt : float
            timestep

        Returns
        -------
        success : bool
            indicator if the timestep was successful
        max_error : float
            maximum local truncation error from integration
        scale : float
            rescale factor for timestep
        """

        #initial timestep rescale and error estimate
        success, max_error_norm, relevant_scales = True, 0.0, []

        #step blocks and get error estimates if available
        for block in self._blocks_dyn:

            #skip inactive internal blocks
            if not block: continue

            suc, err_norm, scl = block.step(t, dt)
            
            #check solver stepping success
            if not suc: 
                success = False

            #update error tracking
            if err_norm > max_error_norm: 
                max_error_norm = err_norm
            
            #update timestep rescale if relevant
            if scl != 1.0 and scl > 0.0: 
                relevant_scales.append(scl)

        #no relevant timestep rescale -> quit early
        if not relevant_scales: 
            return success, max_error_norm, 1.0

        #compute real timestep rescale
        return success, max_error_norm, min(relevant_scales)


    def set_solver(self, Solver, parent, **solver_args):
        """Initialize all blocks with solver for numerical integration
        and additional args for the solver such as tolerances, etc.

        If blocks already have solvers, change the numerical integrator
        to the 'Solver' class.
        
        Parameters
        ----------
        Solver : Solver
            numerical solver definition
        parent : Solver
            numerical solver instance as parent
        solver_args : dict
            args to initialize solver with 
        """

        #set internal dummy engine -> marks block as dynamic
        self.engine = Solver(parent=parent, **solver_args)

        #set integration engines and assemble list of dynamic blocks
        self._blocks_dyn = []
        for block in self.blocks:
            block.set_solver(Solver, parent, **solver_args)
            if block.engine:
                self._blocks_dyn.append(block)


    def revert(self):
        """revert the internal blocks to the state 
        of the previous timestep 
        """
        for block in self._blocks_dyn:
            if block: block.revert()


    def buffer(self, dt):
        """buffer internal states of blocks with 
        internal integration engines 
        
        Parameters
        ----------
        dt : float
            evaluation time for buffering    
        """
        for block in self._blocks_dyn:
            if block: block.buffer(dt)