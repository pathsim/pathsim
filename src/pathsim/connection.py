#########################################################################################
##
##                                   CONNECTION CLASS 
##                                    (connection.py)
##
##              This module implements the 'Connection' class that transfers
##                data between the blocks and their input/output channels
##
#########################################################################################

# IMPORTS ===============================================================================

from .utils.portreference import PortReference

# CLASSES ===============================================================================

class Connection:
    """Class to handle input-output relations of blocks by connecting them (directed graph) 
    and transfering data from the output port of the source block to the input port of 
    the target block.

    The default ports for connection are (0) -> (0), since these are the default inputs 
    that are used in the SISO blocks.

    Examples
    --------
    Lets assume we have some generic blocks 

    .. code-block:: python
    
        from pathsim.blocks._block import Block

        B1 = Block()
        B2 = Block()
        B3 = Block()


    that we want to connect. We initialize a 'Connection' with the blocks directly 
    as the arguments if we want to connect the default ports (0) -> (0) 
    
    .. code-block:: python

        from pathsim import Connection

        C = Connection(B1, B2)


    which is a connection from block 'B1' to 'B2'. If we want to explicitly declare 
    the input and output ports we can do that by utilizing the '__getitem__' method
    of the blocks

    .. code-block:: python
 
        C = Connection(B1[0], B2[0])


    which is exactly the default port setup. Connecting output port (1) of 'B1' to 
    the default input port (0) of 'B2' do

    .. code-block:: python

        C = Connection(B1[1], B2[0])
        

    or just
    
    .. code-block:: python

        C = Connection(B1[1], B2).


    The 'Connection' class also supports multiple targets for a single source. 
    This is specified by just adding more blocks with their respective ports into 
    the constructor like this:
    
    .. code-block:: python

        C = Connection(B1, B2[0], B2[1], B3)


    The port definitions follow the same structure as for single target connections.

    'self'-connections also work without a problem. This is useful for modeling direct 
    feedback of a block to itself.
    
    Port definitions support slicing. This enables direct MIMO connections. For example 
    connecting ports 0, 1, 2 of 'B1' to ports 1, 2, 3 of 'B2' works like this:

    .. code-block:: python

        C = Connection(B1[0:2], B2[1:3])

    
    Port definitions also support lists and tuples of 'int'. For example the slice 
    above is identical to this:
    
    .. code-block:: python

        C = Connection(B1[0, 1], B2[1, 2])


    Or to be more programmatic about it, like this:

    .. code-block:: python
    
        prts_1 = [0, 1]
        prts_2 = [1, 2]

        C = Connection(B1[prts_1], B2[prts_2])


    Another way to define the ports is by using strings. Some blocks have internal 
    aliases for the ports that can be used instead of the integer port indices to 
    define the connections (or access the port data):
    
    .. code-block:: python

        C = Connection(B1["out"], B2["in"])


    Or mixed with integer port indices:

    .. code-block:: python

        C = Connection(B1["out"], B2["in"])


    Parameters
    ----------
    source : PortReference, Block
        source block and optional source output port
    targets : tuple[PortReference], tuple[Block]
        target blocks and optional target input ports


    Attributes
    ----------
    _active : bool
        flag to set 'Connection' as active or inactive
    """

    __slots__ = ["source", "targets", "_active"]


    def __init__(self, source, *targets):
        
        #assign source block and port
        self.source = source if isinstance(source, PortReference) else PortReference(source)

        #assign target blocks and ports
        self.targets = [trg if isinstance(trg, PortReference) else PortReference(trg) for trg in targets]

        #flag to set connection active
        self._active = True
        
        #validate port aliases
        self._validate_ports()
        
        #validate port dimensions at connection creation
        self._validate_dimensions()


    def __str__(self):
        """String representation of the connection"""
        src = f"{self.source.block}[{self.source.ports}]"
        trgs = ", ".join(f"{t.block}[{t.ports}]" for t in self.targets)
        return f"Connection({src} -> {trgs})"


    def __len__(self):
        """Returns the number of ports that are defined in the connection"""
        return len(self.source)


    def __bool__(self):
        return self._active


    def __contains__(self, other):
        """Check if block is part of connection

        Paramters
        ---------
        other : Block
            block to check if its part of the connection

        Returns
        -------
        bool
            is other part of connecion?
        """
        if isinstance(other, Block): 
            return other in self.get_blocks()
        return False


    def _validate_dimensions(self):
        """Check the dimensions of the source and target ports, 
        if they dont match, raises an exception.
        """
        n_src = len(self.source)
        for trg in self.targets:
            if len(trg) != n_src:
                raise ValueError(f"Source and target have different number of ports!")


    def _validate_ports(self):
        """Check the existance of the input and output ports of 
        the defined source and target blocks. 

        Utilizes the `PortReference._validate_output_ports` and 
        `PortReference._validate_input_ports` methods.
        """
        self.source._validate_output_ports()
        for trg in self.targets:
            trg._validate_input_ports()


    def get_blocks(self):
        """Returns all the unique internal source and target blocks 
        of the connection instance

        Returns
        -------
        list[Block]
            internal unique blocks of the connection
        """
        blocks = [self.source.block]
        for trg in self.targets:
            if trg.block not in blocks:
                blocks.append(trg.block)
        return blocks


    def on(self):
        self._active = True


    def off(self):
        self._active = False


    def update(self):
        """Transfers data from the source block output port 
        to the target block input port.
        """
        for trg in self.targets:
            self.source.to(trg)


class Duplex(Connection):
    """Extension of the 'Connection' class, that defines bidirectional 
    connections between two blocks by grouping together the inputs and 
    outputs of the blocks into an IO-pair.
    """

    __slots__ = ["source", "target", "targets", "_active"]


    def __init__(self, source, target):
        
        self.source = source if isinstance(source, PortReference) else PortReference(source)
        self.target = target if isinstance(target, PortReference) else PortReference(target)
        
        #this is required for path length estimation
        self.targets = [self.target, self.source]

        #flag to set connection active
        self._active = True

        import warnings
        warnings.warn("'Duplex' will be deprecated in next release!")
        

    def update(self):
        """Transfers data between the two target blocks 
        and ports bidirectionally.
        """

        #bidirectional data transfer
        self.target.to(self.source)
        self.source.to(self.target)
