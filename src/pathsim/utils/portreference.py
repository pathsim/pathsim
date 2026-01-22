#########################################################################################
##
##                                PORT REFERENCE CLASS 
##                              (utils/portreference.py)
##                              
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np


# CLASS =================================================================================

class PortReference:
    """Container class that holds a reference to a block and a list of ports.
    Optimized with cached integer indices for ultra-fast transfers.

    Note
    ----
    The default port, when no ports are defined in the arguments is `0`.

    Parameters
    ----------
    block : Block
        internal block reference
    ports : list[int, str]
        list of port indices or names
    """

    __slots__ = ["block", "ports", "_input_indices", "_output_indices"]

    def __init__(self, block=None, ports=None):

        # Default port is '0'
        _ports = [0] if ports is None else ports 

        # Type validation for ports
        if not isinstance(_ports, list):            
            raise ValueError(f"'ports' must be list[int, str] but is '{type(_ports)}'!")
        
        for p in _ports:
            # Type validation for individual ports
            if not isinstance(p, (int, str)):
                raise ValueError(f"Port '{p}' must be (int, str) but is '{type(p)}'!")

            # Validation for positive integer
            if isinstance(p, int) and p < 0:
                raise ValueError(f"Port '{p}' is int but must be positive!")
            
            # Key existence validation for string ports
            if not (p in block.inputs or p in block.outputs):        
                raise ValueError(f"Port alias '{p}' not defined for Block {block}!")

        # Port uniqueness validation
        if len(_ports) != len(set(_ports)):
            raise ValueError("'ports' must be unique!")

        self.block = block
        self.ports = _ports
        
        # Cache for resolved integer indices (lazily initialized)
        self._input_indices = None
        self._output_indices = None


    def __len__(self):
        """The number of ports managed by 'PortReference'"""
        return len(self.ports)


    def _get_input_indices(self):
        """Get cached input indices, resolving string aliases to integers.
        Also expands the input array if needed.
        """
        if self._input_indices is None:

            # Resolve indices/aliases through mapping   
            self._input_indices = np.array([
                self.block.inputs._map(p) for p in self.ports
                ], dtype=np.intp)

            # Resize register to accommodate indices
            max_idx = self._input_indices.max()
            self.block.inputs.resize(max_idx + 1)
                                    
        return self._input_indices


    def _get_output_indices(self):
        """Get cached output indices, resolving string aliases to integers.
        Also expands the output array if needed.
        """
        if self._output_indices is None:

            # Resolve indices/aliases through mapping            
            self._output_indices = np.array([
                self.block.outputs._map(p) for p in self.ports
                ], dtype=np.intp)

            # Resize register to accommodate indices
            max_idx = self._output_indices.max()
            self.block.outputs.resize(max_idx + 1)
        
        return self._output_indices


    def _validate_input_ports(self):
        """Check the existence of the input ports, specifically string port 
        aliases for the block inputs. Raises a ValueError if not existent.
        """
        for p in self.ports:
            if not p in self.block.inputs:
                raise ValueError(f"Input port '{p}' not defined for Block {self.block}!")


    def _validate_output_ports(self):
        """Check the existence of the output ports, specifically string port 
        aliases for the block outputs. Raises a ValueError if not existent.
        """
        for p in self.ports:
            if not p in self.block.outputs:
                raise ValueError(f"Output port '{p}' not defined for Block {self.block}!")


    def to(self, other):
        """Transfer the data between two `PortReference` instances, 
        in this direction `self` -> `other`. From outputs to inputs.
        
        Uses numpy fancy indexing with cached integer indices.

        Parameters
        ----------
        other : PortReference
            the `PortReference` instance to transfer data to from `self`
        """

        # Get cached integer indices (lazy, resolved once, reused forever)
        src_indices = self._get_output_indices()
        dst_indices = other._get_input_indices()
        
        # Single vectorized transfer 
        other.block.inputs._data[dst_indices] = self.block.outputs._data[src_indices]


    def get_inputs(self):
        """Return the input values of the block at specified ports

        Returns
        -------
        out : numpy.ndarray
            input values of block
        """
        indices = self._get_input_indices()
        return self.block.inputs._data[indices]


    def set_inputs(self, vals):
        """Set the block inputs with values at specified ports

        Parameters
        ----------
        vals : array-like
            values to set at block input ports
        """
        if not isinstance(vals, np.ndarray):
            vals = np.asarray(vals)
        indices = self._get_input_indices()
        self.block.inputs._data[indices] = vals


    def get_outputs(self):
        """Return the output values of the block at specified ports

        Returns
        -------
        out : numpy.ndarray
            output values of block
        """
        indices = self._get_output_indices()
        return self.block.outputs._data[indices]


    def set_outputs(self, vals):
        """Set the block outputs with values at specified ports

        Parameters
        ----------
        vals : array-like
            values to set at block output ports
        """
        if not isinstance(vals, np.ndarray):
            vals = np.asarray(vals)
        indices = self._get_output_indices()
        self.block.outputs._data[indices] = vals

        
    def to_dict(self):
        """Serialization into dict"""
        return {
            "block": id(self.block),
            "ports": self.ports
        }