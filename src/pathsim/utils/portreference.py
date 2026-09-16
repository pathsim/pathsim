#########################################################################################
##
##                                PORT REFERENCE CLASS
##                              (utils/portreference.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .carrier import Carrier


# CLASS =================================================================================

class PortReference:
    """Container class that holds a reference to a block and a list of ports.
    Optimized with cached integer indices for ultra-fast transfers.

    Ports that are a `Carrier` cover multiple channels of the block register.
    Single channels of a carrier are referenced as a tuple of the port name
    and the channel key, which is what component access creates

    .. code-block:: python

        B["stream"].T     # -> PortReference(B, [("stream", "T")])
        B["stream"]["T"]  # -> PortReference(B, [("stream", "T")])


    The tuple is only resolved to an integer index together with the register
    of the respective direction, so a port name can be a different carrier
    for the inputs and for the outputs of a block.

    Note
    ----
    The default port, when no ports are defined in the arguments is `0`.

    Parameters
    ----------
    block : Block
        internal block reference
    ports : list[int, str, tuple[str, int | str]]
        list of port indices, port names or carrier channels
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
            if not isinstance(p, (int, str, tuple)):
                raise ValueError(f"Port '{p}' must be (int, str) but is '{type(p)}'!")

            # Validation for positive integer
            if isinstance(p, int) and p < 0:
                raise ValueError(f"Port '{p}' is int but must be positive!")

            # Key existence validation for string ports and carrier channels
            if not (self._is_defined(p, block.inputs) or self._is_defined(p, block.outputs)):
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
        """The number of channels managed by 'PortReference'. A port that is
        a `Carrier` counts with all of its channels.

        Note
        ----
        The direction is unknown here, so a port name that is defined for the
        inputs and the outputs counts with the larger of both. Use the '_size'
        method with the register of the respective direction instead.
        """
        return max(
            self._size(self.block.outputs),
            self._size(self.block.inputs)
            )


    def __getitem__(self, key):
        """Reference to a single channel of a port that is a `Carrier`,
        either by channel name or by channel position.

        Parameters
        ----------
        key : int, str
            channel name or channel position within the carrier

        Returns
        -------
        PortReference
            container object that holds the block reference and the channel
        """

        #component access only for a single port by name
        if len(self.ports) != 1 or not isinstance(self.ports[0], str):
            raise ValueError(f"Ports '{self.ports}' of Block {self.block} are no single carrier!")

        port = (self.ports[0], key)

        #channel has to exist in the carrier of at least one direction
        if not (self._is_defined(port, self.block.inputs) or self._is_defined(port, self.block.outputs)):
            raise ValueError(f"Channel '{key}' of port '{self.ports[0]}' not defined for Block {self.block}!")

        return PortReference(self.block, [port])


    def __getattr__(self, key):
        """Channel of a port that is a `Carrier` by name, this is an
        alias for the '__getitem__' method.

        Parameters
        ----------
        key : str
            channel name within the carrier

        Returns
        -------
        PortReference
            container object that holds the block reference and the channel
        """
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self[key]
        except ValueError as e:
            raise AttributeError(str(e)) from None


    @staticmethod
    def _get_carrier(port, register):
        """Get the `Carrier` a port refers to in a register, or 'None' if the
        port is a plain channel there.

        Parameters
        ----------
        port : int, str, tuple[str, int | str]
            port index, port name or carrier channel
        register : Register
            register of the block for the respective direction

        Returns
        -------
        carrier : Carrier | None
            carrier of the port, 'None' for plain channels
        """
        name = port[0] if isinstance(port, tuple) else port
        if not isinstance(name, str):
            return None
        _port = register._mapping.get(name)
        return _port if isinstance(_port, Carrier) else None


    @staticmethod
    def _is_defined(port, register):
        """Check if a port exists in a register. Carrier channels exist,
        if the port is a carrier there that has the channel.

        Parameters
        ----------
        port : int, str, tuple[str, int | str]
            port index, port name or carrier channel
        register : Register
            register of the block for the respective direction

        Returns
        -------
        defined : bool
            port exists in the register
        """
        if not isinstance(port, tuple):
            return port in register

        carrier = PortReference._get_carrier(port, register)
        if carrier is None:
            return False
        try:
            carrier[port[1]]
        except ValueError:
            return False
        return True


    def _size(self, register):
        """Number of channels of the ports in a register, ports that are
        carriers count with all of their channels.

        Parameters
        ----------
        register : Register
            register of the block for the respective direction

        Returns
        -------
        size : int
            number of channels
        """
        size = 0
        for p in self.ports:
            carrier = None if isinstance(p, tuple) else self._get_carrier(p, register)
            size += 1 if carrier is None else len(carrier)
        return size


    def _resolve(self, register):
        """Resolve the ports to integer indices in a register, expanding
        the ports that are carriers to all of their channels.

        Parameters
        ----------
        register : Register
            register of the block for the respective direction

        Returns
        -------
        indices : np.ndarray[int]
            channel indices of the ports in the register
        """
        indices = []
        for p in self.ports:

            #single channel of a carrier
            if isinstance(p, tuple):
                indices.append(self._get_carrier(p, register)[p[1]])
                continue

            _port = register._map(p)
            if isinstance(_port, Carrier):
                indices.extend(_port)
            else:
                indices.append(_port)

        return np.array(indices, dtype=np.intp)


    def _get_input_indices(self):
        """Get cached input indices, resolving string aliases to integers.
        Also expands the input array if needed.
        """
        if self._input_indices is None:

            # Resolve indices/aliases through mapping
            self._input_indices = self._resolve(self.block.inputs)

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
            self._output_indices = self._resolve(self.block.outputs)

            # Resize register to accommodate indices
            max_idx = self._output_indices.max()
            self.block.outputs.resize(max_idx + 1)

        return self._output_indices


    def _validate_input_ports(self):
        """Check the existence of the input ports, specifically string port
        aliases for the block inputs. Raises a ValueError if not existent.
        """
        for p in self.ports:
            if not self._is_defined(p, self.block.inputs):
                raise ValueError(f"Input port '{p}' not defined for Block {self.block}!")


    def _validate_output_ports(self):
        """Check the existence of the output ports, specifically string port
        aliases for the block outputs. Raises a ValueError if not existent.
        """
        for p in self.ports:
            if not self._is_defined(p, self.block.outputs):
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
