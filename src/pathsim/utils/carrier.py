#########################################################################################
##
##                                   CARRIER CLASS
##                               (utils/carrier.py)
##
##            This module defines the 'Carrier' class that groups multiple
##          scalar channels of a block register into a single named entity
##
#########################################################################################

# IMPORTS ===============================================================================

import copy

# CLASSES ===============================================================================

class Carrier:
    """Named range of scalar channels of a block register.

    A carrier describes a contiguous block of channels and gives each of them
    a name. It is placed at an explicit position ('start') in the register of
    a block and used as the value of a port label

    .. code-block:: python

        _in_1 = Carrier(keys=("F", "T"))
        _in_2 = Carrier(_in_1.stop, keys=("F", "T"))

        class Mixer(Block):
            input_port_labels = {"in_1": _in_1, "in_2": _in_2}


    The carrier only exists while the system graph is assembled. Component
    access resolves to plain integer indices, by key or by position, like a
    dict of channel indices

    .. code-block:: python

        _in_1["F"]  # -> 0
        _in_1["T"]  # -> 1
        _in_1[1]    # -> 1


    so the values themselves stay in the flat float64 register of the block
    and nothing carrier specific is left at runtime

    .. code-block:: python

        def func(u):
            return u[_in_1["F"]] + u[_in_2["F"]]


    Note
    ----
    Carriers are addressed as a whole when blocks are connected, a connection
    to a carrier port transfers all its channels at once.


    Parameters
    ----------
    start : int
        index of the first channel in the register of the block
    keys : tuple[str] | list[str] | None
        names of the channels, overrides the class level 'keys'


    Attributes
    ----------
    keys : tuple[str]
        names of the channels
    start : int
        index of the first channel
    stop : int
        index after the last channel
    """

    keys = ()

    def __init__(self, start=0, keys=None):

        #channel names, from the arguments or from the class
        if keys is not None:
            self.keys = tuple(keys)

        #key validation
        for key in self.keys:
            if not isinstance(key, str):
                raise ValueError(f"Key '{key}' must be str but is '{type(key)}'!")
        if len(set(self.keys)) < len(self.keys):
            raise ValueError("Keys must be unique!")

        #position validation
        if not isinstance(start, int) or start < 0:
            raise ValueError(f"Start '{start}' must be a positive int!")

        #size validation
        if self.size < 1:
            raise ValueError("Carrier must have at least one channel!")

        self.start = start
        self.stop = start + self.size


    def __repr__(self):
        return f"{self.__class__.__name__}({self.start}, {self.keys})"


    @property
    def size(self):
        """Number of channels covered by the carrier

        Returns
        -------
        size : int
            number of channels
        """
        return len(self.keys)


    def __len__(self):
        return self.size


    def __getitem__(self, key):
        """Index of a single channel of the carrier, by key or by position.

        Parameters
        ----------
        key : int, str
            channel name or channel position within the carrier

        Returns
        -------
        index : int
            index of the channel in the register of the block
        """

        if isinstance(key, str):
            if key not in self.keys:
                raise ValueError(f"Key '{key}' not defined for {self}!")
            return self.start + self.keys.index(key)

        if isinstance(key, int):
            if key < 0 or key >= self.size:
                raise ValueError(f"Channel '{key}' outside of {self}!")
            return self.start + key

        raise ValueError(f"Key must be (int, str) but is '{type(key)}'!")


    @property
    def slice(self):
        """All channels of the carrier as a slice, for direct access to the
        flat register data, for example 'u[_in_1.slice]'

        Returns
        -------
        slice : slice
            slice covering all channels of the carrier
        """
        return slice(self.start, self.stop)


    def at(self, start):
        """The same layout, same type and keys, placed at another position
        in the register of a block.

        Parameters
        ----------
        start : int
            index of the first channel

        Returns
        -------
        carrier : Carrier
            copy of the carrier starting at 'start'
        """
        if not isinstance(start, int) or start < 0:
            raise ValueError(f"Start '{start}' must be a positive int!")
        placed = copy.copy(self)
        placed.start, placed.stop = start, start + self.size
        return placed


class Vector(Carrier):
    """Carrier for a fixed number of unnamed channels.

    Channels are addressed by position only, which makes this the carrier
    for values that have a size but no component names, such as a spatial
    discretization

    .. code-block:: python

        v = Vector(3)
        v[0]  # -> 0
        v[2]  # -> 2


    Parameters
    ----------
    size : int
        number of channels
    start : int
        index of the first channel in the register of the block
    """

    def __init__(self, size, start=0):

        #size validation
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"Size '{size}' must be a positive int!")

        self._size = size
        super().__init__(start)


    def __repr__(self):
        return f"{self.__class__.__name__}({self.size}, {self.start})"


    @property
    def size(self):
        """Number of channels covered by the carrier

        Returns
        -------
        size : int
            number of channels
        """
        return self._size


# FUNCTIONS =============================================================================

def ports(carrier, names, start=0):
    """Port labels for ports of one layout, placed one after another in the
    register of a block from 'start' on. With a carrier every port is a
    carrier of its layout, without one every port is a single channel.

    .. code-block:: python

        ports(Species(), ["in_0", "in_1"])  # -> {"in_0": Species(0), "in_1": Species(3)}
        ports(None, ["in_0", "in_1"])       # -> {"in_0": 0, "in_1": 1}

    Parameters
    ----------
    carrier : Carrier | None
        layout of the ports
    names : list[str]
        names of the ports
    start : int
        index of the first channel of the first port

    Returns
    -------
    labels : dict[str, int | Carrier]
        port labels for the register of a block
    """
    if carrier is None:
        return {name: start + i for i, name in enumerate(names)}
    return {name: carrier.at(start + i * carrier.size) for i, name in enumerate(names)}
