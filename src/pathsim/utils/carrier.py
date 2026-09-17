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

import numpy as np


# CLASSES ===============================================================================

class Carrier:
    """Named range of scalar channels of a block register.

    A carrier describes a contiguous block of channels and gives each of them
    a name. It is placed at an explicit position ('start') in the register of
    a block and used as the value of a port label

    .. code-block:: python

        class Mixer(Block):
            def __init__(self):
                super().__init__()
                _in_1 = Carrier(keys=("F", "T"))
                _in_2 = Carrier(_in_1.stop, keys=("F", "T"))
                self.input_port_labels = {"in_1": _in_1, "in_2": _in_2}


    The carrier only exists while the system graph is assembled. Component
    access resolves to plain integer indices, either by attribute or by key

    .. code-block:: python

        _in_1.F     # -> 0
        _in_1["T"]  # -> 1
        _in_1[1]    # -> 1


    so the values themselves stay in the flat float64 register of the block
    and nothing carrier specific is left at runtime

    .. code-block:: python

        def func(u):
            return u[_in_1.F] + u[_in_2.F]


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

    #names that would collide with the carrier interface itself
    _reserved = ("keys", "start", "stop", "size")

    def __init__(self, start=0, keys=None):

        #channel names, from the arguments or from the class
        if keys is not None:
            self.keys = tuple(keys)

        #key validation
        for key in self.keys:
            if not isinstance(key, str):
                raise ValueError(f"Key '{key}' must be str but is '{type(key)}'!")
            if key in self._reserved or key.startswith("_"):
                raise ValueError(f"Key '{key}' is reserved!")
        if len(set(self.keys)) < len(self.keys):
            raise ValueError("Keys must be unique!")

        #position validation
        if not isinstance(start, int) or start < 0:
            raise ValueError(f"Start '{start}' must be a positive int!")

        self.start = start
        self.stop = start + self.size

        #channel indices as plain attributes, resolved once
        for i, key in enumerate(self.keys):
            setattr(self, key, start + i)


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


    def __iter__(self):
        """Iteration over the channel indices in the register"""
        return iter(range(self.start, self.stop))


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
            return getattr(self, key)

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


    def to_array(self, values):
        """Build a flat array for the channels of the carrier from a mapping
        of keys to values. Keys that are not provided stay zero.

        Parameters
        ----------
        values : dict[str: float]
            values for the channels, by key

        Returns
        -------
        arr : np.ndarray
            values in the channel order of the carrier
        """
        _arr = np.zeros(self.size)
        for key, val in values.items():
            _arr[self[key] - self.start] = val
        return _arr


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
