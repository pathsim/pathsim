#########################################################################################
##
##                                   Register Class
##                            (pathsim/utils/register.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .carrier import Carrier


# CLASSES ===============================================================================

class Register:
    """This class is a intended to be used for the inputs and outputs of blocks. 

    Its basic functionality is similar to a `dict` but with some additional methods 
    and implemented as a numpy array for fast data transfer. 
    
    The core functionality is that values can be added dynamically and the size of the 
    register doesnt have to be specified. It also implements some methods to interact 
    with numpy arrays and to streamline convergence checks.
    
    Port aliases map to a single channel (int) or to a `Carrier` that groups
    multiple channels. Carriers are resolved to slices of the register when
    it is created, so the register itself only deals with integer indices and
    slices of its flat data array.

    Parameters
    ----------
    size : int, optional
        initial size of the register, at least the size of the port aliases
    mapping : dict[str: int, str: Carrier]
        string aliases for integer ports or carriers

    Attributes
    ----------
    _data : np.ndarray
        internal numpy array that holds the values
    _mapping : dict[str: int, str: slice]
        internal mapping for port aliases from string to int (index) or
        to slice (channels of a carrier)
    _carriers : dict[str: Carrier]
        carriers of the port aliases, for access to single channels
    """
    
    __slots__ = ["_data", "_mapping", "_carriers"]
    
    def __init__(self, size=None, mapping=None, dtype=np.float64):

        self._mapping = {}
        self._carriers = {}

        #resolve port aliases, carriers to slices
        _size = 1 if size is None else size
        for name, port in (mapping or {}).items():
            if isinstance(port, Carrier):
                self._mapping[name] = port.slice
                self._carriers[name] = port
                _size = max(_size, port.stop)
            else:
                self._mapping[name] = port
                _size = max(_size, port + 1)

        self._validate_carriers()

        self._data = np.zeros(_size, dtype=dtype)


    def _validate_carriers(self):
        """Check that the channels of the carriers dont overlap with each
        other or with single channel ports. Raises a ValueError if they do.
        """
        if not self._carriers:
            return

        used = {}
        for name, port in self._mapping.items():
            _channels = range(port.start, port.stop) if isinstance(port, slice) else [port]
            for c in _channels:
                other = used.get(c)

                #single channel ports may alias each other, carriers may not
                if other is not None and (name in self._carriers or other in self._carriers):
                    raise ValueError(f"Ports '{other}' and '{name}' overlap at channel {c}!")
                used[c] = name
    
    
    def _map(self, key):
        """Map string keys to integers defined in '_mapping'

        Parameters
        ----------
        key : int, str
            port key, to map to index

        Returns
        -------
        _key : int, slice
            port index or channels of a carrier
        """
        return self._mapping.get(key, key)


    def _resolve(self, ports):
        """Resolve ports to channel indices, ports that are carriers
        are expanded to all of their channels.

        Parameters
        ----------
        ports : list[int, str, tuple[str, int | str]]
            port indices, port names or single channels of carriers
            as tuple of port name and channel key

        Returns
        -------
        indices : np.ndarray[int]
            channel indices of the ports
        """
        indices = []
        for p in ports:

            #single channel of a carrier
            if isinstance(p, tuple):
                indices.append(self._carriers[p[0]][p[1]])
                continue

            _p = self._mapping.get(p, p)
            if isinstance(_p, slice):
                indices.extend(range(_p.start, _p.stop))
            else:
                indices.append(_p)

        return np.array(indices, dtype=np.intp)
    

    def _get_max_index(self, key):
        """Identify max index from different key types."""
        if isinstance(key, int):
            return key
        elif isinstance(key, slice):
            return key.stop - 1 if key.stop is not None else -1
        elif isinstance(key, (list, tuple, np.ndarray)):
            return max(key) if len(key) else -1
        return -1
    

    def __len__(self):
        return len(self._data)
    

    def __iter__(self):
        """Iteration and unpacking into tuples or lists"""
        return iter(self._data)
    

    def __getitem__(self, key):
        """Get the value for direct access to the 
        register values.
        
        Parameters
        ----------
        key : int, str
            port key, where to get value from

        Returns
        -------
        out : float, np.ndarray, obj
            value from port at `key` position, array of values
            for a port that is a `Carrier`
        """
        if isinstance(key, str):
            key = self._mapping.get(key)
            if key is None:
                return 0.0

            #carrier -> copy of all of its channels
            if isinstance(key, slice):
                return self._data[key].copy()
        
        if isinstance(key, int):
            if key < 0 or key >= len(self._data):
                return 0.0
            return self._data[key]
        
        return self._data[key]
    

    def __setitem__(self, key, value):
        """Set the value at key index for direct access
        to the register values.

        Parameters
        ----------
        key : int, str
            port key, where to set value
        val : float, obj
            value to set at port
        """
        if isinstance(key, str):
            key = self._mapping.get(key, key)

        if isinstance(key, int):
            if key >= len(self._data):
                self.resize(key + 1)

            #convert to scalar if needed to avoid numpy deprecation warning
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()

        else:
            self.resize(self._get_max_index(key) + 1)

            #convert to scalar if needed to avoid numpy deprecation warning
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()

        self._data[key] = value


    def resize(self, size):
        """Resize the internal data array to accommodate more entries.

        Creates a new zero-filled array instead of in-place resize to avoid
        numpy ValueError when other references to the array exist (e.g., from
        fancy indexing in PortReference).

        Parameters
        ----------
        size : int
            new size for the internal data array
        """
        if size > len(self._data):
            new_data = np.zeros(size)
            new_data[:len(self._data)] = self._data
            self._data = new_data          


    def reset(self):
        """Set all stored values to zero."""
        self._data[:] = 0.0
    

    def to_array(self):
        """Returns a copy of the internal array.

        Returns
        -------
        arr : np.ndarray
            converted register as array
        """
        return self._data.copy()
    

    def update_from_array(self, arr):
        """Update the register values from an array in place.

        Parameters
        ----------
        arr : np.ndarray, list, tuple, float
            array or scalar that is used to update internal register values
        """
        if isinstance(arr, np.ndarray):

            #0-d arrays (e.g. from scipy interpolators called on a scalar) have
            #no length and are semantically scalars, not sequences
            if arr.ndim == 0:
                self._data[0] = arr.item()
                return

        elif not isinstance(arr, (list, tuple)):
            self._data[0] = arr
            return

        n = len(arr)
        if n > len(self._data):
            self.resize(n)
        self._data[:n] = arr

    
    def __contains__(self, key):
        """Check if a key is in mapping or is valid integer index. Single
        channels of carriers are keys as tuple of port name and channel key.
        """
        if isinstance(key, tuple):
            carrier = self._carriers.get(key[0])
            if carrier is None:
                return False
            try:
                carrier[key[1]]
            except ValueError:
                return False
            return True

        return key in self._mapping or isinstance(key, int)