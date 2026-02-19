#########################################################################################
##
##                              MUTABLE PARAMETER DECORATOR
##                                 (utils/mutable.py)
##
##         Class decorator that enables runtime parameter mutation with automatic
##         reinitialization. When a decorated parameter is changed, the block's
##         __init__ is re-run with updated values while preserving engine state.
##
#########################################################################################

# IMPORTS ===============================================================================

import inspect
import functools

import numpy as np


# REINIT HELPER =========================================================================

def _do_reinit(block):
    """Re-run __init__ with current parameter values, preserving engine state.

    Uses ``type(block).__init__`` to always reinit from the most derived class,
    ensuring that subclass overrides (e.g. operator replacements) are preserved.

    Parameters
    ----------
    block : Block
        the block instance to reinitialize
    """

    actual_cls = type(block)

    # gather current values for ALL init params of the actual class
    sig = inspect.signature(actual_cls.__init__)
    kwargs = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if hasattr(block, name):
            kwargs[name] = getattr(block, name)

    # save engine
    engine = block.engine if hasattr(block, 'engine') else None

    # re-run init through the wrapped __init__ (handles depth counting)
    block._param_locked = False
    actual_cls.__init__(block, **kwargs)
    # _param_locked is set to True by the outermost new_init wrapper

    # restore engine
    if engine is not None:
        old_dim = len(engine)
        new_dim = len(np.atleast_1d(block.initial_value)) if hasattr(block, 'initial_value') else 0

        if old_dim == new_dim:
            # same dimension - restore the entire engine
            block.engine = engine
        else:
            # dimension changed - create new engine inheriting settings
            block.engine = type(engine).create(
                block.initial_value,
                parent=engine.parent,
            )
            block.engine.tolerance_lte_abs = engine.tolerance_lte_abs
            block.engine.tolerance_lte_rel = engine.tolerance_lte_rel


# DECORATOR =============================================================================

def mutable(cls):
    """Class decorator that makes all ``__init__`` parameters trigger automatic
    reinitialization when changed at runtime.

    Parameters are auto-detected from the ``__init__`` signature. When any parameter
    is changed at runtime, the block's ``__init__`` is re-executed with updated values.
    The integration engine state is preserved across reinitialization.

    A ``set(**kwargs)`` method is also generated for batched parameter updates that
    triggers only a single reinitialization.

    Supports inheritance: if both a parent and child class use ``@mutable``, the init
    guard uses a depth counter to ensure reinitialization only triggers after the
    outermost ``__init__`` completes.

    Example
    -------
    .. code-block:: python

        @mutable
        class PT1(StateSpace):
            def __init__(self, K=1.0, T=1.0):
                self.K = K
                self.T = T
                super().__init__(
                    A=np.array([[-1.0 / T]]),
                    B=np.array([[K / T]]),
                    C=np.array([[1.0]]),
                    D=np.array([[0.0]])
                )

        pt1 = PT1(K=2.0, T=0.5)
        pt1.K = 5.0                    # auto reinitializes
        pt1.set(K=5.0, T=0.3)         # single reinitialization
    """

    original_init = cls.__init__

    # auto-detect all __init__ parameters
    params = [
        name for name in inspect.signature(original_init).parameters
        if name != "self"
        ]

    # -- install property descriptors for all params -------------------------------

    for name in params:
        storage = f"_p_{name}"

        def _make_property(s):
            def getter(self):
                return getattr(self, s)

            def setter(self, value):
                setattr(self, s, value)
                if getattr(self, '_param_locked', False):
                    _do_reinit(self)

            return property(getter, setter)

        setattr(cls, name, _make_property(storage))

    # -- wrap __init__ with depth counter ------------------------------------------

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        self._init_depth = getattr(self, '_init_depth', 0) + 1
        try:
            original_init(self, *args, **kwargs)
        finally:
            self._init_depth -= 1
            if self._init_depth == 0:
                self._param_locked = True

    cls.__init__ = new_init

    # -- generate batched set() method ---------------------------------------------

    def set(self, **kwargs):
        """Set multiple parameters and reinitialize once.

        Parameters
        ----------
        kwargs : dict
            parameter names and their new values

        Example
        -------
        .. code-block:: python

            block.set(K=5.0, T=0.3)
        """
        self._param_locked = False
        for key, value in kwargs.items():
            setattr(self, key, value)
        _do_reinit(self)

    cls.set = set

    # -- store metadata for introspection ------------------------------------------

    existing = getattr(cls, '_mutable_params', ())
    cls._mutable_params = existing + tuple(p for p in params if p not in existing)

    return cls
