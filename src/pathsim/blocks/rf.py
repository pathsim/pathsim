#########################################################################################
##
##                                     RF BLOCK
##                                  (blocks/rf.py)
##
##                  N-port RF network linear time invariant (LTI)
##               multi input multi output (MIMO) state-space model.
##
#########################################################################################

# TODO LIST
# class RFAmplifier Model amplifier in RF systems
# class Resistor/Capacitor/Inductor
# class RFMixer for mixer in RF systems?


# IMPORTS ===============================================================================
from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING, TypeVar

try:
    import skrf as rf

    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

if TYPE_CHECKING and HAS_SKRF:
    NetworkType = rf.Network
else:
    NetworkType = TypeVar("NetworkType")

from inspect import signature
from pathlib import Path

from .lti import StateSpace


# BLOCK DEFINITIONS =====================================================================


class RFNetwork(StateSpace):
    """N-port RF network linear time invariant (LTI) multi input multi output (MIMO) state-space model.

    Uses Vector Fitting for rational approximation of the frequency response using poles and residues.
    The resulting approximation has guaranteed stable poles that are real or come in complex conjugate pairs.

    Assumes N inputs and N outputs, where N is the number of ports of the RF network.

    Note
    ----
    This block requires scikit-rf [skrf]_ to be installed. Its an optional dependency of pathsim,
    to install it:

    .. code-block::

        pip install scikit-rf

    Parameters
    ----------
    ntwk : can be :py:class:`~skrf.network.Network`, str, Path, or file-object.
        scikit-rf [skrf]_ RF Network object, or file to load information from.
        Supported formats are touchstone file V1 (.s?p) or V2 (.ts).

    References
    ----------
    .. [skrf] scikit-rf webpage https://scikit-rf.org/

    """

    def __init__(self, ntwk: NetworkType | str | Path, auto_fit: bool = True, **kwargs):
        # Check if 'skrf' is installed, its an optional dependency,
        # dont raise error at import but at initialization
        if not HAS_SKRF:
            _msg = "The scikit-rf package is required to use this block -> 'pip install scikit-rf'"
            raise ImportError(_msg)

        if isinstance(ntwk, Path) or isinstance(ntwk, str):
            ntwk = rf.Network(ntwk)

        # Select the vector fitting function from scikit-rf
        vf_fun_name = "auto_fit" if auto_fit else "vector_fit"
        vf_fun = getattr(rf.VectorFitting, vf_fun_name)
        # Filter kwargs for the selected vf function
        vf_fun_keys = signature(vf_fun).parameters
        vf_kwargs = {k: v for k, v in kwargs.items() if k in vf_fun_keys}
        # Apply vector fitting
        vf = rf.VectorFitting(ntwk)
        getattr(vf, vf_fun_name)(**vf_kwargs)
        A, B, C, D, _ = vf._get_ABCDE()
        # keep a copy of the network and VF
        self.network = ntwk
        self.vf = vf

        super().__init__(A, B, C, D)

    def s(self, freqs: np.ndarray) -> np.ndarray:
        """
        S-matrix of the vector fitted N-port model calculated from its state-space representation.

        Parameters
        ----------
        freqs : :py:class:`~numpy.ndarray`
            Frequencies (in Hz) at which to calculate the S-matrices.

        Returns
        -------
        s : :py:class:`~numpy.ndarray`
            Complex-valued S-matrices (fxNxN) calculated at frequencies `freqs`.
        """
        return rf.VectorFitting._get_s_from_ABCDE(
            freqs, self.A, self.B, self.C, self.D, 0
        )
