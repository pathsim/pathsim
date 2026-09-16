#########################################################################################
##
##                        SYSTEM LEVEL LINEARIZATION ASSEMBLY
##                            (optim/linearization.py)
##
##        Assembles the local linear models of the blocks into a single global
##            state space model '(A, B, C, D)' for an interconnected diagram
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ..exceptions import LinearizationError


# HELPERS ===============================================================================

def _block_keys(blocks):
    """Canonical identifier per block, using a 'ClassName_index' scheme to
    disambiguate multiple instances of the same block type.

    The keys are assigned once over all blocks of the system, so the state,
    input and output labels of the assembled model all refer to the same block
    by the same name.

    Parameters
    ----------
    blocks : list[Block]
        all blocks of the system

    Returns
    -------
    keys : dict[Block: str]
        canonical identifier per block
    """
    counters, keys = {}, {}
    for blk in blocks:
        name = blk.__class__.__name__
        idx = counters.get(name, 0)
        counters[name] = idx + 1
        keys[blk] = f"{name}_{idx}"
    return keys


def _state_labels(x_layout, keys):
    """One label per state row, in the same order as the assembled 'A' and 'B'.
    Blocks contributing more than one state get one label per state index.

    Parameters
    ----------
    x_layout : list[tuple[Block, int]]
        contributing blocks and their state count, in state vector order
    keys : dict[Block: str]
        canonical identifiers from '_block_keys'

    Returns
    -------
    labels : list[str]
        one label per state
    """
    labels = []
    for blk, nx in x_layout:
        if nx == 1:
            labels.append(keys[blk])
        else:
            labels.extend(f"{keys[blk]}[{i}]" for i in range(nx))
    return labels


def _port_labels(port_refs, keys, inputs=True):
    """One label per resolved channel across a list of 'PortReference', in the
    same order as the assembled 'B' columns or 'C' and 'D' rows.

    Ports that are carriers get one label per channel, named by the channel
    key or by the channel position for carriers without keys.

    Parameters
    ----------
    port_refs : list[PortReference]
        break points or tap points
    keys : dict[Block: str]
        canonical identifiers from '_block_keys'
    inputs : bool
        resolve the ports in the block inputs, otherwise in the block outputs

    Returns
    -------
    labels : list[str]
        one label per channel
    """
    labels = []
    for pr in port_refs:
        register = pr.block.inputs if inputs else pr.block.outputs

        #single channel -> block identifier only
        if pr._size(register) == 1:
            labels.append(keys[pr.block])
            continue

        for p in pr.ports:
            carrier = None if isinstance(p, tuple) else pr._get_carrier(p, register)
            if carrier is None:
                name = f"{p[0]}.{p[1]}" if isinstance(p, tuple) else p
                labels.append(f"{keys[pr.block]}[{name}]")
            else:
                names = carrier.keys or range(len(carrier))
                labels.extend(f"{keys[pr.block]}[{p}.{k}]" for k in names)
    return labels


# GLOBAL STATE SPACE ASSEMBLY ===========================================================

def assemble_from_ports(blocks, connections, in_cols, out_rows, t):
    """Assemble a global linear state space model from an explicit port level
    description of where the system is driven and where it is measured.

    This is the core of the assembly, see 'assemble_statespace' for the
    algorithm and for the usual entry point that works on 'PortReference'.
    The explicit form exists because the interface of a 'Subsystem' drives
    several internal ports from one external input, which a one column per
    port mapping cannot express.

    Parameters
    ----------
    blocks : list[Block]
        all blocks of the system
    connections : list[Connection]
        all connections of the system
    in_cols : list[list[tuple[Block, int]]]
        one entry per input column, each holding the (block, input row) pairs
        that this external input drives. Existing incoming connections at
        those ports are cut
    out_rows : list[tuple[Block, int] | None]
        one (block, output row) pair per output row of the model, 'None' for
        an output that nothing drives
    t : float
        evaluation time for the linearization

    Returns
    -------
    A, B, C, D : np.ndarray
        global state space matrices
    x_layout : list[tuple[Block, int]]
        contributing blocks and their state count, in state vector order

    Raises
    ------
    LinearizationError
        if a block has no linear model, or if the diagram is not well posed
    """

    #resolve the break points -> set of (block, input row) pairs to cut
    broken = {pair for col in in_cols for pair in col}

    #collect the local models first, the state layout follows from what the
    #blocks report and not from their integration engines. A 'Subsystem'
    #carries only a dummy engine whose state is never written, its real states
    #live in the internal blocks and only its local model knows about them
    models = {blk: blk.to_statespace(t) for blk in blocks}

    #column layout of the internal signal vectors 'v' (all block inputs)
    #and 'w' (all block outputs), plus the global state vector 'x'
    v_slices, w_slices, x_slices = {}, {}, {}
    x_layout = []
    n_v = n_w = n_x = 0
    for blk in blocks:
        n_in, n_out = len(blk.inputs.to_array()), len(blk.outputs.to_array())
        v_slices[blk] = slice(n_v, n_v + n_in)
        w_slices[blk] = slice(n_w, n_w + n_out)
        n_v, n_w = n_v + n_in, n_w + n_out

        nx = models[blk][0].shape[0]
        if nx:
            x_slices[blk] = slice(n_x, n_x + nx)
            x_layout.append((blk, nx))
            n_x += nx

    #block diagonal stack of the local models
    A_b, B_b = np.zeros((n_x, n_x)), np.zeros((n_x, n_v))
    C_b, D_b = np.zeros((n_w, n_x)), np.zeros((n_w, n_v))
    for blk in blocks:
        _A, _B, _C, _D = models[blk]

        _v, _w = v_slices[blk], w_slices[blk]
        _x = x_slices.get(blk)

        if _x is not None:
            A_b[_x, _x] = _A
            B_b[_x, _v] = _B
            C_b[_w, _x] = _C
        D_b[_w, _v] = _D

    #interconnection matrix 'L', broken target ports are left open
    L = np.zeros((n_v, n_w))
    for con in connections:
        src_rows = con.source._get_output_indices()
        w_off = w_slices[con.source.block].start
        for trg in con.targets:
            v_off = v_slices[trg.block].start
            for src, dst in zip(src_rows, trg._get_input_indices()):
                if (trg.block, int(dst)) in broken:
                    continue
                L[v_off + int(dst), w_off + int(src)] = 1.0

    #external input matrix 'M', one column may drive several ports
    M = np.zeros((n_v, len(in_cols)))
    for col, pairs in enumerate(in_cols):
        for blk, row in pairs:
            M[v_slices[blk].start + row, col] = 1.0

    #eliminate the internal signals in one solve instead of forming the inverse
    LC, LD = L @ C_b, L @ D_b
    try:
        GLC = np.linalg.solve(np.eye(n_v) - LD, LC)
        GM = np.linalg.solve(np.eye(n_v) - LD, M)
    except np.linalg.LinAlgError:
        raise LinearizationError(
            "System is not well posed for linearization, an algebraic loop "
            "with unity gain survives the input break. Mark an input point "
            "that breaks the loop."
            ) from None

    #output selection matrix 'S' picks the tapped rows out of 'w'
    S = np.zeros((len(out_rows), n_w))
    for row, tap in enumerate(out_rows):
        if tap is None:
            continue
        blk, out = tap
        S[row, w_slices[blk].start + out] = 1.0

    #close the loop
    return (
        A_b + B_b @ GLC,
        B_b @ GM,
        S @ (C_b + D_b @ GLC),
        S @ (D_b @ GM),
        x_layout
        )


def assemble_statespace(blocks, connections, inputs, outputs, t):
    """Assemble a global linear state space model of an interconnected block
    diagram around its current operating point.

    Every block contributes its local model through 'Block.to_statespace', which
    is a pure query and leaves the blocks untouched. The local models are
    stacked block diagonally, the connections are expressed as an interconnection
    matrix, and the internal signals are eliminated in one linear solve

    .. math::

        \\begin{align}
        \\dot{x} &= \\mathbf{A}_b x + \\mathbf{B}_b v\\\\
               w &= \\mathbf{C}_b x + \\mathbf{D}_b v\\\\
               v &= \\mathbf{L} w + \\mathbf{M} u
        \\end{align}


    where 'v' collects all block inputs, 'w' all block outputs, 'L' the internal
    connections and 'M' the external inputs. Substituting gives

    .. math::

        (\\mathbf{I} - \\mathbf{L}\\mathbf{D}_b) v
            = \\mathbf{L}\\mathbf{C}_b x + \\mathbf{M} u


    which is solved directly. Unlike a substitution along a topological order
    this handles algebraic loops that survive the input break, and well
    posedness of the diagram becomes the invertibility of that matrix.

    Note
    ----
    The block operating points have to be current, evaluate the system function
    ('Simulation._update') before calling this.

    Parameters
    ----------
    blocks : list[Block]
        all blocks of the system
    connections : list[Connection]
        all connections of the system
    inputs : list[PortReference]
        break points designating free external inputs, existing incoming
        connections at these ports are cut and replaced by a free input
    outputs : list[PortReference]
        tap points designating the system outputs
    t : float
        evaluation time for the linearization

    Returns
    -------
    A, B, C, D : np.ndarray
        global state space matrices, shaped (nx,nx), (nx,nu), (ny,nx), (ny,nu)
    state_labels, input_labels, output_labels : list[str]
        identifiers for the rows and columns of the matrices

    Raises
    ------
    LinearizationError
        if a block has no linear model, or if the diagram is not well posed
        because an algebraic loop with unity gain survives the input break
    """

    #one input column per marked port, one output row per tapped port
    in_cols = [
        [(pr.block, int(row))] for pr in inputs for row in pr._get_input_indices()
        ]
    out_rows = [
        (pr.block, int(out)) for pr in outputs for out in pr._get_output_indices()
        ]

    A, B, C, D, x_layout = assemble_from_ports(
        blocks, connections, in_cols, out_rows, t
        )

    keys = _block_keys(blocks)

    return (
        A, B, C, D,
        _state_labels(x_layout, keys),
        _port_labels(inputs, keys, inputs=True),
        _port_labels(outputs, keys, inputs=False)
        )
