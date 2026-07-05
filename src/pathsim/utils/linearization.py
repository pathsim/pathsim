#########################################################################################
##
##                       SYSTEM-LEVEL LINEARIZATION ASSEMBLY
##                            (utils/linearization.py)
##
##       Assembles per-block Jacobians into a single global state-space model
##                  '(A, B, C, D)' for an interconnected block diagram
##
#########################################################################################

# IMPORTS ===============================================================================

from collections import namedtuple

import numpy as np

from ..connection import Connection
from ..optim.operator import DynamicOperator
from ..optim.numerical import num_jac
from .graph import Graph
from .portreference import PortReference


# RESULT CONTAINER =======================================================================

LinearizationResult = namedtuple(
    "LinearizationResult",
    ["A", "B", "C", "D", "state_labels", "input_labels", "output_labels"]
    )


# PER-BLOCK LINEARIZATION ADAPTER =======================================================

def _numeric_dynamics_jacobian(block, t, nx, nu):
    """Numerical (A, B) for dynamic blocks without an 'op_dyn' operator
    (e.g. 'Integrator'), by central-differencing 'block.derivative(t)'
    through the block's own state / input registers.
    """
    x0 = np.array(np.atleast_1d(block.state), dtype=float, copy=True)
    u0 = block.inputs.to_array().copy()

    def f_of_x(x):
        block.state = x
        return np.atleast_1d(block.derivative(t))

    def f_of_u(u):
        block.inputs.update_from_array(u)
        return np.atleast_1d(block.derivative(t))

    A = num_jac(f_of_x, x0) if nx else np.zeros((0, 0))
    block.state = x0

    B = num_jac(f_of_u, u0) if nu else np.zeros((nx, 0))
    block.inputs.update_from_array(u0)

    return np.asarray(A).reshape(nx, nx), np.asarray(B).reshape(nx, nu)


def _numeric_output_jacobian(block, t, nx, nu, ny):
    """Numerical (C, D) for blocks without an 'op_alg' operator (e.g. 'ODE',
    whose output is an implicit identity 'y = x'), by central-differencing
    'block.outputs' after perturbing state / inputs and calling 'block.update(t)'.
    """
    x0 = np.array(np.atleast_1d(block.state), dtype=float, copy=True) if nx else None
    u0 = block.inputs.to_array().copy()

    def g_of_x(x):
        block.state = x
        block.update(t)
        return block.outputs.to_array().copy()

    def g_of_u(u):
        block.inputs.update_from_array(u)
        block.update(t)
        return block.outputs.to_array().copy()

    C = num_jac(g_of_x, x0) if nx else np.zeros((ny, 0))
    if nx:
        block.state = x0

    D = num_jac(g_of_u, u0) if nu else np.zeros((ny, 0))
    block.inputs.update_from_array(u0)

    #restore outputs to the pre-perturbation operating point
    block.update(t)

    return np.asarray(C).reshape(ny, nx), np.asarray(D).reshape(ny, nu)


def linearize_block(block, t):
    """Return the local '(A, B, C, D)' Jacobians of a single block at its
    current operating point.

    Dispatches on the block's internal operator setup (see 'Block.linearize')
    rather than the block's concrete type, so it works uniformly across the
    different patterns found in the block library: full dynamic blocks with
    both 'op_dyn'/'op_alg' (e.g. 'StateSpace', 'DynamicalSystem'), dynamic
    blocks with only 'op_dyn' and an implicit output (e.g. 'ODE'), dynamic
    blocks with neither operator (e.g. 'Integrator'), and stateless algebraic
    blocks with only 'op_alg' (either a plain 'Operator' or a 'DynamicOperator'
    called with 'x=None').

    Parameters
    ----------
    block : Block
        block to linearize
    t : float
        evaluation time

    Returns
    -------
    A, B, C, D : np.ndarray
        local state-space matrices, shaped (nx,nx), (nx,nu), (ny,nx), (ny,nu)
    """
    u = block.inputs.to_array()
    nu = len(u)
    ny = len(block.outputs.to_array())
    has_state = block.engine is not None
    nx = len(np.atleast_1d(block.state)) if has_state else 0

    #run the block's own linearization dispatch (handles the Operator vs
    #DynamicOperator / stateless vs stateful distinction)
    block.linearize(t)

    #dynamics: A, B
    if not has_state:
        A, B = np.zeros((0, 0)), np.zeros((0, nu))
    elif block.op_dyn is not None:
        A = np.asarray(block.op_dyn.Jx).reshape(nx, nx)
        B = np.asarray(block.op_dyn.Ju).reshape(nx, nu)
    else:
        A, B = _numeric_dynamics_jacobian(block, t, nx, nu)

    #output map: C, D
    if block.op_alg is None:
        C, D = _numeric_output_jacobian(block, t, nx, nu, ny)
    elif isinstance(block.op_alg, DynamicOperator):
        C = np.asarray(block.op_alg.Jx).reshape(ny, nx) if has_state else np.zeros((ny, 0))
        D = np.asarray(block.op_alg.Ju).reshape(ny, nu)
    else:
        C = np.zeros((ny, 0))
        D = np.asarray(block.op_alg.J).reshape(ny, nu)

    return A, B, C, D


# CONNECTION BREAKING ====================================================================

def _break_connections(connections, broken):
    """Return a new connection list with every (block, input-row) pair in
    'broken' removed from each connection's targets.

    Connections are rebuilt (rather than mutated) since a single 'Connection'
    may fan out to several targets sharing one source-port list; only the
    affected target/port pairs are dropped, everything else is preserved.

    Parameters
    ----------
    connections : list[Connection]
        original connections to filter
    broken : set[tuple[Block, int]]
        (block, resolved input row) pairs to cut

    Returns
    -------
    list[Connection]
        new connections with the broken ports removed
    """
    new_connections = []
    for con in connections:
        src_ports = con.source.ports
        for trg in con.targets:
            trg_idx = trg._get_input_indices()
            keep = [
                i for i, row in enumerate(trg_idx)
                if (trg.block, int(row)) not in broken
                ]
            if not keep:
                continue
            new_src_ports = [src_ports[i] for i in keep]
            new_trg_ports = [trg.ports[i] for i in keep]
            new_connections.append(Connection(
                PortReference(con.source.block, new_src_ports),
                PortReference(trg.block, new_trg_ports)
                ))
    return new_connections


# LABELING ================================================================================
#
# Human-readable identifiers for the rows/columns of the assembled matrices,
# following the same 'ClassName_index' scheme 'Simulation._checkpoint_key'
# already uses to disambiguate multiple instances of the same block type.
# These are exactly the 'states=' / 'inputs=' / 'outputs=' kwargs expected by
# python-control's 'control.StateSpace', not just cosmetic metadata.

def _state_labels(blocks_dyn):
    """One label per state row, in the same order as the assembled 'A'/'B'.
    Multi-state blocks get one label per internal state index.
    """
    counters = {}
    labels = []
    for blk in blocks_dyn:
        name = blk.__class__.__name__
        idx = counters.get(name, 0)
        counters[name] = idx + 1
        key = f"{name}_{idx}"

        nx = len(np.atleast_1d(blk.state))
        if nx == 1:
            labels.append(key)
        else:
            labels.extend(f"{key}[{i}]" for i in range(nx))
    return labels


def _port_ref_labels(port_refs):
    """One label per resolved port across a list of 'PortReference', in the
    same order as the assembled 'B'/'C'/'D' columns or rows. The same block
    referenced by more than one 'PortReference' keeps the same key.
    """
    counters, keys = {}, {}
    labels = []
    for pr in port_refs:
        blk = pr.block
        if blk not in keys:
            name = blk.__class__.__name__
            idx = counters.get(name, 0)
            counters[name] = idx + 1
            keys[blk] = f"{name}_{idx}"
        key = keys[blk]

        if len(pr.ports) == 1:
            labels.append(key)
        else:
            labels.extend(f"{key}[{p}]" for p in pr.ports)
    return labels


# GLOBAL STATE-SPACE ASSEMBLY ============================================================

def assemble_linear_system(blocks, connections, blocks_dyn, inputs, outputs, t):
    """Assemble a global linear state-space model '(A, B, C, D)' for an
    interconnected block diagram, around the current operating point.

    Parameters
    ----------
    blocks : list[Block]
        all blocks of the system
    connections : list[Connection]
        all connections of the system (unbroken)
    blocks_dyn : list[Block]
        the subset of 'blocks' that carry real integration state (have an
        '.engine') -- these define the state vector 'x', in order
    inputs : list[PortReference]
        break points designating free external inputs. Existing incoming
        connections at these ports are cut and replaced by a free input.
    outputs : list[PortReference]
        tap points designating system outputs
    t : float
        evaluation time for linearization

    Returns
    -------
    LinearizationResult
        namedtuple(A, B, C, D, state_labels, input_labels, output_labels).
        The label lists are exactly the 'states=', 'inputs=', 'outputs='
        kwargs expected by python-control's 'control.StateSpace'.

    Raises
    ------
    RuntimeError
        if an algebraic loop survives the input break -- the DAG-substitution
        algorithm used here requires an acyclic graph once the marked inputs
        are cut; true algebraic loops need a linear solve on the loop's
        coupling matrix, which is a Phase 2 extension, not supported here
    """
    #resolve input break points -> (block, row) set + column layout
    broken = set()
    input_slices = []
    n_u = 0
    for pr in inputs:
        input_slices.append((pr, n_u))
        for port in pr.ports:
            row = pr.block.inputs._map(port)
            broken.add((pr.block, row))
        n_u += len(pr.ports)

    #fresh temporary graph over the broken connectivity -- NOT 'sim.graph',
    #which reflects the un-broken wiring and may misclassify blocks whose
    #marked input sits on what was a loop-closing connection
    stripped = _break_connections(connections, broken)
    temp_graph = Graph(blocks, stripped)

    if temp_graph.has_loops:
        loop_blocks = sorted({
            blk.__class__.__name__
            for _, blks, _ in temp_graph.loop()
            for blk in blks
            })
        raise RuntimeError(
            "linearize_system: an algebraic loop survives the input break "
            f"(blocks involved: {loop_blocks}). Mark an input point that "
            "breaks the loop, or reduce it to an equivalent acyclic diagram."
            )

    #state layout: contiguous slice per dynamic block, in 'blocks_dyn' order
    state_rows = {}
    n_x = 0
    for blk in blocks_dyn:
        nx = len(np.atleast_1d(blk.state))
        state_rows[blk] = slice(n_x, n_x + nx)
        n_x += nx

    #per-block maps expressing each block's input as a linear combination of
    #the global state 'x' and the global external input 'u'
    Xin, Uin, Xout, Uout, jac_cache = {}, {}, {}, {}, {}
    for blk in blocks:
        n_in = len(blk.inputs.to_array())
        Xin[blk] = np.zeros((n_in, n_x))
        Uin[blk] = np.zeros((n_in, n_u))

    #apply input breaks: identity into the corresponding external input columns
    for pr, offset in input_slices:
        for i, port in enumerate(pr.ports):
            row = pr.block.inputs._map(port)
            Uin[pr.block][row, offset + i] = 1.0

    #walk the DAG in topological order, mirroring 'Simulation._dag()':
    #each block's Xin/Uin are fully resolved by the time it is visited
    for _, blks, cons in temp_graph.dag():
        for blk in blks:
            A_i, B_i, C_i, D_i = linearize_block(blk, t)
            jac_cache[blk] = (A_i, B_i, C_i, D_i)

            x_slice = state_rows.get(blk)
            ny = C_i.shape[0]
            if x_slice is not None and C_i.shape[1] > 0:
                Xc = np.zeros((ny, n_x))
                Xc[:, x_slice] = C_i
            else:
                Xc = np.zeros((ny, n_x))

            Xout[blk] = Xc + D_i @ Xin[blk]
            Uout[blk] = D_i @ Uin[blk]

        for con in cons:
            src_idx = con.source._get_output_indices()
            for trg in con.targets:
                trg_idx = trg._get_input_indices()
                for s, d in zip(src_idx, trg_idx):
                    if (trg.block, int(d)) in broken:
                        continue
                    Xin[trg.block][d, :] = Xout[con.source.block][s, :]
                    Uin[trg.block][d, :] = Uout[con.source.block][s, :]

    #reduced state dynamics: only 'blocks_dyn' contribute rows/states
    A = np.zeros((n_x, n_x))
    B = np.zeros((n_x, n_u))
    for blk in blocks_dyn:
        rows = state_rows[blk]
        A_i, B_i, _, _ = jac_cache[blk]
        A[rows, rows] += A_i
        A[rows, :] += B_i @ Xin[blk]
        B[rows, :] += B_i @ Uin[blk]

    #outputs: read off the tapped blocks' resolved Xout/Uout
    n_y = sum(len(pr) for pr in outputs)
    C = np.zeros((n_y, n_x))
    D = np.zeros((n_y, n_u))
    row = 0
    for pr in outputs:
        for i in pr._get_output_indices():
            C[row, :] = Xout[pr.block][i, :]
            D[row, :] = Uout[pr.block][i, :]
            row += 1

    return LinearizationResult(
        A, B, C, D,
        state_labels=_state_labels(blocks_dyn),
        input_labels=_port_ref_labels(inputs),
        output_labels=_port_ref_labels(outputs)
        )
