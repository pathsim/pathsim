########################################################################################
##
##                            OPTIMIZED GRAPH ANALYSIS
##
########################################################################################

# IMPORTS ==============================================================================

from collections import defaultdict, deque


# GRAPH CLASS ==========================================================================

class Graph:
    """Optimized graph representation with efficient assembly and cycle detection.

    The Graph class analyzes block diagrams represented as directed graphs to identify
    algebraic loops, compute evaluation depths, and organize blocks into levels for
    efficient simulation. Uses iterative algorithms to avoid recursion limits.

    Parameters
    ----------
    blocks : list, optional
        list of block objects to include in the graph
    connections : list, optional
        list of Connection objects defining the graph edges

    Attributes
    ----------
    has_loops : bool
        flag indicating presence of algebraic loops (cycles)

    Examples
    --------
    Create a simple graph with two blocks:

    .. code-block:: python

        from pathsim.blocks import Amplifier, Integrator
        from pathsim.connection import Connection
        from pathsim.utils.graph import Graph

        amp = Amplifier(gain=2.0)
        integ = Integrator(0.0)

        conn = Connection(amp, integ)

        graph = Graph([amp, integ], [conn])
    """

    def __init__(self, blocks=None, connections=None):
        self.blocks = list(blocks) if blocks else []
        self.connections = list(connections) if connections else []

        # First check the connections for port conflicts
        self._validate_connections()

        # loop flag
        self.has_loops = False

        # depths
        self._alg_depth = 0
        self._loop_depth = 0

        # initialize graph orderings
        self._blocks_dag = defaultdict(list)
        self._blocks_loop_dag = defaultdict(list)
        self._connections_dag = defaultdict(list)
        self._connections_loop_dag = defaultdict(list)
        self._loop_closing_connections = []

        # Build maps in single pass
        self._build_all_maps()

        # assemble dag and loops
        self._assemble()


    def __bool__(self):
        return True


    def __len__(self):
        return len(self.blocks)


    @property
    def size(self):
        """Returns the size of the graph as (number of blocks, number of connections).

        Returns
        -------
        tuple
            (number of blocks, total number of connection targets)
        """
        return len(self.blocks), sum(len(con.targets) for con in self.connections)


    @property
    def depth(self):
        """Returns the depths of the graph as (algebraic depth, loop depth).

        The algebraic depth is the maximum number of levels in the acyclic part
        of the graph. The loop depth is the maximum number of levels within
        algebraic loops.

        Returns
        -------
        tuple
            (algebraic depth, loop depth)
        """
        return self._alg_depth, self._loop_depth


    def _validate_connections(self):
        """Fast O(N) validation that no connections overwrite each other.
        
        Checks that no two connections target the same (block, channel) pair.
        Ports are resolved to channel indices first, so port names, ports that
        are carriers and single carrier channels are compared on the channels
        they actually cover.
        """
        # {(block, channel_idx): connection}
        connected_targets = set()

        for connection in self.connections:
            for target in connection.targets:
                target_block = target.block
                for port_idx in target._resolve(target_block.inputs):
                    key = (target_block, int(port_idx))
                    if key in connected_targets:
                        raise ValueError(
                            f"Connection conflict detected"
                        )
                    connected_targets.add(key)


    def _build_all_maps(self):
        """Build all connection maps in a single pass for efficiency.

        Creates internal dictionaries mapping blocks to their upstream/downstream
        neighbors and outgoing connections. Ensures deterministic ordering by sorting
        connections based on pre-computed block order.
        """

        self._alg_blocks = set()
        self._dyn_blocks = set()

        for blk in self.blocks:
            if len(blk) > 0:
                self._alg_blocks.add(blk)
            else:
                self._dyn_blocks.add(blk)

        self._upst_blk_blk_map = defaultdict(set)
        self._dnst_blk_blk_map = defaultdict(set)
        self._outg_blk_con_map = defaultdict(list)

        for con in self.connections:
            src_blk = con.source.block
            self._outg_blk_con_map[src_blk].append(con)
            
            for trg in con.targets:
                tgt_blk = trg.block
                self._dnst_blk_blk_map[src_blk].add(tgt_blk)
                self._upst_blk_blk_map[tgt_blk].add(src_blk)
            

    def _assemble(self):
        """Optimized assembly using DFS with proper cycle detection.

        Analyzes the graph structure to separate acyclic (DAG) and cyclic (loop)
        components. Computes depths for all blocks and organizes them into levels
        for efficient evaluation during simulation.
        """
        self._blocks_dag.clear()
        self._connections_dag.clear()
        self._blocks_loop_dag.clear()
        self._connections_loop_dag.clear()
        self._loop_closing_connections.clear()
        self.has_loops = False

        # No blocks -> early exit
        if not self.blocks:
            return

        # Handle dynamic blocks at depth 0
        for blk in self._dyn_blocks:
            self._blocks_dag[0].append(blk)
            for con in self._outg_blk_con_map[blk]:
                self._connections_dag[0].append(con)

        # No algebraic blocks -> early exit
        if not self._alg_blocks:
            self._alg_depth = 1
            self._loop_depth = 0
            return

        # Compute depths with cycle detection
        depths = self._compute_depths_iterative()
        
        blocks_loop = set()

        # Single pass to categorize blocks
        for blk in self._alg_blocks:
            depth = depths[blk]
            
            if depth is None:
                blocks_loop.add(blk)
                self.has_loops = True
            else:
                self._blocks_dag[depth].append(blk)
                for con in self._outg_blk_con_map[blk]:
                    self._connections_dag[depth].append(con)

        self._alg_depth = (max(self._blocks_dag) + 1) if self._blocks_dag else 0

        if self.has_loops:
            self._process_loops(blocks_loop)
        else:
            self._loop_depth = 0


    def _compute_depths_iterative(self):
        """Compute algebraic depths using iterative DFS (no recursion limit).

        Uses a stack-based depth-first search with pre-visit and post-visit phases
        to compute the maximum upstream algebraic path length for each block.
        Detects cycles by marking nodes with None depth when back edges are found.

        Returns
        -------
        dict
            mapping from blocks to their algebraic depths (None for cyclic blocks)
        """
        
        # Register states for ALL blocks
        WHITE, GRAY, BLACK = 0, 1, 2
        state = {blk: WHITE for blk in self.blocks}
        
        depths = {}
        
        for start_node in self._alg_blocks:
            if state[start_node] != WHITE:
                continue
            
            # Stack: (node, 'pre'|'post', predecessors_to_check)
            stack = [(start_node, 'pre', None)]
            
            while stack:
                node, visit_type, preds_remaining = stack.pop()

                # Using O(1) set lookup
                is_dyn = node in self._dyn_blocks
                
                if visit_type == 'pre':
                    # Pre-visit: first time seeing this node
                    
                    # Handle terminal cases 
                    if is_dyn:
                        depths[node] = 0
                        state[node] = BLACK
                        continue

                    # Already fully processed
                    if state[node] == BLACK:
                        continue
                    
                    # Back edge = cycle
                    if state[node] == GRAY:
                        depths[node] = None
                        state[node] = BLACK
                        continue
                    
                    # Mark as being processed
                    state[node] = GRAY
                                    
                    # Get predecessors and filtered algebraic
                    preds = list(self._upst_blk_blk_map[node])
                    alg_preds = [prd for prd in preds if prd in self._alg_blocks]
                    
                    if not preds:
                        # No predecessors
                        depths[node] = 0
                        state[node] = BLACK
                        continue
                    elif not alg_preds:
                        # Has predecessors, but all are dynamic
                        depths[node] = 1
                        state[node] = BLACK  
                        continue
                    
                    # Schedule post-visit after all predecessors
                    stack.append((node, 'post', preds))
                    
                    # Schedule predecessor visits (in reverse for correct order)
                    for pred in reversed(preds):
                        if state[pred] == WHITE:
                            stack.append((pred, 'pre', None))
                
                else:  # visit_type == 'post'
                    # Post-visit: all predecessors have been processed
                    
                    max_depth = 0
                    has_cycle = False
                    
                    # Check all predecessor depths
                    for pred in preds_remaining:

                        # Predecessor not finished = back edge = cycle
                        if state[pred] != BLACK:
                            has_cycle = True
                            break
                        
                        pred_depth = depths.get(pred)
                        if pred_depth is None:
                            has_cycle = True
                            break
                        
                        if pred_depth > max_depth:
                            max_depth = pred_depth
                    
                    if has_cycle:
                        depths[node] = None
                    else:
                        depths[node] = max_depth + int(not is_dyn)
                    
                    state[node] = BLACK
        
        return depths


    def _process_loops(self, blocks_loop):
        """Optimized loop processing with minimal overhead.

        Finds strongly connected components (SCCs) within the loop blocks, determines
        entry points for each SCC, and performs BFS to assign local depths. Identifies
        loop-closing connections (back edges) that need special handling.

        Parameters
        ----------
        blocks_loop : set
            set of blocks that are part of algebraic loops
        """
        if not blocks_loop:
            return

        # Find SCCs (already optimized)
        sccs = self._find_strongly_connected_components(blocks_loop)
        
        current_depth = 0

        for scc in sccs:
            scc_set = set(scc)
            
            # Pre-filter downstream neighbors for this SCC once
            scc_neighbors = {}
            for blk in scc:
                neighbors = self._dnst_blk_blk_map.get(blk, ())
                # Filter and sort once, store as list
                scc_neighbors[blk] = [n for n in neighbors if n in scc_set]
            
            # Find entry points efficiently
            entry_points = []
            for blk in scc:
                pred = self._upst_blk_blk_map.get(blk, set())
                # Quick check: if any predecessor not in SCC, it's an entry point
                has_external = any(p not in scc_set for p in pred)
                has_internal = any(p in scc_set for p in pred)
                
                if has_external or not has_internal:
                    entry_points.append(blk)
            
            if not entry_points:
                entry_points = [scc[0]]
            
            # Optimized BFS: single-pass with correct visitation
            local_depths = {}
            max_local_depth = 0
            queue = deque()
            
            # Initialize with entry points
            for ep in entry_points:
                local_depths[ep] = 0
                queue.append((ep, 0))
            
            while queue:
                blk, depth = queue.popleft()
                
                # Skip if we've already processed this node at a shallower depth
                if depth > local_depths.get(blk, float('inf')):
                    continue

                if depth > max_local_depth:
                    max_local_depth = depth
                
                # Process neighbors (already filtered and in cache)
                for next_blk in scc_neighbors.get(blk, []):
                    next_depth = depth + 1
                    
                    # Only enqueue if we found a shorter path
                    if next_depth < local_depths.get(next_blk, float('inf')):
                        local_depths[next_blk] = next_depth
                        queue.append((next_blk, next_depth))
            
            # Assign global depths and classify connections
            for blk in scc:
                blk_local_depth = local_depths.get(blk, 0)
                global_depth = current_depth + blk_local_depth
                self._blocks_loop_dag[global_depth].append(blk)
                
                # Process connections (already sorted in map)
                for con in self._outg_blk_con_map[blk]:
                    is_loop_closing = False
                    
                    # Check all targets
                    for target in con.targets:
                        target_blk = target.block
                        if target_blk in scc_set:
                            target_local_depth = local_depths.get(target_blk, 0)

                            # Back edge if target depth <= source depth
                            if target_local_depth <= blk_local_depth:
                                self._loop_closing_connections.append(con)
                                is_loop_closing = True
                                break
                    
                    if not is_loop_closing:
                        self._connections_loop_dag[global_depth].append(con)
            
            current_depth += max_local_depth + 1
        
        self._loop_depth = (max(self._blocks_loop_dag) + 1) if self._blocks_loop_dag else 0


    def _find_strongly_connected_components(self, blocks):
        """Iterative Tarjan's algorithm using cleaner state machine.

        Finds strongly connected components (cycles) within the given blocks using
        an iterative implementation of Tarjan's algorithm. Avoids recursion limits
        that can occur with deep graphs.

        Parameters
        ----------
        blocks : list
            list of blocks to analyze for SCCs

        Returns
        -------
        list
            list of SCCs, where each SCC is a list of blocks forming a cycle
        """
        if not blocks:
            return []
        
        block_set = set(blocks)
        index_counter = [0]
        index = {}
        lowlink = {}
        onstack = set()
        scc_stack = []
        result = []
        
        # Pre-filter successors
        successors_cache = defaultdict(list)
        for blk in blocks:
            succ = self._dnst_blk_blk_map[blk]
            successors_cache[blk] = [n for n in succ if n in block_set]
        
        for start_node in blocks:
            if start_node in index:
                continue
            
            # Work stack: each entry is (node, successor_index)
            # successor_index = -1 means node not yet initialized
            work_stack = [(start_node, -1)]
            
            while work_stack:
                node, succ_idx = work_stack[-1]
                
                # Initialize node on first visit
                if succ_idx == -1:
                    idx = index_counter[0]
                    index[node] = idx
                    lowlink[node] = idx
                    index_counter[0] += 1
                    
                    scc_stack.append(node)
                    onstack.add(node)
                    
                    # Update to start processing successors
                    work_stack[-1] = (node, 0)
                    continue
                
                # Get successors for this node
                successors = successors_cache[node]
                
                # Check if we've processed all successors
                if succ_idx >= len(successors):
                    # All successors processed - finalize this node
                    work_stack.pop()
                    
                    # Check if this is an SCC root
                    if lowlink[node] == index[node]:
                        # Extract SCC
                        scc = []
                        while True:
                            w = scc_stack.pop()
                            onstack.remove(w)
                            scc.append(w)
                            if w == node:
                                break
                        
                        # Keep only actual cycles
                        if len(scc) > 1:
                            result.append(scc)
                        elif scc[0] in successors_cache[scc[0]]:
                            result.append(scc)
                    
                    # Update parent's lowlink if there is a parent
                    if work_stack:
                        parent, parent_succ_idx = work_stack[-1]
                        if lowlink[node] < lowlink[parent]:
                            lowlink[parent] = lowlink[node]
                    
                    continue
                
                # Process current successor
                succ = successors[succ_idx]
                
                # Move to next successor for next iteration
                work_stack[-1] = (node, succ_idx + 1)
                
                if succ not in index:
                    # Unvisited successor - recurse
                    work_stack.append((succ, -1))
                elif succ in onstack:
                    # Back edge - update lowlink
                    if index[succ] < lowlink[node]:
                        lowlink[node] = index[succ]
        
        return result


    def is_algebraic_path(self, start_block, end_block):
        """Check if blocks are connected through an algebraic path.

        Determines whether there exists a path from start_block to end_block that
        only passes through algebraic blocks (blocks with non-zero length). Uses
        iterative DFS with early termination for efficiency.

        Parameters
        ----------
        start_block : Block
            starting block of the path
        end_block : Block
            ending block of the path

        Returns
        -------
        bool
            True if an algebraic path exists, False otherwise
        """
        # Quick checks
        if start_block is end_block:
            # Self-loop case: need to find path that leaves and returns
            return self._has_algebraic_self_loop(start_block)
        
        # Check if start has any outgoing connections
        if start_block not in self._dnst_blk_blk_map:
            return False
        
        # Check if end is algebraic (non-algebraic blocks can't be part of algebraic path)
        if end_block in self._dyn_blocks:
            return False
        
        # Iterative DFS with visited set
        visited = set()
        # Stack: just nodes (no need for iterators or depth)
        stack = [start_block]
        
        while stack:
            node = stack.pop()
            
            if node in visited:
                continue
            
            visited.add(node)
            
            # Get neighbors - use cached list if available
            neighbors = self._dnst_blk_blk_map[node]
            
            for nbr in neighbors:
                # Found the target!
                if nbr is end_block:
                    return True
                
                # Skip non-algebraic blocks
                if nbr in self._dyn_blocks:
                    continue
                
                # Skip already visited
                if nbr not in visited:
                    stack.append(nbr)
        
        return False


    def _has_algebraic_self_loop(self, block):
        """Check if a block has an algebraic path back to itself.

        The path leaves the block and returns through algebraic blocks. The
        block itself only bounds the path, so it doesn't have to be algebraic.
        This is the case for the 'Interface' of a 'Subsystem', where a path
        back to itself is an algebraic passthrough of the subsystem.

        Parameters
        ----------
        block : Block
            block to check for self-loop

        Returns
        -------
        bool
            True if an algebraic self-loop exists, False otherwise
        """
        # Get immediate neighbors
        neighbors = self._dnst_blk_blk_map[block]

        if not neighbors:
            return False

        # BFS from neighbors to see if any path back
        visited = set()
        stack = list(neighbors)
        
        while stack:
            node = stack.pop()
            
            if node in visited:
                continue
            
            # Found path back to start!
            if node is block:
                return True
            
            visited.add(node)
            
            # Skip non-algebraic
            if node in self._dyn_blocks:
                continue
            
            # Add neighbors
            for nbr in self._dnst_blk_blk_map[node]:
                if nbr not in visited:
                    stack.append(nbr)
        
        return False


    def outgoing_connections(self, block):
        """Returns outgoing connections of a block.

        Parameters
        ----------
        block : Block
            block to get outgoing connections for

        Returns
        -------
        list
            list of Connection objects originating from the block
        """
        return self._outg_blk_con_map[block]


    def dag(self):
        """Generator for DAG levels.

        Yields tuples of (depth, blocks, connections) for each level in the
        acyclic part of the graph, ordered from lowest to highest depth.

        Yields
        ------
        tuple
            (depth level, list of blocks at this depth, list of connections at this depth)
        """
        for d in range(self._alg_depth):
            yield (d, self._blocks_dag[d], self._connections_dag[d])


    def loop(self):
        """Generator for loop DAG levels.

        Yields tuples of (depth, blocks, connections) for each level in the
        algebraic loop part of the graph, ordered from lowest to highest depth.

        Yields
        ------
        tuple
            (depth level, list of blocks at this depth, list of connections at this depth)
        """
        for d in range(self._loop_depth):
            yield (d, self._blocks_loop_dag[d], self._connections_loop_dag[d])


    def loop_closing_connections(self):
        """Returns loop-closing connections.

        Loop-closing connections are back edges in the graph that create algebraic
        loops. These connections need special handling during simulation to resolve
        the implicit equations.

        Returns
        -------
        list
            list of Connection objects that close algebraic loops
        """
        return self._loop_closing_connections