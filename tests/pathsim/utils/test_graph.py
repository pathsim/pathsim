########################################################################################
##
##                                   TESTS FOR 
##                                'utils/graph.py'
##
##                               Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.graph import Graph
from pathsim.blocks import Block, Integrator, Amplifier, Adder
from pathsim.connection import Connection


# TESTS ================================================================================

class TestGraph(unittest.TestCase):
    """tests for the Graph class in pathsim.utils.graph"""

    def setUp(self):
        """Set up common resources for test methods."""
        
        # Algebraic Blocks (assume len=1)
        self.amp1 = Amplifier(gain=2) # len=1
        self.amp2 = Amplifier(gain=-1) # len=1
        self.add1 = Adder() # len=1
        self.add2 = Adder("+-") # Subtractor, len=1

        # Dynamic Blocks (assume len=0)
        self.int1 = Integrator(initial_value=0.0) # len=0
        self.int2 = Integrator(initial_value=1.0) # len=0

        # Longest algebraic path: amp1(1) -> add1(1) -> amp2(1) = length 3, ends at depth 3
        # Graph depth = max_node_depth + 1 = 3 + 1 = 4
        self.c_amp1_add1 = Connection(self.amp1, self.add1[0])
        self.c_int1_add1 = Connection(self.int1, self.add1[1])
        self.c_add1_amp2 = Connection(self.add1, self.amp2)
        self.c_amp2_int2 = Connection(self.amp2, self.int2)

        self.nodes_acyclic = [self.amp1, self.add1, self.int1, self.amp2, self.int2]
        self.conns_acyclic = [self.c_amp1_add1, self.c_int1_add1, self.c_add1_amp2, self.c_amp2_int2]

        # Loop: add2 -> amp1 -> add2
        self.c_add2_amp1 = Connection(self.add2, self.amp1)
        self.c_amp1_add2 = Connection(self.amp1, self.add2[1])

        self.nodes_alg_loop = [self.add2, self.amp1]
        self.conns_alg_loop = [self.c_add2_amp1, self.c_amp1_add2]

        # Loop: int1 -> int2 -> int1
        self.c_int1_int2 = Connection(self.int1, self.int2)
        self.c_int2_int1 = Connection(self.int2, self.int1)

        self.nodes_dyn_loop = [self.int1, self.int2]
        self.conns_dyn_loop = [self.c_int1_int2, self.c_int2_int1]


    def test_init_empty(self):
        """Test initializing an empty graph."""

        g = Graph([], [])
        self.assertEqual(len(g), 0)
        self.assertFalse(g.has_loops)

        alg_depth, loop_depth = g.depth
        self.assertEqual(alg_depth, 0)
        self.assertEqual(loop_depth, 0)


    def test_init_with_mixed_nodes_edges_acyclic(self):
        """Test initializing a graph with mixed nodes and acyclic connections."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)
        self.assertEqual(len(g), 5)
        self.assertFalse(g.has_loops)


    def test_init_with_dynamic_loop(self):
        """Test initializing a graph with a dynamic loop."""

        g = Graph(self.nodes_dyn_loop, self.conns_dyn_loop)
        self.assertEqual(len(g), 2)
        self.assertFalse(g.has_loops)


    def test_init_with_algebraic_loop(self):
        """Test initializing a graph with an algebraic loop."""
        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)
        self.assertEqual(len(g), 2)
        self.assertTrue(g.has_loops)


    def test_has_loops_mixed_false(self):
        """Test loop detection in an acyclic graph with mixed blocks."""
        g = Graph(self.nodes_acyclic, self.conns_acyclic)
        self.assertFalse(g.has_loops)


    def test_has_loops_mixed_true_algebraic(self):
        """Test loop detection with an algebraic loop present."""

        # Add unconnected int1 to ensure it doesn't interfere
        nodes = self.nodes_alg_loop + [self.int1]
        g = Graph(nodes, self.conns_alg_loop)
        self.assertTrue(g.has_loops)


    def test_has_loops_mixed_true_via_integrator(self):
        """Test loop detection when loop path includes integrator (should be false)."""

        # Loop: amp1 -> add1 -> amp2 -> int2 -> amp1
        # Connection needed: Connection(self.amp1, self.add1[0]) -> already have c_amp1_add1
        # Connection needed: Connection(self.add1, self.amp2) -> already have c_add1_amp2
        # Connection needed: Connection(self.amp2, self.int2) -> already have c_amp2_int2
        c_int2_amp1_new = Connection(self.int2, self.amp1)
        nodes = [self.amp1, self.add1, self.amp2, self.int2]
        connections = [self.c_amp1_add1, self.c_add1_amp2, self.c_amp2_int2, c_int2_amp1_new]
        g = Graph(nodes, connections)

        # Algebraic loop detection should return False because int2 breaks the algebraic path
        self.assertFalse(g.has_loops)


    def test_depth_acyclic_mixed(self):
        """Test depth calculation in an acyclic graph with mixed blocks."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)
        alg_depth, loop_depth = g.depth

        # Max upstream algebraic path ends at amp2 or int2, tracing back through amp2(1)+add1(1)+amp1(1) = 3.
        # Graph._alg_depth is max_depth + 1.
        self.assertEqual(alg_depth, 3)
        self.assertEqual(loop_depth, 0)


    def test_depth_algebraic_loop(self):
        """Test depth calculation with an algebraic loop."""

        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)
        alg_depth, loop_depth = g.depth

        # Upstream path length is None for blocks in loop -> _blocks_dag empty -> alg_depth = 0
        # Loop BFS: add2 (entry depth 0) -> amp1 (depth 1). Max loop depth = 1. Graph._loop_depth = max + 1 = 2.
        self.assertEqual(alg_depth, 0)
        self.assertEqual(loop_depth, 2) 


    def test_depth_dynamic_loop(self):
        """Test depth calculation with only a dynamic loop."""

        g = Graph(self.nodes_dyn_loop, self.conns_dyn_loop)

        alg_depth, loop_depth = g.depth

        # Upstream path length for int1/int2 is 0 (stops at non-algebraic block). Max depth = 0.
        # Graph._alg_depth = max + 1 = 1.
        # No algebraic loop -> loop_depth = 0.
        self.assertEqual(alg_depth, 1)
        self.assertEqual(loop_depth, 0) 


    def test_size_property(self):
        """Test the size property returns correct values."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)
        num_blocks, num_connections = g.size

        self.assertEqual(num_blocks, 5)
        self.assertEqual(num_connections, 4)  # Total number of connection targets


    def test_size_property_empty(self):
        """Test the size property on empty graph."""

        g = Graph([], [])
        num_blocks, num_connections = g.size

        self.assertEqual(num_blocks, 0)
        self.assertEqual(num_connections, 0)


    def test_bool_operator(self):
        """Test that Graph always evaluates to True (even when empty)."""

        g_empty = Graph([], [])
        g_full = Graph(self.nodes_acyclic, self.conns_acyclic)

        self.assertTrue(bool(g_empty))
        self.assertTrue(bool(g_full))


    def test_outgoing_connections(self):
        """Test outgoing_connections method."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)

        # amp1 has one outgoing connection
        outgoing_amp1 = g.outgoing_connections(self.amp1)
        self.assertEqual(len(outgoing_amp1), 1)
        self.assertIn(self.c_amp1_add1, outgoing_amp1)

        # add1 has one outgoing connection
        outgoing_add1 = g.outgoing_connections(self.add1)
        self.assertEqual(len(outgoing_add1), 1)
        self.assertIn(self.c_add1_amp2, outgoing_add1)

        # int1 has one outgoing connection
        outgoing_int1 = g.outgoing_connections(self.int1)
        self.assertEqual(len(outgoing_int1), 1)
        self.assertIn(self.c_int1_add1, outgoing_int1)


    def test_is_algebraic_path_simple(self):
        """Test is_algebraic_path with simple acyclic connections."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)

        # Path exists: amp1 -> add1 -> amp2
        self.assertTrue(g.is_algebraic_path(self.amp1, self.amp2))

        # Path exists: amp1 -> add1
        self.assertTrue(g.is_algebraic_path(self.amp1, self.add1))

        # No path: amp2 -> amp1 (wrong direction)
        self.assertFalse(g.is_algebraic_path(self.amp2, self.amp1))

        # Path exists but starts from non-algebraic block
        # is_algebraic_path checks if there's a path through algebraic blocks
        # In this setup: int1 -> add1 -> amp2, the path from int1 does exist
        self.assertTrue(g.is_algebraic_path(self.int1, self.amp2))


    def test_is_algebraic_path_self_loop(self):
        """Test is_algebraic_path with self-loops."""

        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)

        # Algebraic loop: add2 -> amp1 -> add2
        # For self-loop detection, the method checks if path leaves and returns
        # Testing path between blocks in a loop
        self.assertTrue(g.is_algebraic_path(self.add2, self.amp1))
        self.assertTrue(g.is_algebraic_path(self.amp1, self.add2))

        # A block in the loop has an algebraic path back to itself
        self.assertTrue(g.is_algebraic_path(self.add2, self.add2))
        self.assertTrue(g.is_algebraic_path(self.amp1, self.amp1))


    def test_is_algebraic_path_self_loop_dynamic(self):
        """Test is_algebraic_path does not report loops through dynamic blocks."""

        g = Graph(self.nodes_dyn_loop, self.conns_dyn_loop)

        # Loop: int1 -> int2 -> int1, broken by the integrators
        self.assertFalse(g.is_algebraic_path(self.int1, self.int1))


    def test_is_algebraic_path_direct_self_connection(self):
        """Test is_algebraic_path with a direct self-connection."""

        amp = Amplifier(gain=2)
        g = Graph([amp], [Connection(amp, amp)])

        self.assertTrue(g.is_algebraic_path(amp, amp))


    def test_is_algebraic_path_self_loop_bounding_block(self):
        """Test is_algebraic_path where the block bounding the self-loop is 
        not algebraic, like the 'Interface' of a 'Subsystem'."""

        # Algebraic return path
        itg = Integrator()
        amp = Amplifier(gain=2)
        g = Graph([itg, amp], [Connection(itg, amp), Connection(amp, itg)])
        self.assertTrue(g.is_algebraic_path(itg, itg))

        # No return path
        itg = Integrator()
        amp = Amplifier(gain=2)
        g = Graph([itg, amp], [Connection(itg, amp)])
        self.assertFalse(g.is_algebraic_path(itg, itg))


    def test_is_algebraic_path_no_connection(self):
        """Test is_algebraic_path with unconnected blocks."""

        # Create unconnected blocks
        amp3 = Amplifier(gain=3)
        amp4 = Amplifier(gain=4)

        g = Graph([amp3, amp4], [])

        # No path between unconnected blocks
        self.assertFalse(g.is_algebraic_path(amp3, amp4))


    def test_loop_traversal(self):
        """Test loop traversal on a graph with algebraic loops."""

        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)

        loop_result = list(g.loop())

        # Should have 2 depth levels
        self.assertEqual(len(loop_result), 2)

        # Both blocks should be in the loop
        all_blocks = []
        for _, blocks, _ in loop_result:
            all_blocks.extend(blocks)

        self.assertIn(self.add2, all_blocks)
        self.assertIn(self.amp1, all_blocks)


    def test_loop_closing_connections(self):
        """Test loop_closing_connections method."""

        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)

        loop_closing = g.loop_closing_connections()

        # Should have at least one loop-closing connection
        self.assertGreater(len(loop_closing), 0)

        # At least one of the connections should be loop-closing
        self.assertTrue(
            self.c_add2_amp1 in loop_closing or self.c_amp1_add2 in loop_closing
        )


    def test_loop_closing_connections_acyclic(self):
        """Test loop_closing_connections on acyclic graph."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)

        loop_closing = g.loop_closing_connections()

        # Should have no loop-closing connections
        self.assertEqual(len(loop_closing), 0)


    def test_init_with_none_arguments(self):
        """Test initialization with None arguments (should use empty lists)."""

        g = Graph(None, None)

        self.assertEqual(len(g), 0)
        self.assertEqual(len(g.blocks), 0)
        self.assertEqual(len(g.connections), 0)


    def test_single_algebraic_block(self):
        """Test graph with single algebraic block."""

        amp = Amplifier(gain=5)
        g = Graph([amp], [])

        self.assertEqual(len(g), 1)
        self.assertFalse(g.has_loops)

        alg_depth, loop_depth = g.depth
        # Single block with len=1 gives depth of 1
        self.assertEqual(alg_depth, 1)
        self.assertEqual(loop_depth, 0)


    def test_single_dynamic_block(self):
        """Test graph with single dynamic block."""

        integ = Integrator(0.0)
        g = Graph([integ], [])

        self.assertEqual(len(g), 1)
        self.assertFalse(g.has_loops)

        alg_depth, loop_depth = g.depth
        self.assertEqual(alg_depth, 1)
        self.assertEqual(loop_depth, 0)


    def test_self_connection_algebraic(self):
        """Test algebraic block with direct self-connection."""

        amp = Amplifier(gain=0.5)
        c_self = Connection(amp, amp)

        g = Graph([amp], [c_self])

        # Self-connection creates algebraic loop
        self.assertTrue(g.has_loops)


    def test_complex_mixed_graph(self):
        """Test complex graph with both DAG and loop components."""

        # Create a complex structure:
        # amp1 -> add1 -> amp2 (DAG part)
        #         add1 -> add2 -> amp3 -> add2 (loop part)

        amp3 = Amplifier(gain=3)
        c_add1_add2 = Connection(self.add1, self.add2[0])
        c_add2_amp3 = Connection(self.add2, amp3)
        c_amp3_add2 = Connection(amp3, self.add2[1])

        nodes = [self.amp1, self.add1, self.amp2, self.add2, amp3]
        connections = [
            self.c_amp1_add1,
            self.c_add1_amp2,
            c_add1_add2,
            c_add2_amp3,
            c_amp3_add2
        ]

        g = Graph(nodes, connections)

        self.assertTrue(g.has_loops)
        self.assertEqual(len(g), 5)

        alg_depth, loop_depth = g.depth
        self.assertGreater(alg_depth, 0)
        self.assertGreater(loop_depth, 0)


    def test_linear_chain(self):
        """Test long linear chain of blocks."""

        # Create chain: amp1 -> amp2 -> amp3 -> amp4
        amp3 = Amplifier(gain=3)
        amp4 = Amplifier(gain=4)

        c1 = Connection(self.amp1, self.amp2)
        c2 = Connection(self.amp2, amp3)
        c3 = Connection(amp3, amp4)

        g = Graph([self.amp1, self.amp2, amp3, amp4], [c1, c2, c3])

        self.assertFalse(g.has_loops)
        self.assertEqual(len(g), 4)

        alg_depth, loop_depth = g.depth
        # Chain of 4 blocks each with len=1: depth = 4
        self.assertEqual(alg_depth, 4)
        self.assertEqual(loop_depth, 0)

        # Verify traversal
        dag_result = list(g.dag())
        self.assertEqual(len(dag_result), 4)


    def test_multiple_sccs(self):
        """Test graph with multiple strongly connected components."""

        # Create two separate loops:
        # Loop 1: add2 -> amp1 -> add2
        # Loop 2: amp2 -> amp3 -> amp2

        amp3 = Amplifier(gain=3)
        c_amp2_amp3 = Connection(self.amp2, amp3)
        c_amp3_amp2 = Connection(amp3, self.amp2)

        nodes = [self.add2, self.amp1, self.amp2, amp3]
        connections = [
            self.c_add2_amp1,
            self.c_amp1_add2,
            c_amp2_amp3,
            c_amp3_amp2
        ]

        g = Graph(nodes, connections)

        self.assertTrue(g.has_loops)
        self.assertEqual(len(g), 4)


    def test_dag_empty_when_only_loops(self):
        """Test that DAG is empty when graph only contains loops."""

        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)

        dag_result = list(g.dag())

        # DAG should be empty (alg_depth = 0)
        self.assertEqual(len(dag_result), 0)


    def test_loop_empty_when_acyclic(self):
        """Test that loop is empty when graph is acyclic."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)

        loop_result = list(g.loop())

        # Loop should be empty
        self.assertEqual(len(loop_result), 0)


    def test_parallel_branches(self):
        """Test graph with parallel branches."""

        # Create parallel structure:
        #        -> amp1 ->
        # int1                add1
        #        -> amp2 ->

        c_int1_amp1 = Connection(self.int1, self.amp1)
        c_int1_amp2 = Connection(self.int1, self.amp2)
        c_amp1_add1 = Connection(self.amp1, self.add1[0])
        c_amp2_add1 = Connection(self.amp2, self.add1[1])

        nodes = [self.int1, self.amp1, self.amp2, self.add1]
        connections = [c_int1_amp1, c_int1_amp2, c_amp1_add1, c_amp2_add1]

        g = Graph(nodes, connections)

        self.assertFalse(g.has_loops)
        self.assertEqual(len(g), 4)

        alg_depth, loop_depth = g.depth
        self.assertEqual(alg_depth, 3)  # int1(0) -> amp1/amp2(1) -> add1(2)
        self.assertEqual(loop_depth, 0)


    def test_disconnected_components(self):
        """Test graph with multiple disconnected components."""

        # Component 1: amp1 -> add1
        # Component 2: amp2 -> add2 (unconnected to component 1)

        c1 = Connection(self.amp1, self.add1)
        c2 = Connection(self.amp2, self.add2)

        g = Graph([self.amp1, self.add1, self.amp2, self.add2], [c1, c2])

        self.assertFalse(g.has_loops)
        self.assertEqual(len(g), 4)


    def test_is_algebraic_path_through_multiple_blocks(self):
        """Test is_algebraic_path through multiple intermediate blocks."""

        # Chain: amp1 -> add1 -> amp2 -> add2
        amp3 = Amplifier(gain=3)
        c1 = Connection(self.amp1, self.add1)
        c2 = Connection(self.add1, self.amp2)
        c3 = Connection(self.amp2, self.add2)
        c4 = Connection(self.add2, amp3)

        g = Graph([self.amp1, self.add1, self.amp2, self.add2, amp3], [c1, c2, c3, c4])

        # Path from amp1 to amp3
        self.assertTrue(g.is_algebraic_path(self.amp1, amp3))

        # Path from add1 to amp3
        self.assertTrue(g.is_algebraic_path(self.add1, amp3))

        # No reverse path
        self.assertFalse(g.is_algebraic_path(amp3, self.amp1))



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)