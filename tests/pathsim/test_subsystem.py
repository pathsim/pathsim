########################################################################################
##
##                                  TESTS FOR 
##                                'subsystem.py'
##
##                               Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.subsystem import Subsystem, Interface

#for testing
from pathsim.blocks import Block, Integrator, Amplifier, Constant, Scope
from pathsim.blocks.dynsys import DynamicalSystem
from pathsim.connection import Connection
from pathsim.simulation import Simulation
from pathsim.solvers import SSPRK22


# TESTS ================================================================================

class TestInterface(unittest.TestCase):
    """
    Test the implementation of the 'Interface' class
    

    'Interface' is just a container that inherits everything from 'Block'
    """

    def test_len(self):
        I = Interface()
        self.assertEqual(len(I), 0)



class TestSubsystem(unittest.TestCase):
    """
    test implementation of the 'Subsystem' class
    """

    def test_dynamical_system_without_solver(self):
        """Test nesting a custom dynamical system before solver assignment"""
        class Mini(DynamicalSystem):
            def __init__(self):
                super().__init__(
                    func_dyn=lambda x, u, t: -x + u,
                    func_alg=lambda x, u, t: x,
                    initial_value=np.zeros(1),
                )
                self.inputs.resize(1)

        class Nested(Subsystem):
            def __init__(self):
                block, interface = Mini(), Interface()
                interface.register_port_map(
                    port_map_in={"y": 0}, port_map_out={"u": 0}
                )
                super().__init__(
                    [interface, block],
                    [
                        Connection(interface["u"], block),
                        Connection(block, interface["y"]),
                    ],
                )

        self.assertIsInstance(Nested(), Subsystem)


    def test_init(self):

        #test default initialization
        with self.assertRaises(ValueError):
            S = Subsystem()

        #test initialization without interface
        with self.assertRaises(ValueError):
            S = Subsystem(blocks=[Block(), Block()])

        #test specific initialization with interface
        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, I1)
        S = Subsystem(blocks=[B1, B2, B3, I1], connections=[C1, C2])
        self.assertEqual(len(S.blocks), 3)
        self.assertEqual(len(S.connections), 2)

        #test with too many interfaces
        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        I2 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, I1)
        with self.assertRaises(ValueError):
            S = Subsystem(blocks=[B1, B2, B3, I1, I2], connections=[C1, C2])


    def test_check_connections(self):

        #test specific initialization with connecion override
        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, I1)
        C3 = Connection(B2, B3) # <-- this one overrides B3
        with self.assertRaises(ValueError):
            S = Subsystem(blocks=[B1, B2, B3, I1], connections=[C1, C2, C3])


    def test_inputs_property(self): 

        B1 = Block()
        I1 = Interface()
        C1 = Connection(I1, B1, I1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        self.assertEqual(S.interface.outputs[0], 0.0)
        self.assertEqual(len(S.interface.outputs), 1)
        
        S.inputs[0] = 1.1
        S.inputs[1] = 2.2
        S.inputs[2] = 3.3

        self.assertEqual(S.interface.outputs[0], 1.1)
        self.assertEqual(S.interface.outputs[1], 2.2)
        self.assertEqual(S.interface.outputs[2], 3.3)


    def test_outputs_property(self): 

        B1 = Block()
        I1 = Interface()
        C1 = Connection(I1, B1, I1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        S.interface.inputs[0] = 1.1
        S.interface.inputs[1] = 2.2
        S.interface.inputs[2] = 3.3

        self.assertEqual(S.outputs[0], 1.1)
        self.assertEqual(S.outputs[1], 2.2)
        self.assertEqual(S.outputs[2], 3.3)


    def test_update(self): 

        B1 = Block()
        I1 = Interface()
        C1 = Connection(I1, B1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        S.update(0)


    def test_contains(self):

        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, I1)
        S = Subsystem(
            blocks=[B1, B2, B3, I1], 
            connections=[C1]
            )

        self.assertTrue(B1 in S)
        self.assertTrue(B2 in S)
        self.assertTrue(B3 in S)

        self.assertTrue(C1 in S)
        self.assertFalse(C2 in S)


    def test_size(self):   

        #test 3 alg. blocks
        I1 = Interface()
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        S = Subsystem(
            blocks=[I1, B1, B2, B3], 
            connections=[C1, C2, C3]
            )  

        n, nx = S.size
        self.assertEqual(n, 3)
        self.assertEqual(nx, 0)

        #test 1 dyn, 1 alg block
        from pathsim.blocks import Integrator

        I1 = Interface()
        B1, B2 = Block(), Integrator(3)
        C1 = Connection(B1, B2)
        S = Subsystem(
            blocks=[I1, B1, B2], 
            connections=[C1]
            )  

        n, nx = S.size
        self.assertEqual(n, 2)
        self.assertEqual(nx, 0) # <- no internal engine yet

        from pathsim.solvers import EUF
        S.set_solver(EUF, None)

        n, nx = S.size
        self.assertEqual(nx, 1)


    def test_len(self): 

        I1 = Interface()
        B1 = Block()
        C1 = Connection(I1, B1)
        C2 = Connection(B1, I1)
        S = Subsystem(
            blocks=[I1, B1],
            connections=[C1, C2]
            )

        #algebraic passthrough from the subsystem inputs to its outputs
        self.assertEqual(len(S), 1)


    def test_call(self):

        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, I1)
        S = Subsystem(blocks=[B1, B2, B3, I1], connections=[C1, C2])

        #inputs, outputs, states
        i, o, s = S()

        #siso stateless
        self.assertEqual(i, 0)
        self.assertEqual(o, 0)
        self.assertEqual(len(s), 0)


    def test_on_off(self):

        I1 = Interface()
        B1 = Block()
        C1 = Connection(I1, B1)
        C2 = Connection(B1, I1)
        S = Subsystem(
            blocks=[I1, B1], 
            connections=[C1, C2]
            ) 

        #default on
        self.assertTrue(S._active)
        self.assertTrue(B1._active)

        S.off()

        self.assertFalse(S._active)
        self.assertFalse(B1._active)

        S.on()

        self.assertTrue(S._active)
        self.assertTrue(B1._active)


    def test_call_with_dynamic_blocks(self):
        """Test __call__ method with blocks that have internal states"""
        from pathsim.blocks import Integrator
        from pathsim.solvers import EUF

        I1 = Interface()
        B1 = Integrator([1.0, 2.0])  # integrator with 2 states
        B2 = Block()
        C1 = Connection(I1, B1)
        C2 = Connection(B1, I1, B2)
        S = Subsystem(blocks=[I1, B1, B2], connections=[C1, C2])

        # Set solver to enable states
        S.set_solver(EUF, None)

        # Call should return inputs, outputs, and states
        i, o, s = S()

        # Should have states from integrator
        self.assertTrue(len(s) > 0)


    def test_algebraic_loop_with_boosters(self):
        """Test algebraic loop solving with boosters"""
        from pathsim.blocks import Amplifier

        I1 = Interface()
        B1, B2, B3 = Amplifier(gain=1.0), Amplifier(gain=1.0), Amplifier(gain=1.0)

        # Create algebraic loop: I1 -> B1 -> B2 -> B3 -> B1 (loop)
        # Also B1 -> I1 for output
        C1 = Connection(I1, B1)
        C2 = Connection(B1, B2)
        C3 = Connection(B2, B3)
        C4 = Connection(B3, B1[1])  # This closes the loop (to port 1 of B1)
        C5 = Connection(B1, I1)

        S = Subsystem(
            blocks=[I1, B1, B2, B3],
            connections=[C1, C2, C3, C4, C5]
        )

        # Should have boosters for loop closing connections
        self.assertIsNotNone(S.boosters)
        self.assertTrue(len(S.boosters) > 0)
        self.assertTrue(S.graph.has_loops)

        # Should be able to update without error
        S.update(0.0)


    def test_plot_method(self):
        """Test plot method calls plot on internal blocks"""
        from pathsim.blocks import Scope

        I1 = Interface()
        scope = Scope()
        C1 = Connection(I1, scope)
        S = Subsystem(blocks=[I1, scope], connections=[C1])

        # Should not raise error (even though Scope.plot might return None)
        S.plot()


    def test_linearize_delinearize(self):
        """Test linearize and delinearize methods"""
        from pathsim.blocks import Integrator, Amplifier

        I1 = Interface()
        B1 = Amplifier(gain=2.0)
        B2 = Integrator(1.0)
        C1 = Connection(I1, B1, B2)
        C2 = Connection(B1, I1)
        S = Subsystem(blocks=[I1, B1, B2], connections=[C1, C2])

        # Should be able to linearize and delinearize
        S.linearize(0.0)
        S.delinearize()


    def test_events_property(self):
        """Test that events are collected from internal blocks"""
        from pathsim.events import Schedule
        from pathsim.blocks import Scope

        I1 = Interface()
        scope = Scope(sampling_period=0.1)  # Has scheduled event
        C1 = Connection(I1, scope)
        S = Subsystem(blocks=[I1, scope], connections=[C1])

        # Should collect events from scope
        events = S.events
        self.assertTrue(len(events) > 0)


    def test_sample_method(self):
        """Test sample method on internal blocks"""
        from pathsim.blocks import Scope

        I1 = Interface()
        scope = Scope()
        C1 = Connection(I1, scope)
        S = Subsystem(blocks=[I1, scope], connections=[C1])

        # Should not raise error
        S.sample(1.0, 0.1)


    def test_reset_method(self):
        """Test reset method on subsystem and internal blocks"""
        I1 = Interface()
        B1 = Block()
        C1 = Connection(I1, B1, I1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        # Modify state
        S.inputs[0] = 5.0

        # Reset
        S.reset()

        # Should be reset
        self.assertEqual(S.inputs[0], 0.0)


    def test_solve_step_revert_buffer(self):
        """Test that solve, step, revert, and buffer methods exist and are callable"""
        from pathsim.blocks import Integrator
        from pathsim.solvers import RKDP54

        I1 = Interface()
        B1 = Integrator(1.0)
        C1 = Connection(I1, B1, I1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        # Set solver - creates _blocks_dyn list
        S.set_solver(RKDP54, None)

        # Test that _blocks_dyn was created
        self.assertIsNotNone(S._blocks_dyn)
        self.assertEqual(len(S._blocks_dyn), 1)  # One integrator

        # Test buffer method (should not raise)
        S.buffer(0.01)

        # Test revert method (should not raise)
        S.revert()

        # Solver methods would need proper simulation initialization to test fully


    def test_len_with_algebraic_passthrough(self):
        """Test __len__ correctly identifies algebraic passthrough"""
        I1 = Interface()
        B1 = Block()

        # Direct passthrough from interface to itself through B1
        C1 = Connection(I1, B1)
        C2 = Connection(B1, I1)
        S = Subsystem(blocks=[I1, B1], connections=[C1, C2])

        # Interface has algebraic path to itself
        self.assertEqual(len(S), 1)


    def test_len_dynamic_interior(self):
        """A dynamic block on the only path breaks the passthrough"""
        I1 = Interface()
        B1 = Block()
        I2 = Integrator()

        S = Subsystem(
            blocks=[I1, B1, I2],
            connections=[Connection(I1, B1), Connection(B1, I2), Connection(I2, I1)]
            )

        self.assertEqual(len(S), 0)


    def test_len_direct_passthrough(self):
        """A direct connection from the interface to itself is a passthrough"""
        I1 = Interface()
        S = Subsystem(blocks=[I1], connections=[Connection(I1, I1)])

        self.assertEqual(len(S), 1)


    def test_len_no_return_path(self):
        """Inputs that never reach the outputs are no passthrough"""
        I1 = Interface()
        B1 = Block()
        S = Subsystem(blocks=[I1, B1], connections=[Connection(I1, B1)])

        self.assertEqual(len(S), 0)


    def test_passthrough_series_without_delay(self):
        """Purely algebraic subsystems in series introduce no delay,
        see pathsim issue #251"""

        def make_gain(g):
            I1 = Interface()
            A1 = Amplifier(g)
            return Subsystem(
                blocks=[I1, A1],
                connections=[Connection(I1, A1), Connection(A1, I1)]
                )

        C1 = Constant(1.0)
        G1 = make_gain(2.0)
        G2 = make_gain(5.0)
        Sc = Scope()

        Sim = Simulation(
            blocks=[C1, G1, G2, Sc],
            connections=[Connection(C1, G1), Connection(G1, G2), Connection(G2, Sc)],
            Solver=SSPRK22,
            dt=0.01,
            log=False
            )
        Sim.run(0.05)

        _, [y] = Sc.read()
        np.testing.assert_array_almost_equal(y, np.full_like(y, 10.0))


    def test_graph(self): pass
    def test_nesting(self): pass


class TestSubsystemRuntimeMutation(unittest.TestCase):
    """Test runtime mutation of subsystem components (add/remove blocks,
    connections, events) and lazy graph rebuild via _graph_dirty flag."""

    def setUp(self):
        """Set up a subsystem: I1 -> B1 -> B2 -> I1"""
        self.I1 = Interface()
        self.B1 = Block()
        self.B2 = Block()
        self.C1 = Connection(self.I1, self.B1)
        self.C2 = Connection(self.B1, self.B2)
        self.C3 = Connection(self.B2, self.I1)

        self.S = Subsystem(
            blocks=[self.I1, self.B1, self.B2],
            connections=[self.C1, self.C2, self.C3]
        )

    def test_graph_dirty_after_add_block(self):
        """Adding a block marks graph dirty"""
        self.assertFalse(self.S._graph_dirty)

        B = Block()
        self.S.add_block(B)

        self.assertTrue(self.S._graph_dirty)
        self.assertIn(B, self.S.blocks)

    def test_graph_dirty_after_remove_block(self):
        """Removing a block marks graph dirty"""
        # Clear dirty flag by calling update
        self.S.update(0.0)
        self.assertFalse(self.S._graph_dirty)

        self.S.remove_block(self.B2)
        self.assertTrue(self.S._graph_dirty)
        self.assertNotIn(self.B2, self.S.blocks)

    def test_graph_dirty_after_add_connection(self):
        """Adding a connection marks graph dirty"""
        B3 = Block()
        self.S.add_block(B3)
        self.S.update(0.0)
        self.assertFalse(self.S._graph_dirty)

        C = Connection(self.B2, B3)
        self.S.add_connection(C)
        self.assertTrue(self.S._graph_dirty)

    def test_graph_dirty_after_remove_connection(self):
        """Removing a connection marks graph dirty"""
        self.S.update(0.0)
        self.assertFalse(self.S._graph_dirty)

        self.S.remove_connection(self.C3)
        self.assertTrue(self.S._graph_dirty)
        self.assertNotIn(self.C3, self.S.connections)

    def test_lazy_rebuild_on_update(self):
        """Graph is rebuilt lazily when update is called"""
        B = Block()
        self.S.add_block(B)
        self.assertTrue(self.S._graph_dirty)

        self.S.update(0.0)
        self.assertFalse(self.S._graph_dirty)

    def test_remove_block_error(self):
        """Removing a block not in subsystem raises ValueError"""
        B = Block()
        with self.assertRaises(ValueError):
            self.S.remove_block(B)

    def test_remove_connection_error(self):
        """Removing a connection not in subsystem raises ValueError"""
        B1, B2 = Block(), Block()
        C = Connection(B1, B2)
        with self.assertRaises(ValueError):
            self.S.remove_connection(C)

    def test_add_remove_event(self):
        """Adding and removing events works"""
        from pathsim.events._event import Event

        evt = Event(func_evt=lambda t: t - 1.0, func_act=lambda t: None)
        self.S.add_event(evt)
        self.assertIn(evt, self.S._events)

        self.S.remove_event(evt)
        self.assertNotIn(evt, self.S._events)

    def test_add_event_duplicate_error(self):
        """Adding duplicate event raises ValueError"""
        from pathsim.events._event import Event

        evt = Event(func_evt=lambda t: t - 1.0)
        self.S.add_event(evt)

        with self.assertRaises(ValueError):
            self.S.add_event(evt)

    def test_remove_event_error(self):
        """Removing event not in subsystem raises ValueError"""
        from pathsim.events._event import Event

        evt = Event(func_evt=lambda t: t - 1.0)
        with self.assertRaises(ValueError):
            self.S.remove_event(evt)

    def test_multiple_mutations_single_rebuild(self):
        """Multiple mutations only trigger one rebuild"""
        B3, B4 = Block(), Block()
        C = Connection(B3, B4)

        self.S.add_block(B3)
        self.S.add_block(B4)
        self.S.add_connection(C)

        # Still dirty — no rebuild yet
        self.assertTrue(self.S._graph_dirty)

        # Single update clears it
        self.S.update(0.0)
        self.assertFalse(self.S._graph_dirty)

    def test_dynamic_block_with_solver(self):
        """Dynamically added blocks get solver if subsystem has one"""
        from pathsim.blocks import Integrator
        from pathsim.solvers import EUF

        self.S.set_solver(EUF, None)

        new_int = Integrator(0.0)
        self.S.add_block(new_int)

        # Should have been given a solver
        self.assertIsNotNone(new_int.engine)
        self.assertIn(new_int, self.S._blocks_dyn)

    def test_remove_dynamic_block_tracking(self):
        """Removing a dynamic block removes it from _blocks_dyn"""
        from pathsim.blocks import Integrator
        from pathsim.solvers import EUF

        self.S.set_solver(EUF, None)

        new_int = Integrator(0.0)
        self.S.add_block(new_int)
        self.assertIn(new_int, self.S._blocks_dyn)

        self.S.remove_block(new_int)
        self.assertNotIn(new_int, self.S._blocks_dyn)

    def test_mutation_in_simulation_context(self):
        """Subsystem with mutations works inside a Simulation"""
        from pathsim.simulation import Simulation
        from pathsim.blocks import Source, Scope

        Src = Source(lambda t: 1.0)
        Sco = Scope()

        C_in = Connection(Src, self.S)
        C_out = Connection(self.S, Sco)

        Sim = Simulation(
            blocks=[Src, self.S, Sco],
            connections=[C_in, C_out],
            dt=0.01,
            log=False
        )

        # Run a bit
        Sim.run(duration=0.2, reset=True)

        # Add a block to the subsystem
        B_new = Block()
        self.S.add_block(B_new)
        self.assertTrue(self.S._graph_dirty)

        # Continue running — subsystem graph rebuilds internally
        Sim.run(duration=0.2, reset=False)
        self.assertAlmostEqual(Sim.time, 0.4, 1)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
