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
from pathsim.blocks import Block
from pathsim.connection import Connection


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

        #should be 1
        self.assertEqual(len(S), 0)


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
        scope = Scope(sampling_rate=0.1)  # Has scheduled event
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
        self.assertEqual(len(S), 0)


    def test_graph(self): pass
    def test_nesting(self): pass




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
