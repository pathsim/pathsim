########################################################################################
##
##                                  TESTS FOR 
##                               'simulation.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.simulation import Simulation

#for testing
from pathsim.blocks._block import Block
from pathsim.connection import Connection

#modules from pathsim for test case
from pathsim.blocks import (
    Integrator,
    Amplifier,  
    Scope, 
    Adder
    )

from pathsim._constants import (
    SIM_TIMESTEP,
    SIM_TIMESTEP_MIN,
    SIM_TIMESTEP_MAX,
    SIM_TOLERANCE_FPI,
    SIM_ITERATIONS_MAX
    )

from pathsim.blocks import Source, Relay
from pathsim.events.schedule import Schedule
from pathsim.events._event import Event


# TESTS ================================================================================

class TestSimulation(unittest.TestCase):
    """
    Test the implementation of the 'Simulation' class

    only very minimal functonality
    """

    def test_init_default(self):

        #test default initialization
        Sim = Simulation(log=False)
        self.assertEqual(Sim.blocks, set())
        self.assertEqual(Sim.connections, set())
        self.assertEqual(Sim.events, set())
        self.assertEqual(Sim.dt, SIM_TIMESTEP)
        self.assertEqual(Sim.dt_min, SIM_TIMESTEP_MIN)
        self.assertEqual(Sim.dt_max, SIM_TIMESTEP_MAX)
        self.assertEqual(str(Sim.Solver()), "SSPRK22")
        self.assertEqual(Sim.tolerance_fpi, SIM_TOLERANCE_FPI)
        self.assertEqual(Sim.iterations_max, SIM_ITERATIONS_MAX)
        self.assertFalse(Sim.log)


    def test_init_sepecific(self):

        #test specific initialization
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3], 
            dt=0.02, 
            dt_min=0.001, 
            dt_max=0.1, 
            tolerance_fpi=1e-9, 
            tolerance_lte_rel=1e-4, 
            tolerance_lte_abs=1e-6, 
            iterations_max=100, 
            log=False
            )
        self.assertEqual(len(Sim.blocks), 3)
        self.assertEqual(len(Sim.connections), 3)
        self.assertEqual(Sim.dt, 0.02)
        self.assertEqual(Sim.dt_min, 0.001)
        self.assertEqual(Sim.dt_max, 0.1)
        self.assertEqual(Sim.tolerance_fpi, 1e-9)
        self.assertEqual(Sim.solver_kwargs, {"tolerance_lte_rel":1e-4, "tolerance_lte_abs":1e-6})
        self.assertEqual(Sim.iterations_max, 100)

        #test specific initialization with connection override
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B2) # <-- overrides B2
        with self.assertRaises(ValueError):
            Sim = Simulation(
                blocks=[B1, B2, B3], 
                connections=[C1, C2, C3],
                log=False
                )


    def test_contains(self):

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C3],
            log=False
            )

        self.assertTrue(B1 in Sim)
        self.assertTrue(B2 in Sim)
        self.assertTrue(B3 in Sim)

        self.assertTrue(C1 in Sim)
        self.assertTrue(C2 not in Sim)
        self.assertTrue(C3 in Sim)


    def test_add_block(self):
        
        Sim = Simulation(log=False)

        self.assertEqual(Sim.blocks, set())

        #test adding a block
        B1 = Block()
        Sim.add_block(B1)
        self.assertEqual(Sim.blocks, {B1})

        #test adding the same block again
        with self.assertRaises(ValueError):
            Sim.add_block(B1)


    def test_add_connection(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)

        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1],
            log=False
            )

        self.assertEqual(Sim.connections, {C1})

        #test adding a connection
        C2 = Connection(B2, B3)
        Sim.add_connection(C2)
        self.assertEqual(Sim.connections, {C1, C2})

        #test adding the same connection again
        with self.assertRaises(ValueError):
            Sim.add_connection(C2)
        self.assertEqual(Sim.connections, {C1, C2})


    def test_set_solver(self): 

        from pathsim.solvers import SSPRK22, RKCK54
        

        B1, B2 = Block(), Block()
        I1, I2 = Integrator(), Integrator()
        C1 = Connection(B1, B2)

        #check no solvers yet
        self.assertEqual(I1.engine, None)
        self.assertEqual(I2.engine, None)
        self.assertEqual(B1.engine, None)
        self.assertEqual(B2.engine, None)

        Sim = Simulation(
            blocks=[B1, B2, I1, I2], 
            connections=[C1],
            log=False
            )

        #check solvers initialized correctly
        self.assertTrue(isinstance(I1.engine, SSPRK22))
        self.assertTrue(isinstance(I2.engine, SSPRK22))
        self.assertEqual(B1.engine, None)
        self.assertEqual(B2.engine, None)

        Sim._set_solver(RKCK54)

        #check solvers are correctly updated
        self.assertTrue(isinstance(I1.engine, RKCK54))
        self.assertTrue(isinstance(I2.engine, RKCK54))
        self.assertEqual(B1.engine, None)
        self.assertEqual(B2.engine, None)


    def test_size(self):    

        #test 3 alg. blocks
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )  

        n, nx = Sim.size
        self.assertEqual(n, 3)
        self.assertEqual(nx, 0)

        #test 1 dyn, 1 alg block
        B1, B2 = Block(), Integrator()
        C1 = Connection(B1, B2)
        Sim = Simulation(
            blocks=[B1, B2], 
            connections=[C1],
            log=False
            )  

        n, nx = Sim.size
        self.assertEqual(n, 2)
        self.assertEqual(nx, 1)


    def test_update(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )   

        Sim._update(1)


    def test_step(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )   

        self.assertEqual(Sim.time, 0.0)

        #check stepping stats
        suc, err, scl, evl, its = Sim.step(0.1)

        self.assertTrue(suc)
        self.assertEqual(err, 0.0)
        self.assertEqual(scl, 1.0)
        self.assertEqual(evl, 1)
        self.assertEqual(its, 0)
        self.assertEqual(Sim.time, 0.1)

        Sim.reset()
        self.assertEqual(Sim.time, 0.0)
            
        #test time progression
        for i in range(1, 10):

            stats = Sim.step(0.1)
            self.assertAlmostEqual(Sim.time, 0.1*i, 10)


    def test_stop(self):
        """Test that stop() method sets _active flag to False"""
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1],
            log=False
        )
        
        # Initially active
        self.assertTrue(Sim._active)
        self.assertTrue(bool(Sim))  # __bool__ should return _active
        
        # Stop the simulation
        Sim.stop()
        
        # Should be inactive
        self.assertFalse(Sim._active)
        self.assertFalse(bool(Sim))


    def test_run(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )   

        self.assertEqual(Sim.time, 0.0)

        stats = Sim.run(10)

        #check stats
        self.assertEqual(stats["total_steps"], 1001)
            
        #check internal time progression (fixed step solvers naturally overshoot)
        self.assertAlmostEqual(Sim.time, 10, 1)





class TestSimulationIVP(unittest.TestCase):
    """
    special test case:
    linear feedback initial value problem with default solver (SSPRK22)
    """

    def setUp(self):

        #blocks that define the system
        self.Int = Integrator(1.0)
        self.Amp = Amplifier(-1)
        self.Add = Adder()
        self.Sco = Scope(labels=["response"])

        blocks = [self.Int, self.Amp, self.Add, self.Sco]

        #the connections between the blocks
        connections = [
            Connection(self.Amp, self.Add[1]),
            Connection(self.Add, self.Int),
            Connection(self.Int, self.Amp, self.Sco)
            ]

        #initialize simulation with the blocks, connections, timestep and logging enabled
        self.Sim = Simulation(blocks, connections, dt=0.02, log=False)


    def test_init(self):

        from pathsim.solvers import SSPRK22

        #test initialization of simulation
        self.assertEqual(len(self.Sim.blocks), 4)
        self.assertEqual(len(self.Sim.connections), 3)
        self.assertEqual(self.Sim.dt, 0.02)
        self.assertTrue(isinstance(self.Sim.engine, SSPRK22))
        self.assertTrue(self.Sim.engine.is_explicit)
        self.assertFalse(self.Sim.engine.is_adaptive)

        #test if engine setup was correct
        self.assertTrue(isinstance(self.Int.engine, SSPRK22)) # <-- only the Integrator needs an engine
        self.assertTrue(self.Amp.engine is None)
        self.assertTrue(self.Add.engine is None)
        self.assertTrue(self.Sco.engine is None)


    def test_set_solver(self):

        from pathsim.solvers import BDF2, SSPRK22

        #reset first
        self.Sim.reset()
    
        #set to implicit solver
        self.Sim._set_solver(BDF2)
        self.assertTrue(isinstance(self.Sim.engine, BDF2))

        #run with implicit solver
        self.test_run()

        #set to explicit solver
        self.Sim._set_solver(SSPRK22)
        self.assertTrue(isinstance(self.Sim.engine, SSPRK22))

        #run with explicit solver
        self.test_run()


    def test_step(self):

        #reset first
        self.Sim.reset()

        #check if reset was sucecssful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.outputs[0], self.Int.initial_value)

        #step using global timestep
        success, err, scl, te, ts = self.Sim.timestep()
        self.assertEqual(self.Sim.time, self.Sim.dt)
        self.assertEqual(err, 0.0) #fixed solver
        self.assertEqual(scl, 1.0) #fixed solver
        self.assertEqual(ts, 0) #no implicit solver
        
        #step again using custom timestep
        self.Sim.step(dt=2.2*self.Sim.dt)
        self.assertEqual(self.Sim.time, 3.2*self.Sim.dt)
        self.assertLess(self.Int.outputs[0], self.Int.initial_value)

        #test if scope recorded correctly
        time, data = self.Sco.read()
        for a, b in zip(time, [self.Sim.dt, 3.2*self.Sim.dt]):
            self.assertEqual(a, b)

        #reset again
        self.Sim.reset()

        #check if reset was successful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.outputs[0], self.Int.initial_value)


    def test_run(self):

        #reset first
        self.Sim.reset()

        #check if reset was successful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.outputs[0], self.Int.initial_value)

        #test running for some time
        self.Sim.run(duration=2, reset=True)
        self.assertAlmostEqual(self.Sim.time, 2, 5)
        
        time, data = self.Sco.read()
        _time = np.arange(0, 2.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 
        self.assertTrue(np.all(np.diff(data) < 0.0))

        #test running for some time with reset
        self.Sim.run(duration=1, reset=True)
        self.assertAlmostEqual(self.Sim.time, 1, 5)

        time, data = self.Sco.read()
        _time = np.arange(0, 1.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 

        #test running for some time without reset
        self.Sim.run(duration=2, reset=False)
        self.assertAlmostEqual(self.Sim.time, 3, 5)

        time, data = self.Sco.read()
        _time = np.arange(0, 3.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 






class TestSimulationEvents(unittest.TestCase):
    """Test event management and event handling during simulation"""

    def test_add_event(self):
        """Test adding events to simulation"""

        Sim = Simulation(log=False)

        evt = Event(func_evt=lambda t: t - 1.0, func_act=lambda t: None)
        Sim.add_event(evt)
        self.assertEqual(len(Sim.events), 1)
        self.assertIn(evt, Sim.events)

        #adding same event again raises ValueError
        with self.assertRaises(ValueError):
            Sim.add_event(evt)


    def test_contains_event(self):
        """Test __contains__ for events"""

        evt = Event(func_evt=lambda t: t - 1.0)
        evt2 = Event(func_evt=lambda t: t - 2.0)

        B1, B2 = Block(), Block()
        C1 = Connection(B1, B2)
        Sim = Simulation(
            blocks=[B1, B2],
            connections=[C1],
            events=[evt],
            log=False
            )

        self.assertIn(evt, Sim)
        self.assertNotIn(evt2, Sim)


    def test_run_with_schedule_event(self):
        """Test simulation with schedule event covering event system paths"""

        #simple system: integrator with source
        Src = Source(lambda t: 1.0)
        Int = Integrator(0.0)
        Sco = Scope(labels=["output"])

        #schedule event that fires periodically
        counter = [0]
        def count_action(t):
            counter[0] += 1

        evt = Schedule(t_start=0.0, t_period=0.5, func_act=count_action)

        Sim = Simulation(
            blocks=[Src, Int, Sco],
            connections=[
                Connection(Src, Int),
                Connection(Int, Sco)
                ],
            events=[evt],
            dt=0.01,
            log=False
            )

        Sim.run(duration=2.0, reset=True)

        #schedule should have fired multiple times
        self.assertGreater(counter[0], 0)
        self.assertAlmostEqual(Sim.time, 2.0, 2)


    def test_run_with_relay_block(self):
        """Test simulation with Relay block that has internal events"""

        #ramp source crosses relay thresholds
        Src = Source(lambda t: t - 1.0)  # crosses 0 at t=1
        Rly = Relay(threshold_up=0.5, threshold_down=-0.5, value_up=10.0, value_down=-10.0)
        Sco = Scope(labels=["relay"])

        Sim = Simulation(
            blocks=[Src, Rly, Sco],
            connections=[
                Connection(Src, Rly),
                Connection(Rly, Sco)
                ],
            dt=0.01,
            log=False
            )

        Sim.run(duration=3.0, reset=True)
        self.assertAlmostEqual(Sim.time, 3.0, 1)


    def test_reset_with_events(self):
        """Test that reset clears event state"""

        counter = [0]
        evt = Schedule(t_start=0, t_period=1.0, func_act=lambda t: None)

        B1 = Block()
        Sim = Simulation(blocks=[B1], events=[evt], log=False)

        Sim.run(duration=3.0)
        events_before = len(evt)

        #reset should clear event times
        Sim.reset()
        self.assertEqual(len(evt), 0)
        self.assertEqual(Sim.time, 0.0)


class TestSimulationAdvanced(unittest.TestCase):
    """Test advanced simulation features"""

    def setUp(self):
        """Set up a simple feedback system for reuse"""
        self.Int = Integrator(1.0)
        self.Amp = Amplifier(-1)
        self.Add = Adder()
        self.Sco = Scope(labels=["response"])

        blocks = [self.Int, self.Amp, self.Add, self.Sco]
        connections = [
            Connection(self.Amp, self.Add[1]),
            Connection(self.Add, self.Int),
            Connection(self.Int, self.Amp, self.Sco)
            ]

        self.Sim = Simulation(blocks, connections, dt=0.02, log=False)


    def test_linearize_delinearize(self):
        """Test linearize and delinearize methods"""

        self.Sim.reset()

        #linearize should not raise
        self.Sim.linearize()

        #run a few steps with linearized system
        self.Sim.timestep()
        self.Sim.timestep()

        #delinearize should not raise
        self.Sim.delinearize()

        #system should still work after delinearization
        self.Sim.timestep()


    def test_steadystate(self):
        """Test steady state solving"""

        self.Sim.reset()

        #for dx/dt = -x, steady state is x=0
        self.Sim.steadystate(reset=True)

        #state should be close to zero (steady state of dx/dt = -x)
        self.assertAlmostEqual(self.Int.outputs[0], 0.0, 4)


    def test_run_adaptive(self):
        """Test run with adaptive explicit solver"""

        from pathsim.solvers import RKCK54

        self.Sim._set_solver(RKCK54)
        self.Sim.reset()

        #run with adaptive stepping
        stats = self.Sim.run(duration=2.0, reset=True, adaptive=True)

        self.assertAlmostEqual(self.Sim.time, 2.0, 2)

        #solution should still decay
        time, data = self.Sco.read()
        self.assertTrue(np.all(np.diff(data) < 0.0))


    def test_run_adaptive_with_events(self):
        """Test adaptive solver with scheduled events to cover event estimation"""

        from pathsim.solvers import RKCK54

        self.Sim._set_solver(RKCK54)

        #add schedule event
        counter = [0]
        evt = Schedule(t_start=0.5, t_period=0.5, func_act=lambda t: counter[0].__add__(1) or None)
        self.Sim.add_event(evt)

        self.Sim.run(duration=2.0, reset=True, adaptive=True)
        self.assertAlmostEqual(self.Sim.time, 2.0, 2)


    def test_run_adaptive_implicit(self):
        """Test run with adaptive implicit solver"""

        from pathsim.solvers import ESDIRK32

        self.Sim._set_solver(ESDIRK32)
        self.Sim.reset()

        stats = self.Sim.run(duration=2.0, reset=True, adaptive=True)
        self.assertAlmostEqual(self.Sim.time, 2.0, 2)


    def test_run_streaming(self):
        """Test run_streaming generator"""

        self.Sim.reset()

        results = []
        for result in self.Sim.run_streaming(
            duration=1.0,
            reset=True,
            tickrate=100,
            func_callback=lambda: self.Sim.time
            ):
            results.append(result)

        #should have yielded at least one result
        self.assertGreater(len(results), 0)

        #last result should be close to end time
        self.assertAlmostEqual(self.Sim.time, 1.0, 2)


    def test_run_streaming_no_callback(self):
        """Test run_streaming without callback returns None"""

        self.Sim.reset()

        results = []
        for result in self.Sim.run_streaming(
            duration=0.5,
            reset=True,
            tickrate=100
            ):
            results.append(result)

        #without callback, results should be None
        for r in results:
            self.assertIsNone(r)


    def test_collect(self):
        """Test deprecated collect method"""

        self.Sim.run(duration=1.0, reset=True)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = self.Sim.collect()

        #should have scopes key with data
        self.assertIn("scopes", result)
        self.assertIn("spectra", result)
        self.assertGreater(len(result["scopes"]), 0)


    def test_stop_interrupts_run(self):
        """Test that stop() interrupts an active run"""

        #use schedule event to stop simulation at t=0.5
        def stop_action(t):
            self.Sim.stop()

        evt = Schedule(t_start=0.5, t_period=100, func_act=stop_action)
        self.Sim.add_event(evt)
        self.Sim.run(duration=5.0, reset=True)

        #simulation should have stopped early
        self.assertLess(self.Sim.time, 5.0)


    def test_deprecated_timestep_methods(self):
        """Test deprecated timestep methods still work"""

        self.Sim.reset()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            result = self.Sim.timestep_fixed_explicit(dt=0.01)
            self.assertEqual(len(result), 5)

            result = self.Sim.timestep_fixed_implicit(dt=0.01)
            self.assertEqual(len(result), 5)

            result = self.Sim.timestep_adaptive_explicit(dt=0.01)
            self.assertEqual(len(result), 5)

            result = self.Sim.timestep_adaptive_implicit(dt=0.01)
            self.assertEqual(len(result), 5)


    def test_run_realtime(self):
        """Test run_realtime generator"""

        self.Sim.reset()

        results = []
        for result in self.Sim.run_realtime(
            duration=0.2,
            reset=True,
            tickrate=50,
            speed=100.0,  #run 100x faster than real time
            func_callback=lambda: self.Sim.time
            ):
            results.append(result)

        #should have yielded results
        self.assertGreater(len(results), 0)
        self.assertAlmostEqual(self.Sim.time, 0.2, 1)


class TestSimulationRuntimeMutation(unittest.TestCase):
    """Test runtime mutation of simulation components (add/remove blocks,
    connections, events) and lazy graph rebuild via _graph_dirty flag."""

    def setUp(self):
        """Set up a simple system: Src -> Int -> Sco"""
        self.Src = Source(lambda t: 1.0)
        self.Int = Integrator(0.0)
        self.Sco = Scope(labels=["output"])

        self.C1 = Connection(self.Src, self.Int)
        self.C2 = Connection(self.Int, self.Sco)

        self.Sim = Simulation(
            blocks=[self.Src, self.Int, self.Sco],
            connections=[self.C1, self.C2],
            dt=0.01,
            log=False
        )

    def test_graph_dirty_after_add_block(self):
        """Adding a block after init marks graph dirty"""
        self.assertFalse(self.Sim._graph_dirty)

        B = Block()
        self.Sim.add_block(B)

        self.assertTrue(self.Sim._graph_dirty)
        self.assertIn(B, self.Sim.blocks)

    def test_graph_dirty_after_remove_block(self):
        """Removing a block marks graph dirty"""
        B = Block()
        self.Sim.add_block(B)

        # Clear dirty flag by stepping
        self.Sim.step(0.01)
        self.assertFalse(self.Sim._graph_dirty)

        self.Sim.remove_block(B)
        self.assertTrue(self.Sim._graph_dirty)
        self.assertNotIn(B, self.Sim.blocks)

    def test_graph_dirty_after_add_connection(self):
        """Adding a connection marks graph dirty"""
        B = Block()
        self.Sim.add_block(B)

        # Clear dirty flag
        self.Sim.step(0.01)
        self.assertFalse(self.Sim._graph_dirty)

        C = Connection(self.Sco, B)
        self.Sim.add_connection(C)
        self.assertTrue(self.Sim._graph_dirty)

    def test_graph_dirty_after_remove_connection(self):
        """Removing a connection marks graph dirty"""
        self.Sim.step(0.01)
        self.assertFalse(self.Sim._graph_dirty)

        self.Sim.remove_connection(self.C2)
        self.assertTrue(self.Sim._graph_dirty)
        self.assertNotIn(self.C2, self.Sim.connections)

    def test_lazy_rebuild_on_update(self):
        """Graph is rebuilt lazily when _update is called"""
        B = Block()
        self.Sim.add_block(B)
        self.assertTrue(self.Sim._graph_dirty)

        # step triggers _update which should rebuild graph
        self.Sim.step(0.01)
        self.assertFalse(self.Sim._graph_dirty)

    def test_remove_block_error(self):
        """Removing a block not in simulation raises ValueError"""
        B = Block()
        with self.assertRaises(ValueError):
            self.Sim.remove_block(B)

    def test_remove_connection_error(self):
        """Removing a connection not in simulation raises ValueError"""
        B1, B2 = Block(), Block()
        C = Connection(B1, B2)
        with self.assertRaises(ValueError):
            self.Sim.remove_connection(C)

    def test_remove_connection_zeroes_inputs(self):
        """Removing a connection zeroes target inputs on next graph rebuild"""
        # Run a step so the connection pushes data
        self.Src.function = lambda t: 5.0
        self.Sim.step(0.01)

        # Int should have received input from Src
        self.assertNotEqual(self.Int.inputs[0], 0.0)

        # Remove the connection — graph is dirty but not rebuilt yet
        self.Sim.remove_connection(self.C1)
        self.assertTrue(self.Sim._graph_dirty)

        # Next step triggers graph rebuild which resets inputs
        self.Sim.step(0.01)
        self.assertEqual(self.Int.inputs[0], 0.0)

    def test_remove_event(self):
        """Adding and removing events works"""
        evt = Event(func_evt=lambda t: t - 1.0, func_act=lambda t: None)
        self.Sim.add_event(evt)
        self.assertIn(evt, self.Sim.events)

        self.Sim.remove_event(evt)
        self.assertNotIn(evt, self.Sim.events)

    def test_remove_event_error(self):
        """Removing an event not in simulation raises ValueError"""
        evt = Event(func_evt=lambda t: t - 1.0, func_act=lambda t: None)
        with self.assertRaises(ValueError):
            self.Sim.remove_event(evt)

    def test_add_block_during_run(self):
        """Add a block mid-simulation and continue running"""
        # Run for a bit
        self.Sim.run(duration=0.5, reset=True)
        t_before = self.Sim.time

        # Add a new block
        Amp = Amplifier(2)
        self.Sim.add_block(Amp)
        self.assertTrue(self.Sim._graph_dirty)

        # Continue running - graph should rebuild and work
        self.Sim.run(duration=0.5, reset=False)
        self.assertGreater(self.Sim.time, t_before)

    def test_remove_block_during_run(self):
        """Remove a block mid-simulation and continue running"""
        # Run for a bit
        self.Sim.run(duration=0.5, reset=True)

        # Remove scope (leaf node, safe to remove)
        self.Sim.remove_connection(self.C2)
        self.Sim.remove_block(self.Sco)

        # Continue running
        self.Sim.run(duration=0.5, reset=False)
        self.assertAlmostEqual(self.Sim.time, 1.0, 1)

    def test_add_connection_during_run(self):
        """Add a connection mid-simulation and continue"""
        Sco2 = Scope(labels=["extra"])
        self.Sim.add_block(Sco2)

        self.Sim.run(duration=0.5, reset=True)

        C_new = Connection(self.Int, Sco2)
        self.Sim.add_connection(C_new)

        self.Sim.run(duration=0.5, reset=False)
        self.assertAlmostEqual(self.Sim.time, 1.0, 1)

        # Sco2 should have recorded data from the second half
        time, data = Sco2.read()
        self.assertGreater(len(time), 0)

    def test_remove_connection_during_run(self):
        """Remove a connection mid-simulation and continue"""
        self.Sim.run(duration=0.5, reset=True)

        self.Sim.remove_connection(self.C2)

        self.Sim.run(duration=0.5, reset=False)
        self.assertAlmostEqual(self.Sim.time, 1.0, 1)

    def test_multiple_mutations_single_rebuild(self):
        """Multiple mutations only trigger one rebuild"""
        B1, B2 = Block(), Block()
        C = Connection(B1, B2)

        self.Sim.add_block(B1)
        self.Sim.add_block(B2)
        self.Sim.add_connection(C)

        # Should still be dirty (not rebuilt yet)
        self.assertTrue(self.Sim._graph_dirty)

        # Single step triggers single rebuild
        self.Sim.step(0.01)
        self.assertFalse(self.Sim._graph_dirty)

    def test_dynamic_block_tracking(self):
        """Dynamically added integrators get solver and are tracked"""
        new_int = Integrator(0.0)
        self.Sim.add_block(new_int)

        # Should have been given a solver
        self.assertIsNotNone(new_int.engine)
        self.assertIn(new_int, self.Sim._blocks_dyn)

    def test_remove_dynamic_block_tracking(self):
        """Removing a dynamic block removes it from _blocks_dyn"""
        new_int = Integrator(0.0)
        self.Sim.add_block(new_int)
        self.assertIn(new_int, self.Sim._blocks_dyn)

        self.Sim.remove_block(new_int)
        self.assertNotIn(new_int, self.Sim._blocks_dyn)

    def test_streaming_with_mutations(self):
        """Add blocks between streaming generator steps using run_streaming"""
        # Use a longer duration + high tickrate to get many yields
        gen = self.Sim.run_streaming(
            duration=10.0, reset=True, tickrate=1000
        )

        # Consume a few ticks
        first_result = next(gen)

        # Mutate mid-stream: add a new scope
        Sco2 = Scope(labels=["live"])
        self.Sim.add_block(Sco2)
        C_new = Connection(self.Int, Sco2)
        self.Sim.add_connection(C_new)

        # Continue streaming — should rebuild graph and keep going
        results = list(gen)
        self.assertGreater(len(results), 0)

        # New scope should have data from after connection was added
        time_data, data = Sco2.read()
        self.assertGreater(len(time_data), 0)

    def test_parameter_mutation_during_run(self):
        """Change block parameters mid-simulation"""
        self.Sim.run(duration=0.5, reset=True)

        # Change amplifier gain-equivalent by modifying source
        # Integrator initial value is immutable at runtime, but we can
        # change the source function
        self.Src.function = lambda t: 2.0  # double the input

        self.Sim.run(duration=0.5, reset=False)
        self.assertAlmostEqual(self.Sim.time, 1.0, 1)

        # Integration result should reflect the change
        time, data = self.Sco.read()
        self.assertGreater(len(time), 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)