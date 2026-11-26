########################################################################################
##
##                                  TESTS FOR 
##                               'solvers/bdf.py'
##
##                            Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.bdf import *

from tests.pathsim.solvers._referenceproblems import PROBLEMS

import matplotlib.pyplot as plt


# TESTS ================================================================================

class TestBDF2(unittest.TestCase):
    """
    Test the implementation of the 'BDF2' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF2()

        self.assertEqual(solver.initial_value, 0)
        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        self.assertEqual(solver.n, 2)
        
        #test specific initialization
        solver = BDF2(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = BDF2()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(0)

            #test bdf buffer length
            buffer_length = len(solver.history)
            self.assertEqual(buffer_length, k+1 if k < solver.n else solver.n)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err, scale = solver.step(0.0, 1)


    def test_step(self):

        solver = BDF2()

        for i, t in enumerate(solver.stages(0, 1)):

            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertIsNone(scale)  # No rescale needed


    def test_integrate_fixed(self):
        
        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = BDF2(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        problem.jac,
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test convergence order, expected 2 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 1.5) # <- due to startup DIRK3

            #log stats
            stats[problem.name] = {"n":p, "err":errors, "dt":timesteps}

        # fig, ax = plt.subplots(dpi=120, tight_layout=True)
        # fig.suptitle(solver.__class__.__name__)
        # for name, stat in stats.items(): 
        #     ax.loglog(stat["dt"], stat["err"], label=name)
        # ax.loglog(timesteps, timesteps**solver.n, c="k", ls="--", label=f"n={solver.n}")
        # ax.legend()
        # plt.show()




class TestBDF3(unittest.TestCase):
    """
    Test the implementation of the 'BDF3' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF3()

        self.assertEqual(solver.initial_value, 0)
        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        self.assertEqual(solver.n, 3)
        
        #test specific initialization
        solver = BDF3(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = BDF3()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(0)

            #test bdf buffer length
            buffer_length = len(solver.history)
            self.assertEqual(buffer_length, k+1 if k < solver.n else solver.n)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err, scale = solver.step(0.0, 1)


    def test_step(self):

        solver = BDF3()

        for i, t in enumerate(solver.stages(0, 1)):

            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertIsNone(scale)  # No rescale needed


    def test_integrate_fixed(self):
        
        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = BDF3(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        problem.jac,
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test convergence order, expected 2 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 2.5) # <- due to startup DIRK3

            #log stats
            stats[problem.name] = {"n":p, "err":errors, "dt":timesteps}

        # fig, ax = plt.subplots(dpi=120, tight_layout=True)
        # fig.suptitle(solver.__class__.__name__)
        # for name, stat in stats.items(): 
        #     ax.loglog(stat["dt"], stat["err"], label=name)
        # ax.loglog(timesteps, timesteps**solver.n, c="k", ls="--", label=f"n={solver.n}")
        # ax.legend()
        # plt.show()



class TestBDF4(unittest.TestCase):
    """
    Test the implementation of the 'BDF4' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF4()

        self.assertEqual(solver.initial_value, 0)
        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        self.assertEqual(solver.n, 4)
        
        #test specific initialization
        solver = BDF4(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = BDF4()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(0)

            #test bdf buffer length
            buffer_length = len(solver.history)
            self.assertEqual(buffer_length, k+1 if k < solver.n else solver.n)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err, scale = solver.step(0.0, 1)


    def test_step(self):

        solver = BDF4()

        for i, t in enumerate(solver.stages(0, 1)):

            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertIsNone(scale)  # No rescale needed


    def test_integrate_fixed(self):
        
        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = BDF4(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        problem.jac,
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test convergence order, expected 3 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 3) # <- due to startup DIRK3

            #log stats
            stats[problem.name] = {"n":p, "err":errors, "dt":timesteps}

        # fig, ax = plt.subplots(dpi=120, tight_layout=True)
        # fig.suptitle(solver.__class__.__name__)
        # for name, stat in stats.items(): 
        #     ax.loglog(stat["dt"], stat["err"], label=name)
        # ax.loglog(timesteps, timesteps**solver.n, c="k", ls="--", label=f"n={solver.n}")
        # ax.legend()
        # plt.show()



class TestBDF5(unittest.TestCase):
    """
    Test the implementation of the 'BDF5' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF5()

        self.assertEqual(solver.initial_value, 0)
        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        self.assertEqual(solver.n, 5)
        
        #test specific initialization
        solver = BDF5(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = BDF5()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(0)

            #test bdf buffer length
            buffer_length = len(solver.history)
            self.assertEqual(buffer_length, k+1 if k < solver.n else solver.n)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err, scale = solver.step(0.0, 1)


    def test_step(self):

        solver = BDF5()

        for i, t in enumerate(solver.stages(0, 1)):

            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertIsNone(scale)  # No rescale needed


    def test_integrate_fixed(self):
        
        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = BDF5(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        problem.jac,
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test convergence order, expected 3 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 3) # <- due to startup DIRK3

            #log stats
            stats[problem.name] = {"n":p, "err":errors, "dt":timesteps}

        # fig, ax = plt.subplots(dpi=120, tight_layout=True)
        # fig.suptitle(solver.__class__.__name__)
        # for name, stat in stats.items(): 
        #     ax.loglog(stat["dt"], stat["err"], label=name)
        # ax.loglog(timesteps, timesteps**solver.n, c="k", ls="--", label=f"n={solver.n}")
        # ax.legend()
        # plt.show()


class TestBDF6(unittest.TestCase):
    """
    Test the implementation of the 'BDF6' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = BDF6()

        self.assertEqual(solver.initial_value, 0)
        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        self.assertEqual(solver.n, 6)
        
        #test specific initialization
        solver = BDF6(
            initial_value=1,  
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = BDF6()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(0)

            #test bdf buffer length
            buffer_length = len(solver.history)
            self.assertEqual(buffer_length, k+1 if k < solver.n else solver.n)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err, scale = solver.step(0.0, 1)


    def test_step(self):

        solver = BDF6()

        for i, t in enumerate(solver.stages(0, 1)):

            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertIsNone(scale)  # No rescale needed


    def test_integrate_fixed(self):
        
        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 2, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = BDF6(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        problem.jac,
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test convergence order, expected 3 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 3) # <- due to startup DIRK3

            #log stats
            stats[problem.name] = {"n":p, "err":errors, "dt":timesteps}

        # fig, ax = plt.subplots(dpi=120, tight_layout=True)
        # fig.suptitle(solver.__class__.__name__)
        # for name, stat in stats.items(): 
        #     ax.loglog(stat["dt"], stat["err"], label=name)
        # ax.loglog(timesteps, timesteps**solver.n, c="k", ls="--", label=f"n={solver.n}")
        # ax.legend()
        # plt.show()


class TestBDFTimestepChange(unittest.TestCase):
    """
    Test that the 'BDF' solvers restart their history when the timestep changes
    """

    def test_changed_timestep_restarts_history(self):

        #the equal-step coefficients integrate a constant exactly, but only
        #on a history with the same timestep as the current step
        func = lambda x, t: np.ones_like(x)
        jac = lambda x, t: np.zeros((1, 1))

        for Solver in [BDF2, BDF3, BDF4, BDF5, BDF6]:

            with self.subTest(Solver.__name__):

                solver = Solver()
                time = 0.0

                for dt in [0.1]*10 + [0.03] + [0.1]*10 + [0.05]*10:
                    solver.integrate_singlestep(func, jac, time, dt)
                    time += dt

                self.assertAlmostEqual(solver.get()[0], time, 10)


    def test_reset_forgets_timestep(self):

        solver = BDF2()
        solver.buffer(0.1)
        solver.reset()
        self.assertIsNone(solver._history_dt)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':

    unittest.main(verbosity=2)