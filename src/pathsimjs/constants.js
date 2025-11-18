/**
 * GLOBAL CONSTANTS AND TOLERANCES FOR PATHSIMJS
 *
 * This module contains all the default constants and tolerances used throughout
 * the PathSimJS simulation framework.
 */

// Global floating point tolerance
export const TOLERANCE = 1e-16;

// Simulation default constants
export const SIM_TIMESTEP = 0.01;          // Transient simulation timestep (initial)
export const SIM_TIMESTEP_MIN = 1e-16;     // Min allowed transient timestep
export const SIM_TIMESTEP_MAX = null;      // Max allowed transient timestep
export const SIM_TOLERANCE_FPI = 1e-10;    // Tolerance for optimizer / algebraic loop solver
export const SIM_ITERATIONS_MAX = 200;     // Max number of optimizer / algebraic loop solver iterations

// Solver default constants
export const SOL_TOLERANCE_LTE_ABS = 1e-8;  // Absolute local truncation error (adaptive solvers)
export const SOL_TOLERANCE_LTE_REL = 1e-4;  // Relative local truncation error (adaptive solvers)
export const SOL_TOLERANCE_FPI = 1e-9;      // Tolerance for optimizer convergence (implicit solvers)
export const SOL_ITERATIONS_MAX = 200;      // Max number of optimizer iterations (for standalone implicit solvers)
export const SOL_SCALE_MIN = 0.1;           // Min allowed timestep rescale factor (adaptive solvers)
export const SOL_SCALE_MAX = 10;            // Max allowed timestep rescale factor (adaptive solvers)
export const SOL_BETA = 0.9;                // Safety for timestep control (adaptive solvers)

// Optimizer default constants
export const OPT_RESTART = false;  // Enable restart of anderson acceleration
export const OPT_HISTORY = 4;      // Max history length for anderson acceleration

// Event default constants
export const EVT_TOLERANCE = 1e-4;  // Tolerance for event detection (zero-crossing, condition)

// Logging default constants
export const LOG_ENABLE = true;        // Logging is enabled by default
export const LOG_MIN_INTERVAL = 1.0;   // Logging interval in seconds for progress, etc.
export const LOG_UPDATE_EVERY = 0.2;   // Logging update milestone every 0.2 -> every 20%

// Colors for visualization
export const COLOR_RED = "#e41a1c";
export const COLOR_BLUE = "#377eb8";
export const COLOR_GREEN = "#4daf4a";
export const COLOR_PURPLE = "#984ea3";
export const COLOR_ORANGE = "#ff7f00";
export const COLORS_ALL = [
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_ORANGE
];
