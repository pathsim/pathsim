##################################################################################
##
##                  GLOBAL CONSTANTS AND TOLERANCES FOR PATHSIM
##                               (_constants.py)
##
##################################################################################

# global floating point tolerance ------------------------------------------------

TOLERANCE = 1e-16


# simulation default constants ---------------------------------------------------

SIM_TIMESTEP = 0.01         # transient simulation timestep (initial) 
SIM_TIMESTEP_MIN = 1e-16    # min allowed transient timestep
SIM_TIMESTEP_MAX =  None    # max allowed transient timestep
SIM_TOLERANCE_FPI = 1e-10   # tolerance for optimizer / alg. loop solver
SIM_ITERATIONS_MAX = 200    # max number of optimizer / alg. loop solver iterations


# solver default constants -------------------------------------------------------

SOL_TOLERANCE_LTE_ABS = 1e-8   # absolute local truncation error (adaptive solvers)
SOL_TOLERANCE_LTE_REL = 1e-4   # relative local truncation error (adaptive solvers)
SOL_TOLERANCE_FPI = 1e-9       # tolerance for optimizer convergence (implicit solvers)
SOL_ITERATIONS_MAX = 200       # max number of optimizer iterations (for standalone implicit solvers)
SOL_SCALE_MIN = 0.1            # min allowed timestep rescale factor (adaptive solvers)
SOL_SCALE_MAX = 10             # max allowed timestep rescale factor (adaptive solvers)
SOL_BETA = 0.9                 # safety for timestep control (adaptive solvers)


# optimizer default constants ----------------------------------------------------

OPT_RESTART = False    # enable restart of anderson acceleration
OPT_HISTORY = 4        # max history length for anderson acceleration


# event default constants --------------------------------------------------------

EVT_TOLERANCE = 1e-4   # tolerance for event detection (zero-crossing, condition)


# logging default constants ------------------------------------------------------

LOG_ENABLE = True        # logging is enabled by default  
LOG_MIN_INTERVAL = 1.0   # logging interval in seconds for progress, etc.
LOG_UPDATE_EVERY = 0.2   # logging update milestone every 0.2 -> every 20%


# colors for visualization -------------------------------------------------------

COLOR_RED = "#e41a1c"
COLOR_BLUE = "#377eb8"
COLOR_GREEN = "#4daf4a"
COLOR_PURPLE = "#984ea3"
COLOR_ORANGE = "#ff7f00"
COLORS_ALL = [
	COLOR_RED, 
	COLOR_BLUE, 
	COLOR_GREEN, 
	COLOR_PURPLE, 
	COLOR_ORANGE
	]
