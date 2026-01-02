########################################################################################
##
##                        LOGGER MANAGER SINGLETON CLASS
##                              (utils/logger.py)
##
##              Centralized logging configuration for PathSim package.
##          Provides a singleton manager for consistent logging across modules.
##
########################################################################################

# IMPORTS ==============================================================================

import logging
import sys


# LOGGER MANAGER SINGLETON =============================================================

class LoggerManager:
    """Singleton class for centralized logging configuration in PathSim.

    Provides a unified interface for creating and configuring loggers throughout
    the PathSim package. All loggers follow a hierarchical naming scheme under
    the 'pathsim' root logger, allowing fine-grained control over logging levels
    and output destinations.

    The singleton pattern ensures that logging configuration is consistent across
    the entire application, with all modules sharing the same handler setup and
    formatting rules.

    Examples
    --------

    .. code-block:: python

        # Create and configure logging in one step
        from pathsim.utils.logger import LoggerManager

        mgr = LoggerManager(
            enabled=True,
            output="simulation.log",  # File path or None for stdout
            level=logging.INFO
        )

        # Get a logger for a specific module
        logger = mgr.get_logger("simulation")
        logger.info("Simulation started")

        # Set different log levels for different modules
        mgr.set_level(logging.DEBUG, "progress")
        mgr.set_level(logging.WARNING, "analysis")

        # Reconfigure later if needed
        mgr.configure(enabled=False)  # Disable logging


    Notes
    -----
    The LoggerManager uses a hierarchical logger structure:

    - pathsim (root)
      - pathsim.simulation
      - pathsim.progress
        - pathsim.progress.TRANSIENT
        - pathsim.progress.STEADYSTATE
      - pathsim.analysis
        - pathsim.analysis.timer
        - pathsim.analysis.profiler

    This hierarchy allows you to control logging at different granularities:
    set the level on 'pathsim' to affect all loggers, or set it on
    'pathsim.progress' to affect only progress tracking loggers.

    """

    _instance = None
    _initialized = False

    def __new__(cls, enabled=False, output=None, level=logging.INFO,
                format=None, date_format='%H:%M:%S'):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self, enabled=False, output=None, level=logging.INFO,
                 format=None, date_format='%H:%M:%S'):
        """Initialize the logger manager and setup root logger.

        Configuration is applied immediately on first instantiation if enabled=True.
        Subsequent instantiations return the existing singleton (parameters ignored).
        Use configure() to change settings after initialization.

        Parameters
        ----------
        enabled : bool, optional
            Whether logging is enabled. Defaults to False.
        output : str or None, optional
            Output destination. If string, logs to file. If None, logs to stdout.
            Defaults to None.
        level : int, optional
            Logging level. Defaults to logging.INFO.
        format : str or None, optional
            Log message format. Defaults to "%(asctime)s - %(levelname)s - %(message)s".
        date_format : str or None, optional
            Date format for timestamps. Defaults to '%H:%M:%S'.
        """
        if not LoggerManager._initialized:
            self._setup_root_logger()
            LoggerManager._initialized = True

            #apply configuration if enabled
            if enabled:
                self.configure(
                    enabled=True,
                    output=output,
                    level=level,
                    format=format,
                    date_format=date_format
                    )


    def _setup_root_logger(self):
        """Setup the root PathSim logger with default configuration.

        Creates the 'pathsim' root logger and initializes it with no handlers.
        Handlers are added via the configure() method. Also sets up Python
        warnings to be captured through the logging system.
        """

        #get the root pathsim logger
        self.root_logger = logging.getLogger("pathsim")

        #prevent propagation to root logger
        self.root_logger.propagate = False

        #capture Python warnings through logging
        logging.captureWarnings(True)

        #store configuration state
        self._enabled = False
        self._output = None
        self._level = logging.INFO
        self._format = "%(asctime)s - %(levelname)s - %(message)s"
        self._date_format = '%H:%M:%S'  #shorter timestamp format

        #store handler reference for reconfiguration
        self._current_handler = None


    def configure(self, enabled=True, output=None, level=logging.INFO, 
                  format=None, date_format=None):
        """Configure the root PathSim logger and all child loggers.

        This method sets up the logging system with the specified parameters.
        All loggers created via get_logger() will inherit this configuration.
        Can be called multiple times to reconfigure logging.

        Parameters
        ----------
        enabled : bool, optional
            Whether logging is enabled. If False, all logging is disabled.
            Defaults to True.
        output : str or None, optional
            Output destination for logs. If a string, interpreted as a file path
            and logs are written to that file. If None, logs are written to stdout.
            Defaults to None (stdout).
        level : int, optional
            Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
            Defaults to logging.INFO.
        format : str or None, optional
            Log message format string. If None, uses default format.
            Defaults to "%(asctime)s - %(levelname)s - %(message)s".
        date_format : str or None, optional
            Date format string for timestamps (e.g., '%H:%M:%S').
            If None, uses default format. Defaults to None.

        Examples
        --------

        .. code-block:: python

            mgr = LoggerManager()

            # Log to stdout with INFO level
            mgr.configure(enabled=True)

            # Log to file with DEBUG level
            mgr.configure(enabled=True, output="debug.log", level=logging.DEBUG)

            # Disable all logging
            mgr.configure(enabled=False)

            # Custom format with time only
            mgr.configure(
                enabled=True,
                format="%(asctime)s - %(message)s",
                date_format='%H:%M:%S'
            )

        """

        #store configuration
        self._enabled = enabled
        self._output = output
        self._level = level
        self._format = format or self._format
        self._date_format = date_format

        #remove existing handler if present
        if self._current_handler is not None:
            self.root_logger.removeHandler(self._current_handler)
            self._current_handler.close()
            self._current_handler = None

        #if logging is disabled, remove all handlers and return
        if not enabled:
            self.root_logger.handlers.clear()
            self.root_logger.setLevel(logging.CRITICAL + 1)  #effectively disable
            return

        #create appropriate handler
        if isinstance(output, str):
            #file handler for logging to file
            handler = logging.FileHandler(output)
        else:
            #stream handler for logging to stdout
            handler = logging.StreamHandler(sys.stdout)

        #set formatter
        formatter = logging.Formatter(self._format, datefmt=self._date_format)
        handler.setFormatter(formatter)

        #add handler to root logger
        self.root_logger.addHandler(handler)
        self.root_logger.setLevel(level)

        #store handler reference
        self._current_handler = handler


    def get_logger(self, name):
        """Get or create a logger with PathSim hierarchy.

        Returns a logger under the 'pathsim' namespace. The logger inherits
        configuration from the root logger but can be individually configured
        via set_level().

        Parameters
        ----------
        name : str
            Name of the logger, will be prefixed with 'pathsim.' to create
            hierarchical logger (e.g., 'simulation' -> 'pathsim.simulation').

        Returns
        -------
        logging.Logger
            Logger instance with the specified name under pathsim hierarchy.

        Examples
        --------

        .. code-block:: python

            mgr = LoggerManager()
            mgr.configure(enabled=True)

            # Get logger for simulation module
            sim_logger = mgr.get_logger("simulation")
            sim_logger.info("Starting simulation")

            # Get logger for progress tracking
            progress_logger = mgr.get_logger("progress.TRANSIENT")
            progress_logger.debug("Progress update")

        """

        #create full logger name with pathsim prefix
        full_name = f"pathsim.{name}"

        #get or create logger
        logger = logging.getLogger(full_name)

        #ensure logger propagates to root pathsim logger
        logger.propagate = True

        return logger


    def set_level(self, level, module=None):
        """Set logging level globally or for a specific module.

        Allows fine-grained control over logging verbosity. Can set the level
        for all loggers (when module=None) or for a specific logger in the
        hierarchy.

        Parameters
        ----------
        level : int
            Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING,
            logging.ERROR, logging.CRITICAL).
        module : str or None, optional
            Module name to set level for (e.g., 'progress', 'analysis.timer').
            If None, sets level for the root pathsim logger, affecting all
            child loggers that don't have their own level set. Defaults to None.

        Examples
        --------

        .. code-block:: python

            mgr = LoggerManager()
            mgr.configure(enabled=True)

            # Set global level to INFO
            mgr.set_level(logging.INFO)

            # Set debug level for progress tracking only
            mgr.set_level(logging.DEBUG, "progress")

            # Quiet analysis logs
            mgr.set_level(logging.WARNING, "analysis")

        """

        if module is None:
            #set level for root pathsim logger
            self.root_logger.setLevel(level)
            self._level = level
        else:
            #set level for specific module logger
            logger = self.get_logger(module)
            logger.setLevel(level)


    def is_enabled(self):
        """Check if logging is currently enabled.

        Returns
        -------
        bool
            True if logging is enabled, False otherwise.

        """
        return self._enabled


    def get_effective_level(self, module=None):
        """Get the effective logging level.

        Parameters
        ----------
        module : str or None, optional
            Module name to check level for. If None, returns root logger level.
            Defaults to None.

        Returns
        -------
        int
            The effective logging level (e.g., logging.INFO).

        """
        if module is None:
            return self.root_logger.level
        else:
            logger = self.get_logger(module)
            return logger.getEffectiveLevel()
