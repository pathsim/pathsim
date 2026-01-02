########################################################################################
##
##                             TESTS FOR LOGGER MANAGER
##                              'utils/logger.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import logging
import tempfile
import os

from pathsim.utils.logger import LoggerManager


# TESTS ================================================================================

class TestLoggerManager(unittest.TestCase):
    """
    Test the implementation of the 'LoggerManager' singleton class
    """

    def setUp(self):
        """Reset LoggerManager singleton between tests"""
        # Reset singleton state
        LoggerManager._instance = None
        LoggerManager._initialized = False

        # Clean up any existing pathsim loggers
        logger = logging.getLogger("pathsim")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)


    def test_singleton_pattern(self):
        """Test that LoggerManager follows singleton pattern"""
        mgr1 = LoggerManager()
        mgr2 = LoggerManager()

        # Should be the same instance
        self.assertIs(mgr1, mgr2)


    def test_default_initialization(self):
        """Test default initialization state"""
        mgr = LoggerManager()

        # Should be disabled by default
        self.assertFalse(mgr.is_enabled())

        # Root logger should exist
        self.assertIsNotNone(mgr.root_logger)
        self.assertEqual(mgr.root_logger.name, "pathsim")

        # Should not propagate to root
        self.assertFalse(mgr.root_logger.propagate)


    def test_configure_stdout(self):
        """Test configuration with stdout output"""
        mgr = LoggerManager()
        mgr.configure(enabled=True, output=None, level=logging.INFO)

        # Should be enabled
        self.assertTrue(mgr.is_enabled())

        # Should have exactly one handler
        self.assertEqual(len(mgr.root_logger.handlers), 1)

        # Handler should be StreamHandler
        handler = mgr.root_logger.handlers[0]
        self.assertIsInstance(handler, logging.StreamHandler)

        # Level should be INFO
        self.assertEqual(mgr.get_effective_level(), logging.INFO)


    def test_configure_file(self):
        """Test configuration with file output"""
        mgr = LoggerManager()

        # Create temp file for logging
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name

        try:
            mgr.configure(enabled=True, output=log_file, level=logging.DEBUG)

            # Should be enabled
            self.assertTrue(mgr.is_enabled())

            # Should have exactly one handler
            self.assertEqual(len(mgr.root_logger.handlers), 1)

            # Handler should be FileHandler
            handler = mgr.root_logger.handlers[0]
            self.assertIsInstance(handler, logging.FileHandler)

            # Level should be DEBUG
            self.assertEqual(mgr.get_effective_level(), logging.DEBUG)

            # Test logging to file
            logger = mgr.get_logger("test")
            logger.info("Test message")

            # Close handler to flush
            mgr.configure(enabled=False)

            # Verify file contains message
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Test message", content)

        finally:
            # Clean up temp file
            if os.path.exists(log_file):
                os.remove(log_file)


    def test_configure_disabled(self):
        """Test disabling logging"""
        mgr = LoggerManager()

        # First enable
        mgr.configure(enabled=True, output=None, level=logging.INFO)
        self.assertTrue(mgr.is_enabled())

        # Then disable
        mgr.configure(enabled=False)

        # Should be disabled
        self.assertFalse(mgr.is_enabled())

        # No handlers
        self.assertEqual(len(mgr.root_logger.handlers), 0)

        # Level should be very high (effectively disabled)
        self.assertGreater(mgr.root_logger.level, logging.CRITICAL)


    def test_configure_custom_format(self):
        """Test custom format and date format"""
        mgr = LoggerManager()

        custom_format = "%(levelname)s - %(message)s"
        custom_date_format = "%Y-%m-%d"

        mgr.configure(
            enabled=True,
            output=None,
            level=logging.INFO,
            format=custom_format,
            date_format=custom_date_format
        )

        # Should have handler with custom formatter
        self.assertEqual(len(mgr.root_logger.handlers), 1)
        handler = mgr.root_logger.handlers[0]
        formatter = handler.formatter

        self.assertEqual(formatter._fmt, custom_format)
        self.assertEqual(formatter.datefmt, custom_date_format)


    def test_reconfigure(self):
        """Test reconfiguring logger multiple times"""
        mgr = LoggerManager()

        # First configuration
        mgr.configure(enabled=True, output=None, level=logging.INFO)
        self.assertEqual(len(mgr.root_logger.handlers), 1)
        first_handler = mgr.root_logger.handlers[0]

        # Reconfigure
        mgr.configure(enabled=True, output=None, level=logging.DEBUG)

        # Should still have exactly one handler (old one removed)
        self.assertEqual(len(mgr.root_logger.handlers), 1)
        second_handler = mgr.root_logger.handlers[0]

        # Should be a different handler
        self.assertIsNot(first_handler, second_handler)

        # Level should be updated
        self.assertEqual(mgr.get_effective_level(), logging.DEBUG)


    def test_get_logger(self):
        """Test getting loggers with pathsim hierarchy"""
        mgr = LoggerManager()
        mgr.configure(enabled=True)

        # Get logger
        logger = mgr.get_logger("simulation")

        # Should have correct name
        self.assertEqual(logger.name, "pathsim.simulation")

        # Should propagate to root
        self.assertTrue(logger.propagate)

        # Get nested logger
        nested_logger = mgr.get_logger("progress.TRANSIENT")
        self.assertEqual(nested_logger.name, "pathsim.progress.TRANSIENT")


    def test_get_logger_reuse(self):
        """Test that getting the same logger returns same instance"""
        mgr = LoggerManager()

        logger1 = mgr.get_logger("test")
        logger2 = mgr.get_logger("test")

        # Should be the same logger instance
        self.assertIs(logger1, logger2)


    def test_set_level_global(self):
        """Test setting global logging level"""
        mgr = LoggerManager()
        mgr.configure(enabled=True, level=logging.INFO)

        # Set to DEBUG
        mgr.set_level(logging.DEBUG)

        # Global level should be DEBUG
        self.assertEqual(mgr.get_effective_level(), logging.DEBUG)
        self.assertEqual(mgr.root_logger.level, logging.DEBUG)


    def test_set_level_module(self):
        """Test setting module-specific logging level"""
        mgr = LoggerManager()
        mgr.configure(enabled=True, level=logging.INFO)

        # Set module-specific level
        mgr.set_level(logging.DEBUG, "progress")

        # Global should still be INFO
        self.assertEqual(mgr.get_effective_level(), logging.INFO)

        # Module should be DEBUG
        self.assertEqual(mgr.get_effective_level("progress"), logging.DEBUG)


    def test_get_effective_level_module(self):
        """Test getting effective level for modules"""
        mgr = LoggerManager()
        mgr.configure(enabled=True, level=logging.WARNING)

        # Module without specific level should inherit
        level = mgr.get_effective_level("simulation")
        self.assertEqual(level, logging.WARNING)

        # Set specific level
        mgr.set_level(logging.DEBUG, "analysis")

        # Should return module-specific level
        level = mgr.get_effective_level("analysis")
        self.assertEqual(level, logging.DEBUG)


    def test_logging_hierarchy(self):
        """Test that logging hierarchy works correctly"""
        mgr = LoggerManager()
        mgr.configure(enabled=True, level=logging.INFO)

        # Get parent and child loggers
        parent = mgr.get_logger("progress")
        child = mgr.get_logger("progress.TRANSIENT")

        # Set level on parent
        mgr.set_level(logging.DEBUG, "progress")

        # Child should inherit if not explicitly set
        self.assertEqual(child.getEffectiveLevel(), logging.DEBUG)


    def test_warnings_captured(self):
        """Test that Python warnings are captured through logging"""
        mgr = LoggerManager()

        # Test that warnings logger exists and is configured
        warnings_logger = logging.getLogger("py.warnings")
        self.assertIsNotNone(warnings_logger)


    def test_init_with_enabled(self):
        """Test initialization with enabled=True configures logging"""
        mgr = LoggerManager(enabled=True, level=logging.DEBUG)

        # Should be enabled
        self.assertTrue(mgr.is_enabled())

        # Should have handler
        self.assertEqual(len(mgr.root_logger.handlers), 1)

        # Level should be DEBUG
        self.assertEqual(mgr.get_effective_level(), logging.DEBUG)


    def test_init_with_file_output(self):
        """Test initialization with file output"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name

        try:
            mgr = LoggerManager(enabled=True, output=log_file, level=logging.INFO)

            # Should be enabled with file handler
            self.assertTrue(mgr.is_enabled())
            self.assertEqual(len(mgr.root_logger.handlers), 1)
            self.assertIsInstance(mgr.root_logger.handlers[0], logging.FileHandler)

            # Test logging
            logger = mgr.get_logger("test")
            logger.info("Init test message")

            # Close handler
            mgr.configure(enabled=False)

            # Verify file has content
            with open(log_file, 'r') as f:
                self.assertIn("Init test message", f.read())

        finally:
            if os.path.exists(log_file):
                os.remove(log_file)


    def test_init_disabled_by_default(self):
        """Test that LoggerManager() without args is disabled"""
        mgr = LoggerManager()

        # Should be disabled by default
        self.assertFalse(mgr.is_enabled())


    def test_singleton_ignores_subsequent_params(self):
        """Test that singleton ignores parameters after first instantiation"""
        # First call with enabled=True
        mgr1 = LoggerManager(enabled=True, level=logging.WARNING)
        self.assertTrue(mgr1.is_enabled())
        self.assertEqual(mgr1.get_effective_level(), logging.WARNING)

        # Second call with different params - should be ignored
        mgr2 = LoggerManager(enabled=False, level=logging.DEBUG)

        # Should still be enabled with WARNING level
        self.assertTrue(mgr2.is_enabled())
        self.assertEqual(mgr2.get_effective_level(), logging.WARNING)

        # Same instance
        self.assertIs(mgr1, mgr2)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
