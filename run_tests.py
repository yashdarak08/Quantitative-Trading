#!/usr/bin/env python
"""
Run all tests for the Quantitative Trading System.
"""

import unittest
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load test modules
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('tests', pattern='test_*.py')

# Run tests
test_runner = unittest.TextTestRunner(verbosity=2)
test_result = test_runner.run(test_suite)

# Return non-zero exit code if tests failed
sys.exit(not test_result.wasSuccessful())