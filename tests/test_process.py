"""
Tests for the process module.
"""

import unittest
import numpy as np
from comprocess_ai.process import Processor, run


class TestProcessor(unittest.TestCase):
    """Test cases for the Processor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = Processor()
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertTrue(self.processor.is_initialized())
    
    def test_initialization_with_config(self):
        """Test processor initialization with config."""
        config = {"key": "value"}
        processor = Processor(config)
        self.assertEqual(processor.config, config)
    
    def test_process_data_normalization(self):
        """Test data normalization."""
        data = np.array([1, 2, 3, 4, 5])
        result = self.processor.process_data(data)
        
        # Check that data is normalized to [0, 1]
        self.assertAlmostEqual(np.min(result), 0.0)
        self.assertAlmostEqual(np.max(result), 1.0)
    
    def test_process_data_empty(self):
        """Test processing empty data."""
        data = np.array([])
        result = self.processor.process_data(data)
        self.assertEqual(len(result), 0)
    
    def test_process_data_single_value(self):
        """Test processing single value."""
        data = np.array([5])
        result = self.processor.process_data(data)
        self.assertEqual(result[0], 0.0)
    
    def test_process_data_list_input(self):
        """Test processing list input."""
        data = [1, 2, 3, 4, 5]
        result = self.processor.process_data(data)
        self.assertIsInstance(result, np.ndarray)
        self.assertAlmostEqual(np.min(result), 0.0)
        self.assertAlmostEqual(np.max(result), 1.0)


class TestRun(unittest.TestCase):
    """Test cases for the run function."""
    
    def test_run_without_data(self):
        """Test run function without input data."""
        result = run()
        
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["processor_initialized"])
        self.assertIsInstance(result["original_data"], list)
        self.assertIsInstance(result["processed_data"], list)
    
    def test_run_with_data(self):
        """Test run function with input data."""
        data = np.array([1, 2, 3, 4, 5])
        result = run(data)
        
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["processor_initialized"])
        self.assertEqual(len(result["original_data"]), 5)
        self.assertEqual(len(result["processed_data"]), 5)


if __name__ == "__main__":
    unittest.main()
