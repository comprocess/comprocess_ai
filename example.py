"""
Example usage of comprocess_ai.

This script demonstrates how to use the comprocess_ai package
for basic AI/ML data processing tasks.
"""

import numpy as np
from comprocess_ai import run
from comprocess_ai.process import Processor


def example_basic_usage():
    """Demonstrate basic usage with default parameters."""
    print("=" * 50)
    print("Example 1: Basic usage with default parameters")
    print("=" * 50)
    
    result = run()
    print(f"Status: {result['status']}")
    print(f"Original data (first 5): {result['original_data'][:5]}")
    print(f"Processed data (first 5): {result['processed_data'][:5]}")
    print()


def example_custom_data():
    """Demonstrate usage with custom data."""
    print("=" * 50)
    print("Example 2: Processing custom data")
    print("=" * 50)
    
    # Create custom data
    custom_data = np.array([10, 20, 30, 40, 50])
    print(f"Input data: {custom_data}")
    
    result = run(custom_data)
    print(f"Status: {result['status']}")
    print(f"Normalized output: {result['processed_data']}")
    print()


def example_processor_class():
    """Demonstrate direct use of Processor class."""
    print("=" * 50)
    print("Example 3: Using the Processor class directly")
    print("=" * 50)
    
    # Create processor with configuration
    config = {"description": "Custom processor"}
    processor = Processor(config)
    
    # Process some data
    data = np.array([100, 200, 300, 400, 500])
    print(f"Input data: {data}")
    
    processed = processor.process_data(data)
    print(f"Processed data: {processed}")
    print(f"Processor initialized: {processor.is_initialized()}")
    print()


def example_list_input():
    """Demonstrate processing with list input."""
    print("=" * 50)
    print("Example 4: Processing list input")
    print("=" * 50)
    
    # Use a list instead of numpy array
    list_data = [5, 15, 25, 35, 45]
    print(f"Input list: {list_data}")
    
    result = run(np.array(list_data))
    print(f"Status: {result['status']}")
    print(f"Processed data: {result['processed_data']}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 50)
    print("comprocess_ai Examples")
    print("=" * 50 + "\n")
    
    example_basic_usage()
    example_custom_data()
    example_processor_class()
    example_list_input()
    
    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
