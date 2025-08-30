"""
Test script for the NMT system
Run this to verify everything is working correctly
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow imported successfully (version: {tf.__version__})")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import train_test_split
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_dataset():
    """Test if the dataset can be loaded"""
    print("\nTesting dataset...")
    
    dataset_path = "english to bengali.csv (2)/english to bengali.csv"
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found at: {dataset_path}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        print(f"✓ Dataset loaded successfully: {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Show first few rows
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_nmt_system():
    """Test if the NMT system can be imported and initialized"""
    print("\nTesting NMT system...")
    
    try:
        from simple_transformer_nmt import SimpleNMTSystem
        
        # Initialize system
        nmt_system = SimpleNMTSystem(
            max_length=20,
            d_model=64,
            num_heads=2,
            num_layers=1,
            max_words=1000
        )
        print("✓ NMT system initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ NMT system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== NMT System Test Suite ===")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install required packages:")
        print("pip install -r requirements.txt")
        return False
    
    # Test dataset
    if not test_dataset():
        print("\n❌ Dataset test failed. Please check the dataset path.")
        return False
    
    # Test NMT system
    if not test_nmt_system():
        print("\n❌ NMT system test failed. Please check the implementation.")
        return False
    
    print("\n✅ All tests passed! The NMT system is ready to use.")
    print("\nTo start training, run:")
    print("python simple_transformer_nmt.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
