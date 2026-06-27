import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from tests.test_engine import test_engine_folding
from tests.test_ab_initio import test_ab_initio_folding

if __name__ == "__main__":
    print("Running engine folding test...")
    try:
        test_engine_folding()
        print("Engine folding test passed.")
    except Exception as e:
        print(f"Engine folding test failed: {e}")
        sys.exit(1)
        
    print("\nRunning ab initio folding test...")
    try:
        test_ab_initio_folding()
        print("Ab initio folding test passed.")
    except Exception as e:
        print(f"Ab initio folding test failed: {e}")
        sys.exit(1)

    print("\nAll tests completed successfully.")
