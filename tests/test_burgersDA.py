import unittest
import numpy as np
from burgersDA import Block 

class TestBlockMethods(unittest.TestCase):
    
    def test_initialize_grid(self):

        IPs = {
            "Initial Condition"     : "Toro 1D",
            "Block Dimensions"      : np.array([1,1,1]),
            "Number of Cells"       : np.array([10,10,10]),
            "Reconstruction Order"  : 1
        }

        block = Block(IPs)
        block.initialize_grid()
        self.assertAlmostEqual(block.grid[2][2][2].X[0], 0.05)
        self.assertAlmostEqual(block.grid[10+2-1][10+2-1][10+2-1].X[1], 1-0.05)
        self.assertAlmostEqual(block.grid[-1][-1][-1].X[1], 0.95 + 2*0.1)

if __name__ == '__main__':
    unittest.main()