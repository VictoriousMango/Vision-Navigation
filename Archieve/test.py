# import numpy as np

# def generator():
#     count = 0
#     while True:
#         count += 1
#         yield count

import g2o
import numpy as np

def test_g2o_installation():
    """Test basic G2O functionality"""
    # Create optimizer
    optimizer = g2o.SparseOptimizer()
    print("✓ G2O optimizer created successfully.")
    
    # Try to create solver
    linear_solver = g2o.LinearSolverDense6d()
    print*("✓ G2O linear solver created successfully.")
    block_solver = g2o.BlockSolver6d(linear_solver)
    print("✓ G2O block solver created successfully.")
    algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
    print("✓ G2O optimization algorithm created successfully.")
    optimizer.set_algorithm(algorithm)
    print("✓ G2O algorithm set successfully.")
    
    # Add a simple vertex
    vertex = g2o.VertexSE3Expmap()
    print("✓ G2O vertex created successfully.")

if __name__=="__main__":
    test_g2o_installation()
    print("G2O installation test passed.")