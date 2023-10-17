import burgersDA as burgers
import numpy as np

InputParameters = {
            "Initial Condition"     : "Gaussian Bump 2D",
            "Block Dimensions"      : np.array([np.pi,np.pi,0.1]), 
            "Number of Cells"       : np.array([100,100,1]), 
            "Reconstruction Order"  : 2,
            "Time Integration Order": 2,
            "Maximum Time"          : 2,
            "CFL"                   : 0.4,
            "Limiter"               : "VanLeer"
        }

Toro1DSolver = burgers.Solver(InputParameters)
Toro1DSolver.time_integrate()
Toro1DSolver.solutionBlock.apply_BCs()
burgers.Plotter.plot2D(Toro1DSolver.solutionBlock,[0,1],None,0)
