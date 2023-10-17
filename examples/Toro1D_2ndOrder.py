import burgersDA as burgers
import numpy as np

InputParameters = {
            "Initial Condition"     : "Toro 1D",
            "Block Dimensions"      : np.array([1.5,0.1,0.1]), 
            "Number of Cells"       : np.array([75,5,5]), 
            "Reconstruction Order"  : 2,
            "Time Integration Order": 2,
            "Maximum Time"          : 0.5,
            "CFL"                   : 0.6,
            "Limiter"               : "VanLeer"
        }

Toro1DSolver = burgers.Solver(InputParameters)
Toro1DSolver.time_integrate()
burgers.Plotter.plot1D(Toro1DSolver.solutionBlock,0)
Toro1DSolver.solutionBlock.apply_BCs()
burgers.Plotter.plot2D(Toro1DSolver.solutionBlock,[1,2],57)
burgers.Plotter.plot2D(Toro1DSolver.solutionBlock,[0,1])
burgers.Plotter.plot2D(Toro1DSolver.solutionBlock,[0,2])
