import pickle
import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters

n = 4
m = 4
rank = 2

A1 = [[0.5841,    0.8178,    0.4253,    0.4229],
      [0.1078,    0.2607,    0.3127,    0.0942],
      [0.9063,    0.5944,    0.1615,    0.5985],
      [0.8797,    0.0225,    0.1788,    0.4709]]
A2 = [[0,0,1,0],
      [0,0,0,0],
      [1,0,0,0],
      [0,0,0,0]]
A3 = [[0,0,0,1],
      [0,0,0,0],
      [0,0,0,0],
      [1,0,0,0]]
A4 = [[0,0,0,0],
      [0,0,1,0],
      [0,1,0,0],
      [0,0,0,0]]   

A = np.array([A1,A2,A3,A4], dtype=np.float64)
b0 = np.array([0., 0., 0., 0.])
bf = np.array([2., 2., 2., 2.])        

Y_0 = np.array(
     [[ 1.7754,   -0.5091],
      [-1.9442,    1.3520],
      [1.3195,    2.6371],
      [-0.0258,   -2.0540]], 
     dtype=np.float64)
 
lam_0 = np.array([-8.3254,
    7.7897,
    7.2236,
    4.2793]) 
 
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))

predcorr.run(b0, bf, A, A, Y_0, lam_0) 