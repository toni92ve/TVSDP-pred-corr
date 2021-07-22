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
b0 = np.array([2., 2., 2., 2.])
bf = np.array([4., 4., 4., 4.])        

Y_0 = np.array(
     [[ 1.7755,   -0.5082],
      [-1.9446,    1.3507],
      [1.3183,    2.6382],
      [-0.0247,   -2.0543]], 
     dtype=np.float64)
 
lam_0 = np.array([-8.3264,
    7.7902,
    7.2246,
    4.2793]) 
 
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))

predcorr.run(b0, bf, A, A, Y_0, lam_0) 