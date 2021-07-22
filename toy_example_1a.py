import pickle
import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters

n = 3
m = 3
rank = 2

A1 = [[0,1,0],[1,0,0],[0,0,0]]
A2 = [[0,0,2],[0,0,0],[2,0,0]]
A3 = [[0,0,0],[0,0,3],[0,3,0]]

A = np.array([A1,A2,A3], dtype=np.float64)
b0 = np.array([0., 0., 1.])
bf = np.array([1., 1., 1.])        

Y_0 = np.array([[0, 0], [-0.4082, 0], [-0.4082, 0]])

lam_0 = np.array([0,0,1/3]) 
 
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))
 
predcorr.run(b0, bf, A, A, Y_0, lam_0)
 