import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters

n = 3
m = 3
rank = 2

lt = pc._LinearTerm(n=n, m=m, rank=rank)
qt = pc._QuadraticTerm(n=n, rank=rank)
cs = pc._Constraints(n=n, m=m, rank=rank)

A1 = [[0,1,0],[1,0,0],[0,0,0]]
A2 = [[0,0,1],[0,0,0],[1,0,0]]
A3 = [[0,0,0],[0,0,1],[0,1,0]]
A = np.array([A1,A2,A3], dtype=np.float64)

b0 = np.array([2., 2., 2.])
bf = np.array([4, 4, 4])        

y = 1/np.sqrt(2)
Y_0 = np.array([[y,y],[y,y],[y,y]], dtype=np.float64)
lam_0 = np.array([.5,.5,.5]) 
 
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))
 
predcorr.run(b0, bf, A, A, Y_0, lam_0)
 