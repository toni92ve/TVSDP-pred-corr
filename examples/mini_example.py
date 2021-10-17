import pickle
import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters
 
n = 2
m = 1
rank = 1

lt = pc._LinearTerm(n=n, m=m, rank=rank)
qt = pc._QuadraticTerm(n=n, rank=rank)
cs = pc._Constraints(n=n, m=m, rank=rank)

A1 = [[0,0],[0,1]] 
A  = np.array([A1], dtype=np.float64)
b1 = np.array([1.], dtype=np.float64)  
b2 = np.array([2.], dtype=np.float64)  

Y_0 = np.array([[0],[1]], dtype=np.float64)
lam_0 = np.array([1]) 

predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))
output = np.full((m,),fill_value=0.0)
 
predcorr.run(b1, b2, A, A, Y_0, lam_0)