import pickle
import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters
from visualize import visualize_sol

n = 4
m = 6
rank = 2

A1 = [[0,1,0,0],
      [1,0,0,0],
      [0,0,0,0],
      [0,0,0,0]]
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
A5 = [[0,0,0,0],
      [0,0,0,1],
      [0,0,0,0],
      [0,1,0,0]]
A6 = [[0,0,0,0],
      [0,0,0,0],
      [0,0,0,1],
      [0,0,1,0]]      

A = np.array([A1,A2,A3,A4,A5,A6], dtype=np.float64)
b = np.array([2., 2., 2., 2., 2., 2.])
 

y = 1/np.sqrt(2)
# Y_0 = np.array([[y,y],[y,y],[y,y],[y,y]], dtype=np.float64)
Y_0 = np.array([[1,0],[1,0],[1,0],[1,0]], dtype=np.float64)
# y = 1/np.sqrt(6)
# Y_0 = np.array([[y,y,y],[y,y,y],[y,y,y],[y,y,y]], dtype=np.float64)

l = 1/3
lam_0 = np.array([l,l,l,l,l,l]) 
 
predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))

predcorr.run(b, A, Y_0, lam_0)   
 
visualize_sol(params=getParameters(print_par=False))