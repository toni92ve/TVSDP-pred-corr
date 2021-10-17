import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters
from visualize import visualize_sol

n = 3
m = 3
rank = 2

A1 = [[1,0,0],[0,0,0],[0,0,0]]
A2 = [[0,0,0.5],[0,0,0],[0.5,0,0]]
A3 = [[0,0,0],[0,0,0],[0,0,1]]
A0 = np.array([A1,A2,A3], dtype=np.float64)
Z = np.zeros((3,3))
A_lin = np.array([Z,Z,Z], dtype=np.float64)

b0 = np.array([2., 1., 1.])  
b_lin = np.array([2., 1., 1.])  
 
y1 = (np.sqrt(2)+np.sqrt(6))/4
y2 = (np.sqrt(2)-np.sqrt(6))/4

Y_0 = np.array([[1,1],[0,0],[y1,y2]], dtype=np.float64)
lam_0 = np.array([-1.,0.,-1.]) 
 
predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))
 
predcorr.run(A0, A_lin, b0, b_lin, Y_0, lam_0)
visualize_sol(params=getParameters(print_par=False))
 