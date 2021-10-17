import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters
from visualize import visualize_sol

params=getParameters(print_par=False)

n = 3
m = 3
rank = 2

A1 = [[1,1,0],
      [1,0,0],
      [0,0,0]]
A2 = [[0,0,1],
      [0,0,0],
      [1,0,0]]
A3 = [[0,0,0],
      [0,0,1],
      [0,1,1]]
A0 = np.array([A1,A2,A3], dtype=np.float64)
Z = np.zeros((3,3))
A_lin = np.array([Z,Z,Z], dtype=np.float64)

b0 = np.array([1., 1., 1.])  
b_lin = np.array([1., 1., 1.])  
 
x = 1/np.sqrt(2)
y = 1/np.sqrt(8)
Y_0 = np.array([[x,0],[y,0],[x,0]], dtype=np.float64)
pen_coef = float(params["problem"]["pen_coef"])
eta = 1. + pen_coef/2
lam_x = -0.25*eta
lam_y = -0.625*eta
lam_0 = np.array([lam_x,lam_y,lam_x]) 
 
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))
 
predcorr.run(A0, A_lin, b0, b_lin, Y_0, lam_0)
visualize_sol(params=getParameters(print_par=False))