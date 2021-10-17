import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters
from visualize import visualize_sol

params=getParameters(print_par=False)
pen_coef = float(params["problem"]["pen_coef"])

n = 3
m = 4
rank = 2

A1 = [[1,0,0],
      [0,0,0],
      [0,0,0]]
A2 = [[0,0.5,0],
      [0.5,0,0],
      [0,0,0]]
A3 = [[0,0,0],
      [0,0,0.5],
      [0,0.5,0]]
A4 = [[0,0,0.5],
      [0,1,0],
      [0.5,0,0]]
A0 = np.array([A1,A2,A3,A4], dtype=np.float64)
Z = np.zeros((3,3))
A_lin = np.array([Z,Z,Z,Z], dtype=np.float64)

b0 = np.array([1., 0., 0., 1.])  
b_lin = np.array([ 0., 0., 0., 1.])  

x = 1/np.sqrt(2+pen_coef)
y = (1+pen_coef)/(2+pen_coef)
Y_0 = np.array([[1.,0],[0,x],[y,0]], dtype=np.float64)

lam_x = -(3+2*pen_coef)/(4+2*pen_coef)
lam_y = -(1+pen_coef) 
lam_0 = np.array([lam_x , 0., 0., lam_y]) 
 
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))
 
predcorr.run(A0, A_lin, b0, b_lin, Y_0, lam_0)
visualize_sol(params=getParameters(print_par=False))