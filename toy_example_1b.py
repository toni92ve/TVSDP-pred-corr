import pickle
import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters

n = 3
m = 3
rank = 2

lt = pc._LinearTerm(n=n, m=m, rank=rank)
qt = pc._QuadraticTerm(n=n, rank=rank)
cs = pc._Constraints(n=n, m=m, rank=rank)

A1 = [[1,0,0],[0,0,0],[0,0,0]]
A2 = [[0,0,1],[0,0,0],[1,0,0]]
A3 = [[0,0,0],[0,0,1],[0,1,0]]
A = np.array([A1,A2,A3], dtype=np.float64)
b0 = np.array([1., 1., 1.])
b1 = np.array([1.05, 1.05, 1.])
bf = np.array([2., 2., 1.])        

Y_0 = np.array([[1,0],[.5,.5],[.5,.5]], dtype=np.float64)
Y_1 = np.array([[1.02476394, 0.01558481],
                 [0.5108176,  0.49216977],
                 [0.50468668, 0.4921699 ]])

y_0 = Y_0.ravel()
constant_term = 2*y_0

out = np.ndarray((n*rank,m),dtype=np.float64)
for cons_nr in range(m):
    for h in range(n):
        for k in range(rank):  
            out[h*rank+k, cons_nr] = 2 * np.dot(A[cons_nr,h,:],Y_0[:,k])

solution=np.linalg.lstsq(out,constant_term,rcond=None)
print(solution[0])
# lam_0 = solution[0]
lam_0 = [1.,0.,1.]
# lam_1 = [-1.01080577, -0.01540662, -0.99217523]
lam_1 = [1.00312006,  0.00624413 ,1.00327856]  # OBTAINED BY LEAST SQUARES
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=getParameters(print_par=False))
 

predcorr.run(b0, bf, A, A, Y_0, lam_0)
# predcorr.run(b1, bf, A, A, Y_1, lam_1)

