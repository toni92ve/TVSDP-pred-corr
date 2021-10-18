import predictor_corrector as pc 
import examples_data  as ex
import initial_point as ip
import parameters as par
import visualize as vis
import numpy as np
import os

problem_ID = input("Enter example ID:")
problem_input = "example_%s" %problem_ID


params = par.getParameters(print_par=True)
n, m, rank, A, A_lin, b, b_lin = ex._choose_example(problem_input)
 
Y_0 , lam_0  = ip._get_initial_point(n=n, m=m, rank=rank, A = A, b=b)

predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params)
predcorr.run(A, A_lin, b, b_lin, Y_0, lam_0)

vis.visualize_sol(params=params)
# print(A)

dirListing = os.listdir("results/") 
for f in dirListing:
    os.remove(os.path.join("results/", f))