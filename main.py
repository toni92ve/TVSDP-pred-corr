import predictor_corrector as pc 
import initial_point as ip
import examples_data  as ex
import parameters as par
import visualize as vis

problem_ID = input("Enter example ID:")
problem_input = "example_%s" %problem_ID

problem = ex._Examples(problem_input)

params = par.getParameters(print_par=True)
n, m, rank, A, A_lin, b, b_lin = problem._choose_example()
 
initial_point = ip._InitialPoint(A = A, b=b, n=n, m=m, rank=rank)
Y_0 , lam_0 = initial_point._get_initial_point()
 
predcorr = pc._PredictorCorrector(n=n, m=m, rank=rank, params=params)
predcorr.run(A, A_lin, b, b_lin, Y_0, lam_0)

vis.visualize_sol(params=params)
 