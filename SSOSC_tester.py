import numpy as np
import predictor_corrector_toni as pc 
from parameters_toni import getParameters
from visualize import visualize_sol
from scipy import optimize
from scipy import sparse
import mosek
import sys

params=getParameters(print_par=False)

n = 3
m = 3
rank = 2

A1 = [[0,.5,0],
      [.5,0,0],
      [0,0,0]]
A2 = [[0,0,.5],
      [0,0,0],
      [.5,0,0]]
A3 = [[0,0,0],
      [0,0,.5],
      [0,.5,0]]

A1_sparse = sparse.find(sparse.tril(A1))
A2_sparse = sparse.find(sparse.tril(A2))
A3_sparse = sparse.find(sparse.tril(A3))
A_sparse = np.array([A1_sparse,A2_sparse,A3_sparse]).T[0]

A0 = np.array([A1,A2,A3], dtype=np.float64)
Z = np.zeros((3,3))
A_lin = np.array([Z,Z,Z], dtype=np.float64)

b0 = np.array([1., 1., 1.])  
b_lin = np.array([1., 1., 1.])  

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def call_mosek(A_sparse: np.ndarray):
    # Make mosek environment
    with mosek.Env() as env:

        # Create a task object and attach log stream printer
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            # Bound values for constraints RH - side
            blc = [1.0,1.0,1.0]
            buc = [1.0,1.0,1.0]

            # Sparse representation of the matrix data

            barci = [0, 1, 2]
            barcj = [0, 1, 2]
            barcval = [1.0, 1.0, 1.0]

            barai = np.zeros((m,1,1)) 
            baraj = np.zeros((m,1,1)) 
            baraval = np.zeros((m,1,1))    
            for k in range(m):
                barai[k]=np.array([A_sparse[0][k]])
                baraj[k]=np.array([A_sparse[1][k]])
                baraval[k]=np.array([A_sparse[2][k]])
            
            # Append  empty constraints and matrix variables 
            task.appendcons(m) 
            task.appendbarvars([n])

            # Set the bounds on constraints. 
            for i in range(m):
                task.putconbound(i, mosek.boundkey.fx, blc[i], buc[i])
 
            symc = task.appendsparsesymmat(n,
                                           barci,
                                           barcj,
                                           barcval)
        
            syma0 = task.appendsparsesymmat(n,
                                            barai[0],
                                            baraj[0],
                                            baraval[0])
          
            syma1 = task.appendsparsesymmat(n,
                                            barai[1],
                                            baraj[1],
                                            baraval[1])

            syma2 = task.appendsparsesymmat(n,
                                            barai[2],
                                            baraj[2],
                                            baraval[2])
            
            task.putbarcj(0, [symc], [1.0])
            task.putbaraij(0, 0, [syma0], [1.0]) 
            task.putbaraij(1, 0, [syma1], [1.0]) 
            task.putbaraij(2, 0, [syma2], [1.0]) 

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            if (solsta == mosek.solsta.optimal):
                lenbarvar = n * (n + 1) / 2
                barx = [0.] * int(lenbarvar)
                task.getbarxj(mosek.soltype.itr, 0, barx)
                y = [0.] * m
                task.gety(mosek.soltype.itr,y)
                 
                X_lt = np.ndarray((n, n))
                for k in range(n):
                    X_lt[k] = np.append(np.zeros(k), barx[:n-k])
                    barx = barx[n-k:] 
                X = X_lt+X_lt.T
                for i in range(n):
                    X[i,i] *= .5
                 
                return X, [i * -1 for i in y]
                
               

            elif (solsta == mosek.solsta.dual_infeas_cer or
                  solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

# call the main function
try:
    X, lam_0 = call_mosek(A_sparse=A_sparse)
except mosek.MosekException as e:
    print("ERROR: %s" % str(e.errno))
    if e.msg is not None:
        print("\t%s" % e.msg)
        sys.exit(1)
except:
    import traceback
    traceback.print_exc()
    sys.exit(1)

# The eigenvalues in descending order
eig_vals_X = np.flip(np.linalg.eigh(X)[0])
eig_vecs_X = np.linalg.eigh(X)[1]
eig_vecs_X = np.flip(eig_vecs_X,1)
 
Y_0 = np.ndarray((n,rank), dtype=np.float64)
 
for i in range(rank):
     Y_0[:,i] = eig_vecs_X[:,i]*np.sqrt(eig_vals_X[i])
 
# sq= 1/np.sqrt(2)
# R1=np.array([[sq,sq],[-sq,sq]])
# Y_0 = np.matmul(Y_0,R1) 
# print(Y_0)
# pen_coef = float(params["problem"]["pen_coef"])
 
# eta = 1. + pen_coef/2
# lam_0 = np.array([-eta,-eta,-eta]) 
 
 
predcorr = pc.PredictorCorrector(n=n, m=m, rank=rank, params=params)

predcorr.run(A0, A_lin, b0, b_lin, Y_0, lam_0)
visualize_sol(params=params)
 