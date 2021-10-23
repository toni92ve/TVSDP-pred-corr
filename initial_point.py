import numpy as np  
import scipy as sc
import mosek
import time
import sys


# Define a stream printer to grab output from MOSEK
def _streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def _sparsify(m: np.int, A: np.ndarray):
        A_sparse = [list]*3
        for h in range(3): 
            sparse_datas = []
            for k in range(m):
                sparse_data = sc.sparse.find(sc.sparse.tril(A[k]))[h].tolist()
                sparse_datas.append(sparse_data)
            A_sparse[h] = sparse_datas  
        return A_sparse

def _get_SDP_solution(n: np.int, m: np.int, A: np.ndarray, b: np.ndarray):

    A_sparse = _sparsify(m, A)
     
    start_time = time.time() 
    # Make mosek environment
    
    with mosek.Env() as env:

        # Create a task object and attach log stream printer
        with env.Task(0, 0) as task:

            # task.set_Stream(mosek.streamtype.log, _streamprinter)
            
            # Trace objective
            barci, barcj, barcval = list(range(0,n)), list(range(0,n)), [1.0]*n 

            # Constraints RHS
            blc = b
            buc = blc

            # Constraints LHS
            barai, baraj, baraval = [], [], [] 
            
            for k in range(m):
                barai.append(A_sparse[0][k])
                baraj.append(A_sparse[1][k])
                baraval.append(A_sparse[2][k]) 
            
            # Append  empty constraints and matrix variables 
            task.appendcons(m) 
            task.appendbarvars([n])

            # Set the bounds on constraints. 
            for i in range(m):
                task.putconbound(i, mosek.boundkey.fx, blc[i], buc[i])

            symc = task.appendsparsesymmat(n, barci, barcj, barcval)
            task.putbarcj(0, [symc], [1.0])

            for k in range(m):
                syma = task.appendsparsesymmat(n, barai[k], baraj[k], baraval[k])
                task.putbaraij(k, 0, [syma], [1.0])  

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)

            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)

            # Get status information about the solution
            prosta = task.getprosta(mosek.soltype.itr)
            solsta = task.getsolsta(mosek.soltype.itr)

            run_time = time.time() - start_time

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
                
                return run_time, X, [i * -1 for i in y]
                
            elif (solsta == mosek.solsta.dual_infeas_cer or
                solsta == mosek.solsta.prim_infeas_cer):
                print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

def _get_initial_point(n: np.int, m: np.int, rank: np.int, A: np.ndarray, b: np.ndarray):

    run_time, X0, lam0 = _get_SDP_solution(n=n, m=m, A=A, b=b)

    
    # The eigenvalues in descending order
    eig_dec = np.linalg.eigh(X0)
    eig_vals_X = np.flip(eig_dec[0]) 
    eig_vecs_X = np.flip(eig_dec[1],1)
    
    Y0=np.ndarray((n,rank),dtype=np.float)
    for i in range(rank):
        Y0[:,i] = eig_vecs_X[:,i] * np.sqrt(eig_vals_X[i])

    # print(self._lam0)
    # print(self._A)
    # print("0\n:",np.tensordot(self._lam0, self._A, 0))
    # print("1\n:",)
    # print("2\n:",np.tensordot(self._lam0, self._A, 2))
    if np.all(np.linalg.eigvals(np.eye(n)+np.tensordot(lam0, A, 1)) > 0):
        print("OK")
        
        return Y0, lam0 
            
        