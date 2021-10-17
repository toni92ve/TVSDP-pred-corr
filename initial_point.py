import numpy as np  
import scipy as sc
import mosek
import sys

class _InitialPoint:

    def __init__(self, A: np.ndarray, b: np.ndarray, n: np.int, m: np.int, rank: np.int):

        self._A = A
        self._b = b.tolist()
        self._n = n
        self._m = m
        self._rank = rank
        self._A_sparse = [[],[],[]]
        self._X0 = np.ndarray((n,n), dtype=np.float64)
        self._Y0 = np.ndarray((n,rank), dtype=np.float64)
        self._lam0 = np.ndarray(m, dtype=np.float64)

    def _sparsify(self):

        for k in range(self._m): 
            for h in range(3):
                sparse_data = sc.sparse.find(sc.sparse.tril(self._A[k]))[h].tolist()
                self._A_sparse[h].append(sparse_data)  

    # Define a stream printer to grab output from MOSEK
    def _streamprinter(self,text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def _get_SDP_solution(self, X0: np.ndarray, lam0: np.ndarray):
        n = self._n
        m = self._m

        self._sparsify()
        # Make mosek environment
        
        with mosek.Env() as env:

            # Create a task object and attach log stream printer
            with env.Task(0, 0) as task:
                task.set_Stream(mosek.streamtype.log, self._streamprinter)
                
                # Trace objective
                barci = list(range(0,n))
                barcj = list(range(0,n))
                barcval = [1.0]*n

                # Constraints RHS
                blc = self._b
                buc = blc

                # Constraints LHS
                barai = []
                baraj = []
                baraval = []
                
                for k in range(m):
                    barai.append(self._A_sparse[0][k])
                    baraj.append(self._A_sparse[1][k])
                    baraval.append(self._A_sparse[2][k]) 
                
                # Append  empty constraints and matrix variables 
                task.appendcons(m) 
                task.appendbarvars([n])

                # Set the bounds on constraints. 
                for i in range(m):
                    task.putconbound(i, mosek.boundkey.fx, blc[i], buc[i])
    
                symc = task.appendsparsesymmat(n, barci, barcj, barcval)
                task.putbarcj(0, [symc], [1.0])

                for k in range(self._m):
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
                    
                    np.copyto(X0, X)
                    np.copyto(lam0, [i * -1 for i in y])
                    
                elif (solsta == mosek.solsta.dual_infeas_cer or
                    solsta == mosek.solsta.prim_infeas_cer):
                    print("Primal or dual infeasibility certificate found.\n")
                elif solsta == mosek.solsta.unknown:
                    print("Unknown solution status")
                else:
                    print("Other solution status")

    def _get_initial_point(self):

        self._get_SDP_solution(self._X0, self._lam0)
        # The eigenvalues in descending order
        eig_dec = np.linalg.eigh(self._X0)
        eig_vals_X = np.flip(eig_dec[0]) 
        eig_vecs_X = np.flip(eig_dec[1],1)
            
        for i in range(self._rank):
            self._Y0[:,i] = eig_vecs_X[:,i] * np.sqrt(eig_vals_X[i])


        print(self._X0)
        print(self._Y0)
        print(self._lam0) 
        return self._Y0, self._lam0 