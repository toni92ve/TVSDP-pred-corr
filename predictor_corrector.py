import time
import pickle
import numpy as np 
import remove_redundancy as rr 

def _getYY(Y: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    Computes: matrix YY^T
    """
    assert out.ndim == Y.ndim == 2
    n, n = out.shape
    assert Y.shape[0] == n and id(out) != id(Y) 

    np.dot(Y, Y.T, out=out)             # YY^T 
    return out.ravel()                  # returned in vectorial form

def _getGradientYY(n: int, rank: int, Y: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    Computes: grad YY^T
    """ 
    assert Y.shape == (n,rank) 
    assert out.shape == (n*n,n*rank)
    for alpha in range(n*n):
        for beta in range(n*rank):
            (i,j) = np.divmod(alpha,n)
            (h,k) = np.divmod(beta,rank)
            if i == h:
                out[alpha,beta] += Y[j,k] 
            if j == h:
                out[alpha,beta] += Y[i,k] 
    return out

def _getGradientAYY(n: int, m:int, rank: int, A:np.ndarray, Y: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    Computes: grad <A,YY^T> 
    """ 
    for cons_nr in range(m):
        for i in range(n):
            for j in range(rank):   
                out[cons_nr, i*rank+j] = 2 * np.dot(A[cons_nr,i,:],Y[:,j]) 
    return out

def _getLamTimesA(lam: np.ndarray, A: np.ndarray, out: np.ndarray) -> np.ndarray:
     
    for i in range(lam.size): 
        out +=  lam[i] * A[i,:,:] 
    return out

def _getLinear(n: np.int, rank: np.int, Y: np.ndarray, gradYY: np.ndarray, lamTimesA: np.ndarray, penalty: np.float,
                out: np.ndarray) -> np.ndarray:

    np.matmul(gradYY, lamTimesA, out=out) 
    for i in range(n):
        for j in range(rank):
                out[j+i*rank] += 2 * (1+(j+1)*penalty/rank)* Y[i,j]

    return out

class _AccuracyCriterion:

    def __init__(self, n: int, m: int, rank: int):
        """
        Constructor pre-allocates caches for the temporary variables.
        """ 

        self._grad_AYY = np.full((m, n*rank), fill_value=0.0)
        self._YY = np.full((n, n), fill_value=0.0)
        self.constr_err = np.full((m, ), fill_value=0.0)
        self.lagran_grad = np.zeros((n*rank,))

    def residual(self, n: int, m:int, rank:int, b: np.ndarray, A: np.ndarray, Y: np.ndarray,
                 lam: np.ndarray, penalty: np.float) -> float:
        """
        Computes the residual.
        """
        # print("Inside the residual: ")
        # Compute Lagrangian residual
        # print(Y, lam, A, b)
        _getGradientAYY(n, m, rank, A, Y, self._grad_AYY)  
        # print("self._grad_AYY\n", self._grad_AYY)
        # print("times lam", np.matmul(lam, self._grad_AYY))
        # print(self.lagran_grad.ravel())
        for i in range(n):
            for j in range(rank):
                self.lagran_grad[j+i*rank] += 2 * (1+(j+1)*penalty/rank)* Y[i,j]
        # print(self.lagran_grad.ravel())
        self.lagran_grad += np.matmul(lam, self._grad_AYY) 
        # print(self.lagran_grad.ravel())
        resA = np.linalg.norm(self.lagran_grad.ravel(), np.inf)
        # print("resA",resA)
        # Compute constraints residual
        _getYY(Y=Y, out=self._YY)
        np.copyto(self.constr_err, -b)
        for i in range(m):   
            self.constr_err[i] += np.dot(self._YY.ravel(), A[i,:,:].ravel())  
        resB = np.linalg.norm(self.constr_err, np.inf)
        # print("resB",resB)
        return max(resA, resB)

class _LinearTerm:

    def __init__(self, n: int, m: int, rank: int):
        """
        Constructor pre-allocates caches for the temporary variables.
        """ 

        self.n = n
        self.rank = rank

        self._diffA = np.full((m,n,n), fill_value=0.0)
        self._lamTimesA = np.full((n,n), fill_value=0.0)
        self._gradYY = np.full((n*n,n*rank), fill_value=0.0)
        self._q = np.full((n*rank,), fill_value=0.0)

    def compute(self, A1: np.ndarray, A2: np.ndarray, 
                Y: np.ndarray, lam: np.ndarray, penalty: np.float) -> np.ndarray:
        """
        Computes the linear term of objective function in QP problem.
        """ 

        np.copyto(self._diffA, A2)
        self._diffA -= A1                       # diffA = A2 - A1
        
        _getLamTimesA(lam=lam, A=self._diffA, out=self._lamTimesA)
        _getGradientYY(n=self.n, rank=self.rank, Y=Y, out=self._gradYY)
        _getLinear(n=self.n, rank=self.rank,Y=Y, gradYY=self._gradYY.T ,lamTimesA=self._lamTimesA.ravel(), penalty = penalty, out=self._q)
        return self._q.ravel()
 
class _QuadraticTerm:

    def __init__(self, n: int, rank: int): 

        self._n = n
        self._rank = rank
        self._nvars = n*rank 

    def compute(self, A: np.ndarray, lam: np.ndarray, penalty: np.float) -> np.ndarray:
        
        n = self._n
        rank = self._rank
        nvars = n*rank
        
        _P = np.full((self._nvars,self._nvars), fill_value=0.0)

        _sum_lam_A = np.full((self._n,self._n), fill_value=0.0)
        _getLamTimesA(lam, A, out=_sum_lam_A)

        for alpha in range(nvars):
            for beta in range(nvars):
                (i,j) = np.divmod(alpha,rank)
                (h,k) = np.divmod(beta,rank)
                if j == k:
                    _P[alpha,beta] += 2*_sum_lam_A[i,h]
                    if i == h:
                        _P[alpha,beta] += 2
                        _P[alpha,beta] += 2*penalty*(j+1)/rank
         
        return _P

class _Constraints:
    
    def __init__(self, n: int, m: int, rank: int):
        """
        Constructor pre-allocates and pre-computes persistent data structures.
        For more readable formulation see the file: test_constraints.py.
        """ 

        self._n = n
        self._m = m
        self._rank = rank
        nvars = n * rank        # number of variables in matrix Y 
 
        self._A = np.full((m, 2 * rank), fill_value=0.0)
        self._d = np.full((m,), fill_value=0.0) 

        self._YY = np.full((n,n), fill_value=0.0)
        self._C = np.full((m,nvars), fill_value=0.0) 

    def compute(self,
                 A1: np.ndarray, A2: np.ndarray,
                 b: np.ndarray,
                 Y: np.ndarray):
         
        n, m, rank = self._n, self._m, self._rank
 
        _getYY(Y=Y, out=self._YY)
 
        for i in range(m):
            self._d[i] = b[i]
            self._d[i] -= np.dot(self._YY.ravel(), A2[i,:,:].ravel())
         
        for cons_nr in range(m):
                for i in range(n):
                    for j in range(rank):  
                        self._C[cons_nr, i*rank+j] = 2 * np.dot(A1[cons_nr,i,:],Y[:,j])
                         
        return self._C, self._d

class _PredictorCorrector:

    def __init__(self, n: int, m: int, rank: int, params: dict):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """ 

        self._n = n
        self._m = m
        self._rank = rank

        # Create all necessary modules and pre-allocate the workspace
        self._accuracy_crit =_AccuracyCriterion(n=n, m=m, rank=rank)  
        self._lin_term = _LinearTerm(n=n, m=m, rank=rank)
        self._quad_term = _QuadraticTerm(n=n, rank=rank)
        self._constraints = _Constraints(n=n, m=m, rank=rank)
        self._currb = np.zeros((m, ))
        self._nextb = np.zeros((m, ))
        self._currA = np.zeros((m,n,n))
        self._nextA = np.zeros((m,n,n))
        self._Y = np.zeros((n, rank)) 
        self._lam = np.zeros((m, ))   
        self._candidate_Y = np.zeros((n, rank))
        self._candidate_lam = np.zeros((m, )) 
        self._solution_X = np.zeros((n, n))

        # Parameters
        self._base_delta = float(params["problem"]["delta"])
        self._delta = self._base_delta
        self._delta_expand = float(params["problem"]["delta_expand"])
        self._delta_shrink = np.power(1.0 / self._delta_expand, 0.25).item()
        self._max_delta_expansions = int(params["problem"]["max_delta_expansions"])
        self._eta1 = float(params["problem"]["eta1"])
        self._eta2 = float(params["problem"]["eta2"])
        self._gamma1 = float(params["problem"]["gamma1"])
        self._gamma2 = float(params["problem"]["gamma2"])
        self._max_retry_attempts = float(params["problem"]["max_retry_attempts"])
        self._res_tol = float(params["problem"]["res_tol"])
        self._pen_coef = float(params["problem"]["pen_coef"])
        self._min_delta_t = float(params["problem"]["min_delta_t"])
        self._ini_stepsize = float(params["problem"]["ini_stepsize"])
        self._final_time = float(params["problem"]["final_time"])
        self._initial_time = float(params["problem"]["initial_time"])
        self._verbose = int(params["verbose"])

    def run(self, A0: np.ndarray, A_lin: np.ndarray, 
                  b0: np.ndarray, b_lin: np.ndarray,
                  Y_0: np.ndarray, lam_0: np.ndarray): 

        # Get copies of all problem parameters
        final_time = self._final_time 
        initial_time = self._initial_time

        n, m, rank = self._n,  self._m, self._rank

        gamma1 = self._gamma1
        gamma2 = self._gamma2 
        res_tol = self._res_tol
        pen_coef = self._pen_coef
        min_delta_t = self._min_delta_t
        dt = self._ini_stepsize  
        assert 0 < min_delta_t <= dt <= 1.0 

        np.copyto(self._Y, Y_0)
        np.copyto(self._lam, lam_0) 

        iteration = 0 
        reduction_steps = 0
        
        curr_time = initial_time
        next_time = initial_time + dt
          
        res_0 = self._accuracy_crit.residual(n=n, m=m, rank=rank, b=b0, A=A0,
                                            Y=self._Y, lam=self._lam, penalty=pen_coef )
        f = open(f"results/0.pkl", "wb")  
        pickle.dump([Y_0,np.matmul(Y_0,Y_0.T),lam_0, res_0, 0.0, 0,0], f)
        f.close()

        while curr_time < final_time: 

            np.copyto(self._currA, A0 + A_lin*curr_time)
            np.copyto(self._nextA, A0 + A_lin*next_time)
            np.copyto(self._currb, b0 + b_lin*curr_time) 
            np.copyto(self._nextb, b0 + b_lin*next_time)  
             
            
            q = self._lin_term.compute(A1=self._currA, A2=self._nextA,
                                    Y=self._Y, lam=self._lam, penalty = pen_coef)

            P = self._quad_term.compute(A=self._currA, lam=self._lam, penalty = pen_coef)  
              
            C, d = self._constraints.compute(
                A1 = self._currA, A2=self._nextA,
                b = self._nextb, Y=self._Y) 


            success, run_time = _SolveQP_NSpace(n=n, m=m, rank=rank, P=P, q=q, C=C, d=d, Y_0=self._Y.ravel(),
                         dY=self._candidate_Y, lam=self._candidate_lam )
             
            # if success: print("QP solved! \n")  
            self._candidate_Y += self._Y
             
            res = self._accuracy_crit.residual(n=n, m=m, rank=rank, b=self._nextb, A=self._nextA,
                                            Y=self._candidate_Y, lam=self._candidate_lam, penalty=pen_coef)
            
             # print("ITERATION", iteration) 
            print("n =",n,"m =",m,"rank =",rank,)
            print("current b term \n",  self._currb)  
            print("candidate Y\n", self._candidate_Y)
            print("current X\n", np.matmul(self._Y,self._Y.T))
            print("current lam\n", self._lam)
            print("res VS res_tol:",res, self._res_tol )
            # print("q:\n",q)
            # print("P:\n",P) 
            # print("d:\n",d)
            # print("C:\n",C)

            if res>res_tol: 
                dt *= gamma1
                next_time = curr_time + dt 
                reduction_steps += 1 
                 
            else:

                # Update the solution
                np.copyto(self._Y, self._candidate_Y)
                np.copyto(self._lam, self._candidate_lam) 
                np.copyto(self._solution_X,np.matmul(self._Y,self._Y.T))

                f = open(f"results/{iteration+1}.pkl", "wb")  
                pickle.dump([self._Y,self._solution_X,self._lam,res,dt,run_time,reduction_steps], f)
                f.close() 
 
                # Go to the next iteration. Try to shrink 'delta' a little bit.
                curr_time += dt
                iteration += 1
                reduction_steps = 0
                dt = min(final_time - curr_time, gamma2 * dt)
                next_time = curr_time+dt
                 

def _SolveQP_NSpace(n: int, m:int, rank: int, P: np.ndarray, q: np.ndarray,
                    C: np.ndarray, d: np.ndarray, Y_0: np.ndarray,
                    dY: np.ndarray, lam: np.ndarray): 
    
    start_time = time.time() 

    "check for redunt constraints" 
    constr_reduction = False
     
    if m > np.linalg.matrix_rank(C):
        
        print("Reducing constraints...")
        constr_reduction = True
        rem_red = rr._remove_redundancy_svd(C,d) 
        clean_C = rem_red[0] 
        clean_d = rem_red[1] 
        clean_m = np.shape(clean_C)[0]

        redund_cons_index = [] 
        j = 0
        for i in range(m):
            if not np.array_equal(C[i], clean_C[j]): 
                redund_cons_index = np.append(i,redund_cons_index)
            else:  j += 1
            if j >= clean_m: break
        redund_cons_index = np.flip(redund_cons_index)
        if not m == clean_m:
            redund_cons_index = redund_cons_index.astype(int) 
        
        C = clean_C
        d = clean_d
        m = clean_m
   
    "finding the null space and its completion"
    Q = np.linalg.qr(C.T,'complete')[0]
    W = Q[:,:m]
    Z = Q[:,m:]     

    C_W = np.matmul(C, W)
    h = d - np.matmul(C,Y_0)
    g = q + np.matmul(P,Y_0)
    Y_W = np.linalg.solve(C_W, h) 
     
    "Solve system for Y_Z" 
    Z_trans_P = np.matmul(Z.T,P) 
    Z_trans_P_Z = np.matmul(Z_trans_P,Z)  
    Z_trans_P_W = np.matmul(Z_trans_P,W)  
    r_term_1 = -np.matmul(Z_trans_P_W,Y_W)
    r_term_2 = -np.matmul(Z.T,g)
    r_term = r_term_1+r_term_2
     
    Y_Z = np.linalg.solve(Z_trans_P_Z,r_term)

    "Get the Y step"
    Y_sol_ns = np.matmul(W,Y_W) + np.matmul(Z,Y_Z)
    Y_step = Y_sol_ns + Y_0 
    np.copyto(dY, Y_step.reshape((n,rank)))
    
    "Get the lambda step"
    r_term_3 = g+np.matmul(P,Y_sol_ns)
    r_term_4 = np.matmul(W.T,r_term_3) 
    new_lambda = np.linalg.solve(-C_W.T, r_term_4)  
    
    "Add 0 as multiplier for the redundant constraints"
    if constr_reduction == True:
        for j in redund_cons_index:
            new_lambda = np.insert(new_lambda,j,0) 

    np.copyto(lam, new_lambda)

    run_time = time.time() - start_time

    return True , run_time
    



 

