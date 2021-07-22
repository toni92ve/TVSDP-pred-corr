from operator import matmul
from numpy.core.fromnumeric import size
import osqp
import time
import pickle
import numpy as np 
import cvxpy as cp
from cvxpy import solvers
from parameters_toni import getParameters
from scipy import sparse
from scipy.linalg import null_space 
from scipy.linalg import svd
import remove_red as rr
import os
import glob

np.set_printoptions(edgeitems=30, linewidth=100000
    # ,formatter=dict(float=lambda x: "%.5g" % x)
    )
 

_STATUSES = {
    osqp.constant("OSQP_SOLVED")                      : "solved",
    osqp.constant("OSQP_SOLVED_INACCURATE")           : "solved inaccurate",
    osqp.constant("OSQP_MAX_ITER_REACHED")            : "maximum iterations reached",
    osqp.constant("OSQP_PRIMAL_INFEASIBLE")           : "primal infeasible",
    osqp.constant("OSQP_PRIMAL_INFEASIBLE_INACCURATE"): "primal infeasible inaccurate",
    osqp.constant("OSQP_DUAL_INFEASIBLE")             : "dual infeasible",
    osqp.constant("OSQP_DUAL_INFEASIBLE_INACCURATE") 
     : "dual infeasible inaccurate",
    # osqp.constant("OSQP_SIGINT")                      : "interrupted by user",
    osqp.constant("OSQP_TIME_LIMIT_REACHED")          : "run time limit reached",
    osqp.constant("OSQP_UNSOLVED")                    : "unsolved",
    osqp.constant("OSQP_NON_CVX")                     : "problem non convex",
}

_ACCEPTABLE_STATUSES = {
    _STATUSES[osqp.constant("OSQP_SOLVED")],
    _STATUSES[osqp.constant("OSQP_SOLVED_INACCURATE")],
    _STATUSES[osqp.constant("OSQP_PRIMAL_INFEASIBLE_INACCURATE")],
    _STATUSES[osqp.constant("OSQP_DUAL_INFEASIBLE_INACCURATE")]
}

_FATAL_STATUSES = {
    _STATUSES[osqp.constant("OSQP_UNSOLVED")],
    _STATUSES[osqp.constant("OSQP_NON_CVX")],
}

def _getInterpolatedMatrix(A1: np.ndarray, A2: np.ndarray,
                           tau: float, out: np.ndarray) -> np.ndarray:
    """
    Returns data matrix linearly interpolated between two timestamps t and t+1:
    out = A1 + tau * (A2 - A1)
    """
    assert isinstance(A1, np.ndarray) and isinstance(A2, np.ndarray)
    assert isinstance(out, np.ndarray) 
    assert A1.shape == A2.shape == out.shape
    # assert 0.0 <= tau <= 1.0

    np.copyto(out, A2)
    out -= A1
    out *= tau
    out += A1

    return out

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

def _getLinear(Y: np.ndarray, gradYY: np.ndarray, lamTimesA: np.ndarray,
                out: np.ndarray) -> np.ndarray:

    # SOME ASSERTIONS  
   
    np.matmul(gradYY, lamTimesA, out=out)
    out += Y.ravel()
    out += Y.ravel()

    return out

class _AccuracyCriterion:

    def __init__(self, n: int, m: int, rank: int):
        """
        Constructor pre-allocates caches for the temporary variables.
        """
        assert isinstance(n, int)
        assert isinstance(rank, int) and 0 < rank <= n

        self._grad_AYY = np.full((m, n*rank), fill_value=0.0)
        self._YY = np.full((n, n), fill_value=0.0)
        self.constr_err = np.full((m, ), fill_value=0.0)
        # self._tmp3 = np.full((T, N), fill_value=0.0)

    def residual(self, n: int, m:int, rank:int, b: np.ndarray, A: np.ndarray, Y: np.ndarray,
                 lam: np.ndarray) -> float:
        """
        Computes the residual.
        """
        # ASSERTIONS

        # Compute residual A.
        _getGradientAYY(n, m, rank, A, Y, self._grad_AYY) 
        res = Y.ravel() 
        # print("residual")
        # print(res)
        res += Y.ravel()
        # print(res)
        res += np.matmul(lam, self._grad_AYY)
        # print(res)
        resA = np.linalg.norm(res.ravel(), np.inf)
        
        # Compute residual B.
        _getYY(Y=Y, out=self._YY)
        self.constr_err = -b
        for i in range(m):  
            self.constr_err[i] += np.dot(self._YY.ravel(), A[i,:,:].ravel()) 
                            
        res = np.minimum(lam, self.constr_err) 
        resB = np.linalg.norm(res.ravel(), np.inf) 
        return max(resA, resB)

class _LinearTerm:

    def __init__(self, n: int, m: int, rank: int):
        """
        Constructor pre-allocates caches for the temporary variables.
        """
        assert isinstance(n, int)
        assert isinstance(rank, int) and 0 < rank <= n

        self.n = n
        self.rank = rank

        self._diffA = np.full((m,n,n), fill_value=0.0)
        self._lamTimesA = np.full((n,n), fill_value=0.0)
        self._gradYY = np.full((n*n,n*rank), fill_value=0.0)
        self._q = np.full((n*rank,), fill_value=0.0)

    def compute(self, A1: np.ndarray, A2: np.ndarray, 
                Y: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """
        Computes the linear term of objective function in QP problem.
        """
        assert A1.shape == A2.shape
        # assert M1.shape == self._M_LmM.shape == self._diffM.shape
        assert Y.ravel().shape == self._q.shape

        np.copyto(self._diffA, A2)
        self._diffA -= A1                       # diffA = A2 - A1
        
        _getLamTimesA(lam=lam, A=self._diffA, out=self._lamTimesA)
        _getGradientYY(n=self.n, rank=self.rank, Y=Y, out=self._gradYY)
        _getLinear(Y=Y, gradYY=self._gradYY.T ,lamTimesA=self._lamTimesA.ravel(), out=self._q)
        return self._q.ravel()
 
class _QuadraticTerm:

    def __init__(self, n: int, rank: int):

        assert isinstance(n, int)
        assert isinstance(rank, int) and 0 < rank <= n
        

        self._n = n
        self._rank = rank
        self._nvars = n*rank 

    def compute(self, A: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """
        Assembles a sparse, quadratic term matrix given current state.
        """  
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
         
        return _P

class _Constraints:
    
    def __init__(self, n: int, m: int, rank: int):
        """
        Constructor pre-allocates and pre-computes persistent data structures.
        For more readable formulation see the file: test_constraints.py.
        """
        assert isinstance(n, int) and isinstance(m, int)
        assert isinstance(rank, int) and 0 < rank <= n

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
                 Y: np.ndarray, lam: np.ndarray) -> (np.ndarray, np.ndarray):
         
        n, m, rank = self._n, self._m, self._rank

        # ASSERTIONS 

        _getYY(Y=Y, out=self._YY)
 
        for i in range(m):
            self._d[i] = b[i]
            self._d[i] -= np.dot(self._YY.ravel(), A2[i,:,:].ravel())
 
        for cons_nr in range(m):
                for i in range(n):
                    for j in range(rank):  
                        self._C[cons_nr, i*rank+j] = 2 * np.dot(A1[cons_nr,i,:],Y[:,j])
          
        return self._C, self._d

class PredictorCorrector:

    def __init__(self, n: int, m: int, rank: int, params: dict):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """
        # assert isinstance(T, int) and 
        assert isinstance(n, int)
        assert isinstance(rank, int) and isinstance(params, dict)

        # self._T = T
        self._n = n
        self._m = m
        self._rank = rank

        # Create all necessary modules and pre-allocate the workspace.
        self._accuracy_crit =_AccuracyCriterion(n=n, m=m, rank=rank)  
        self._lin_term = _LinearTerm(n=n, m=m, rank=rank)
        self._quad_term = _QuadraticTerm(n=n, rank=rank)
        self._constraints = _Constraints(n=n, m=m, rank=rank)
        self._currb = np.zeros((m, ))
        self._nextb = np.zeros((m, ))
        self._currA = np.zeros((m,n,n))
        self._nextA = np.zeros((m,n,n))
        self._Y = np.zeros((n, rank))
        self._dY = np.zeros((n, rank))
        self._lam = np.zeros((m, )) 
        self._d_lam = np.zeros((m, )) 

        # Parameters. Note, 'delta' can change over time, but 'base_delta'
        # remains the same. The latter defines the minimum value for the
        # parameter 'delta'.

        self._base_delta = float(params["problem"]["delta"])
        self._delta = self._base_delta
        self._delta_expand = float(params["problem"]["delta_expand"])
        self._delta_shrink = np.power(1.0 / self._delta_expand, 0.25).item()
        self._max_delta_expansions = int(params["problem"]["max_delta_expansions"])
        self._eta1 = float(params["problem"]["eta1"])
        self._eta2 = float(params["problem"]["eta2"])
        self._gamma1 = float(params["problem"]["gamma1"])
        self._gamma2 = 1.0 / self._gamma1
        self._max_retry_attempts = float(params["problem"]["max_retry_attempts"])
        self._res_tol = float(params["problem"]["res_tol"])
        self._min_delta_tau = float(params["problem"]["min_delta_tau"])
        self._ini_stepsize = float(params["problem"]["ini_stepsize"])
        self._final_time = float(params["problem"]["final_time"])
        self._verbose = int(params["verbose"])

    def run(self, b1: np.ndarray, b2: np.ndarray,
                  A1: np.ndarray, A2: np.ndarray,
                  Y_0: np.ndarray, lam_0: np.ndarray):
        """ Runs the inner loop of predictor-corrector algorithm between
            timestamps t and t+1. """ 

        # Get copies of all problem parameters.  
        final_time = self._final_time 
        n = self._n
        m = self._m
        rank = self._rank

        gamma1 = self._gamma1
        gamma2 = self._gamma2
        max_retry_attempts = self._max_retry_attempts
        res_tol = self._res_tol
        min_delta_tau = self._min_delta_tau
        dtau = self._ini_stepsize 
        verbose = self._verbose
        # assert 0 < min_delta_tau <= dtau <= 1.0 

        np.copyto(self._Y, Y_0)
        np.copyto(self._lam, lam_0) 

        iteration = 0
        tau = 0.0

        # while dtau > np.sqrt(np.finfo(np.float64).eps).item():

        # if verbose >= 1:
        #     print("+" if attempt == 0 else "a", end="", flush=True)

        f = open(f"results/0.pkl", "wb")  
        pickle.dump([Y_0,np.matmul(Y_0,Y_0.T),lam_0], f)
        f.close()

        while tau < final_time:

            _getInterpolatedMatrix(b1, b2, tau, self._currb)
            _getInterpolatedMatrix(b1, b2, tau + dtau, self._nextb)
            _getInterpolatedMatrix(A1, A2, tau, self._currA)
            _getInterpolatedMatrix(A1, A2, tau + dtau, self._nextA) 

            time = self._currb[0]/b1[0]

            print("\n ITERATION NR:", iteration, "\n interpolation coefficient =", tau,"\n"
            ,"time: ", time, "\n")
            
            print("current b term \n",  self._currb) 
            # print("next b term \n",  self._nextb) 
            print("current Y\n", self._Y)
            print("current X\n", np.matmul(self._Y,self._Y.T))
            print("current lam\n", self._lam)

            q = self._lin_term.compute(A1=self._currA, A2=self._nextA,
                                    Y=self._Y, lam=self._lam)
             
            P = self._quad_term.compute(A=self._currA, lam=self._lam)  

            C, d = self._constraints.compute(
                A1=self._currA, A2=self._nextA,
                b=self._nextb, Y=self._Y, lam=self._lam) 
            
           
            # print("P: \n", P)
            # print("C: \n", C, np.linalg.matrix_rank(C))

            print("\nSolving the QP with null space method... \n")

            _SolveQP_NSpace(n=n, m=m, rank=rank, P=P, q=q, C=C, d=d, Y_0=self._Y.ravel(),
                        dY=self._dY, lam=self._lam)
            # _SolveQP(n=n, m=m, rank=rank, P=P, q=q, C=C, d=d,
            #             dY=self._dY, lam=self._d_lam)
            
            # print("primal step:\n", self._dY)   
            # print("lambda step: \n", self._lam)
            # print("my calculations:", .5 + (tau+dtau)/2)

            self._dY += self._Y             # dY <-- Y + dY
            Y_new = self._dY                # alias to Y + dY
            X_new = np.matmul(Y_new,Y_new.T) 

            # self._d_lam += self._lam            # d_lam <-- lam + d_lam
            # lam_new = self._d_lam               # alias to lam + d_lam
 
            # res = self._accuracy_crit.residual(b=self._nextb, A=self._nextA,
                                            # Y=Y_new, lam=lam_new)

            f = open(f"results/{iteration+1}.pkl", "wb")  
            pickle.dump([Y_new,X_new,self._lam], f)
            f.close() 

            # Update the solution.
            np.copyto(self._Y, Y_new)
            # np.copyto(self._lam, lam_new) 
            # np.copyto(self._lam, solution[0]) 

            # Go to the next iteration. Try to shrink 'delta' a little bit.
            tau += dtau
            iteration += 1
            #dtau = min(final_time - tau, gamma2 * dtau)

def _SolveQP_NSpace(n: int, m:int, rank: int, P: np.ndarray, q: np.ndarray,
                    C: sparse.csc_matrix, d: np.ndarray, Y_0: np.ndarray,
                    dY: np.ndarray, lam: np.ndarray) -> bool: 
    
    "check for redunt constraints" 
    constr_reduction = False
    print(m, np.linalg.matrix_rank(C))
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
    
    print("W: \n", W)
    print("C: \n", C)
    print("clean_C: \n", clean_C)
    print("C_W: \n", C_W)
    h = d - np.matmul(C,Y_0)
    
    g = q + np.matmul(P,Y_0)
    Y_W = np.linalg.solve(C_W, h) 
    # print("Y_W: \n", Y_W)

    "Solve system for Y_Z"
    Z_trans_P = np.matmul(Z.T,P)
    # print("Z^TP=\n", Z_trans_P)
    Z_trans_P_Z = np.matmul(Z_trans_P,Z)
    # print("Z^TPZ=\n",Z_trans_P_Z)
    Z_trans_P_W = np.matmul(Z_trans_P,W) 
    # print("Z^TPW=\n",Z_trans_P_W)
    r_term_1 = -np.matmul(Z_trans_P_W,Y_W)
    r_term_2 = -np.matmul(Z.T,g)
    r_term = r_term_1+r_term_2
    Y_Z = np.linalg.solve(Z_trans_P_Z,r_term)
    # print("Y_Z: \n", Y_Z)

    "Get the Y step"
    Y_sol_ns = np.matmul(W,Y_W) + np.matmul(Z,Y_Z)
    # print("Y_sol_ns", Y_sol_ns)
    
    Y_step = Y_sol_ns + Y_0 
    np.copyto(dY, Y_step.reshape((n,rank)))
    
    "Get the lambda step"
    r_term_3 = g+np.matmul(P,Y_sol_ns)
    r_term_4 = np.matmul(W.T,r_term_3) 
    # print("r_term_3 :\n",r_term_3)
    # print("r_term_4 :\n",r_term_4)
    # print("g: \n", g)
    # print("g+Py: \n", r_term_3)
    
    new_lambda = np.linalg.solve(C_W.T, r_term_4)  
    
    "Add 0 as multiplier for the redundant constraints"
    if constr_reduction == True:
        for j in redund_cons_index:
            new_lambda = np.insert(new_lambda,j,0) 

    # print("new_lam: \n",new_lambda)
    np.copyto(lam, new_lambda)

    return True
    
def _SolveQP(n: int, m:int, rank: int, P: np.ndarray, q: np.ndarray,
             C: sparse.csc_matrix, d: np.ndarray, 
             dY: np.ndarray, lam: np.ndarray) -> bool:
    """ Solves QP problem using CVXPY library. """
    # ASSERTIONS 
 

    # Create an CVXPY object, setup workspace and solve the problem. 
  

    # Define and solve the CVXPY problem.
    x = cp.Variable(n*rank)

    start_time = time.time()
    objective = cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x)
     
    prob = cp.Problem(objective, [C @ x == d])
    res = prob.solve(solver=cp.SCS, use_indirect=False)
    run_time = time.time() - start_time

    # Check status of the solver.
    ok = True
    # assert isinstance(res.info.status, str)
    # if res.info.status in _ACCEPTABLE_STATUSES:
    #     print("QP problem status:", res.info.status)
    #     print("#iter: {:d}, obj_val: {:f}, pri_res: {:f}, dua_res: {:f}, "
    #           "run_time: {:f}".format(res.info.iter, res.info.obj_val,
    #                                   res.info.pri_res, res.info.dua_res,
    #                                   run_time))
    # elif res.info.status in _FATAL_STATUSES:
    #     raise RuntimeError("QP solver fatally failed with status: {:s}"
    #                        .format(res.info.status))
    # else:
    #     print("WARNING: QP solver failed with status: {:s}"
    #           .format(res.info.status))
    #     ok = False

    # Get solution for Y's increment and dual variables.
    
    if ok:    
        np.copyto(dY, x.value.reshape(n, rank))
        print("constr:", prob.constraints[0].dual_value)
        # for i in range(m):
        #     lam[i] = prob.constraints[i].dual_value 

    return ok


 
