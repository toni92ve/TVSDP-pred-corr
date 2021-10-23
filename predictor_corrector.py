import null_space_method as nsm 
import initial_point as ip
import numpy as np 
import residual
import pickle


def _get_YY(Y: np.ndarray, out: np.ndarray):
    """
    Computes: matrix YY^T
    """
    assert out.ndim == Y.ndim == 2
    n, n = out.shape
    assert Y.shape[0] == n and id(out) != id(Y) 

    np.dot(Y, Y.T, out=out)             # YY^T 
     
def _get_gradient_YY(n: int, rank: int, Y: np.ndarray, out: np.ndarray):
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
        
        np.copyto(self._lamTimesA,np.tensordot(lam, self._diffA, 1))
        _get_gradient_YY(n=self.n, rank=self.rank, Y=Y, out=self._gradYY) 
        np.matmul(self._gradYY.T, self._lamTimesA.ravel(), out=self._q) 

        for i in range(self.n):
            for j in range(self.rank):
                    self._q[j+i*self.rank] += 2 * (1+(j+1)*penalty/self.rank)* Y[i,j] 

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
        np.copyto(_sum_lam_A, np.tensordot(lam, A, 1))

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
 
        _get_YY(Y=Y, out=self._YY)
        _gradYY = np.full((n*n,n*rank), fill_value=0.0)
        _get_gradient_YY(n=n, rank=rank, Y=Y, out=_gradYY) 
 
        for i in range(m):
            self._d[i] = b[i]
            self._d[i] -= np.dot(self._YY.ravel(), A2[i,:,:].ravel())

        for cons_nr in range(m):
            self._C[cons_nr] = np.dot(A1[cons_nr].ravel(),_gradYY)
    
        return self._C, self._d

class _PredictorCorrector:

    def __init__(self, n: int, m: int, rank: int, params: dict):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """ 

        self._n = n
        self._m = m
        self._rank = rank

        # Create all necessary modules and pre-allocate the workspace 
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

        self._total_QP_runtime = 0.0
        self._total_SDP_runtime = 0.0

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
        
        res_0 = residual.resid(n=n, m=m, rank=rank, b=b0, A=A0,
                                            Y=self._Y, lam=self._lam, penalty=pen_coef )
        f = open(f"results/0.pkl", "wb")  
        pickle.dump([Y_0, np.matmul(Y_0,Y_0.T), None, lam_0, res_0, 0.0, 0.0, 0.0, 0, 0.], f)
        f.close()

        while curr_time < final_time: 
            
            print("ITERATION", iteration) 

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

            run_time_QP = nsm._SolveQP_NSpace(n=n, m=m, rank=rank, P=P, q=q, C=C, d=d, Y_0=self._Y.ravel(),
                         dY=self._candidate_Y, lam=self._candidate_lam )
            self._total_QP_runtime += run_time_QP
             
            self._candidate_Y += self._Y
             
            res = residual.resid(n=n, m=m, rank=rank, b=self._nextb, A=self._nextA,
                                            Y=self._candidate_Y, lam=self._candidate_lam, penalty=pen_coef)

            candidate_X = np.matmul(self._candidate_Y,self._candidate_Y.T)
            run_time_SDP, actual_X, actual_lam = ip._get_SDP_solution(n, m, self._nextA, self._nextb)
            self._total_SDP_runtime += run_time_SDP
            frob_error = np.linalg.norm(actual_X-candidate_X, 'fro')

            
            if False:
                print("current b term \n",  self._currb)  
                print("candidate Y\n", self._candidate_Y)
                print("candidate X\n", candidate_X)
                print("actual X\n", actual_X)
                print("actual lam\n", actual_lam)
                print("current lam\n", self._lam)
                print("candidate lam\n", self._candidate_lam)
                print("error in Fro norm", frob_error)
                print("res VS res_tol:",res, self._res_tol )
                print("q:\n",q)
                print("P:\n",P) 
                print("d:\n",d)
                print("C:\n",C)

            if res>res_tol: 
                dt *= gamma1
                next_time = curr_time + dt 
                reduction_steps += 1 
                print("reducing stepsize...")
                print("res =", res)
                # print("VS res tol =", res_tol)
                 
            else:

                # Update the solution
                np.copyto(self._Y, self._candidate_Y)
                np.copyto(self._lam, self._candidate_lam) 
                np.copyto(self._solution_X,np.matmul(self._Y,self._Y.T))

                f = open(f"results/{iteration+1}.pkl", "wb")  
                pickle.dump([self._Y,self._solution_X, actual_X,self._lam,res,dt,run_time_QP,run_time_SDP,reduction_steps, frob_error], f)
                f.close() 
 
                # Go to the next iteration. Try to shrink 'delta' a little bit.
                curr_time += dt
                iteration += 1
                reduction_steps = 0
                dt = min(final_time - curr_time, gamma2 * dt)
                next_time = curr_time+dt