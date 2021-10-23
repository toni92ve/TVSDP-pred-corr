import  predictor_corrector as pc 
import numpy as np 

def resid(n: int, m:int, rank:int, b: np.ndarray, A: np.ndarray, Y: np.ndarray,
                 lam: np.ndarray, penalty: np.float) -> float:
    """
    Computes the residual.
    """
    enable_print = False

    _YY = np.full((n, n), fill_value=0.0) 
    _grad_YY = np.full((n*n, n*rank), fill_value=0.0)
    _lamTimesA = np.full((n,n), fill_value=0.0)
    lagran_grad = np.zeros((n*rank,)) 

    pc._get_YY(Y=Y, out=_YY)
    pc._get_gradient_YY(n, rank, Y, _grad_YY)
    constr_err = -b

    if enable_print: print("Inside the residual: ")
   
    # Compute Lagrangian residual 
    np.copyto(_lamTimesA,np.tensordot(lam, A, 1))
    np.matmul(_grad_YY.T, _lamTimesA.ravel(), out=lagran_grad) 

    for i in range(n):
        for j in range(rank):
            lagran_grad[j+i*rank] += 2 * (1+(j+1)*penalty/rank)* Y[i,j] 
    
    # if enable_print: print("OF+lam_grad",lagran_grad.ravel())
    
    resA = np.linalg.norm(lagran_grad.ravel(), np.inf)
    if enable_print: print("resA",resA)

    # Compute constraints residual
 
    if enable_print: print("-b:", constr_err)

    for i in range(m):   
        if enable_print: print(np.dot(_YY.ravel(), A[i,:,:].ravel()))
        constr_err[i] += np.dot(_YY.ravel(), A[i,:,:].ravel())  
        if enable_print: print(constr_err[i])

    resB = np.linalg.norm(constr_err, np.inf)
    if enable_print: print("resB",resB)

    return max(resA, resB)
