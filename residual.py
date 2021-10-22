import  predictor_corrector as pc 
import numpy as np 

def resid(n: int, m:int, rank:int, b: np.ndarray, A: np.ndarray, Y: np.ndarray,
                 lam: np.ndarray, penalty: np.float) -> float:
    """
    Computes the residual.
    """
    enable_print = True

    _grad_AYY = np.full((m, n*rank), fill_value=0.0)
    lagran_grad = np.zeros((n*rank,)) 
    _YY = np.full((n, n), fill_value=0.0) 
    pc._getYY(Y=Y, out=_YY)
    constr_err = -b

    if enable_print: print("Inside the residual: ")
    # Compute Lagrangian residual 

    pc._getGradientAYY(n, m, rank, A, Y, _grad_AYY)  
    # if enable_print: print("self._grad_AYY\n", _grad_AYY)
    # if enable_print: print("times lam", np.matmul(lam, _grad_AYY))

    for i in range(n):
        for j in range(rank):
            lagran_grad[j+i*rank] += 2 * (1+(j+1)*penalty/rank)* Y[i,j]
    # if enable_print: print("OF",lagran_grad.ravel())

    lagran_grad += np.matmul(lam, _grad_AYY) 
    # if enable_print: print("OF+lam_grad",lagran_grad.ravel())
    
    resA = np.linalg.norm(lagran_grad.ravel(), np.inf)
    # if enable_print: print("resA",resA)

    # Compute constraints residual
 
    if enable_print: print("-b:", constr_err)

    for i in range(m):   
        if enable_print: print(np.dot(_YY.ravel(), A[i,:,:].ravel()))
        constr_err[i] += np.dot(_YY.ravel(), A[i,:,:].ravel())  
        if enable_print: print(constr_err[i])

    resB = np.linalg.norm(constr_err, np.inf)
    if enable_print: print("resB",resB)

    return max(resA, resB)
