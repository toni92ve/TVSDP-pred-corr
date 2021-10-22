import remove_redundancy as rr 
import numpy as np
import time

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
    



 

