import cvxpy as cp  

def _SolveQP(n: int, m:int, rank: int, P: np.ndarray, q: np.ndarray,
             C: np.ndarray, d: np.ndarray, 
             dY: np.ndarray, lam: np.ndarray):
    """ Solves QP problem using CVXPY library. """
     
    # Define and solve the CVXPY problem.
    x = cp.Variable(n*rank ) 

    start_time = time.time() 
    objective = cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x)
     
    prob = cp.Problem(objective, [C @ x == d])
    prob.solve(solver=cp.SCS, use_indirect=False)
    # try:
    #     res = prob.solve(solver=cp.SCS, use_indirect=False)
    # except cp_error.DCPError:
    #     objective = cp.Minimize((1/2)*cp.quad_form(x, P+0.001*cp.diag([1.0]*n*rank)) + q.T @ x)
    #     prob = cp.Problem(objective, [C @ x == d])
    #     res = prob.solve(solver=cp.SCS, use_indirect=False)
    
    run_time = time.time() - start_time

    # Check status of the solver.
    ok = True 

    # Get solution for Y's increment and dual variables. 
    np.copyto(dY, x.value.reshape(n, rank))  
    np.copyto(lam,prob.constraints[0].dual_value) 
     
    return ok, run_time