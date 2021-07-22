import numpy as np
from scipy.sparse import csc_matrix
import osqp, time

_STATUSES = {
    osqp.constant("OSQP_SOLVED")                      : "solved",
    osqp.constant("OSQP_SOLVED_INACCURATE")           : "solved inaccurate",
    osqp.constant("OSQP_MAX_ITER_REACHED")            : "maximum iterations reached",
    osqp.constant("OSQP_PRIMAL_INFEASIBLE")           : "primal infeasible",
    osqp.constant("OSQP_PRIMAL_INFEASIBLE_INACCURATE"): "primal infeasible inaccurate",
    osqp.constant("OSQP_DUAL_INFEASIBLE")             : "dual infeasible",
    osqp.constant("OSQP_DUAL_INFEASIBLE_INACCURATE")  : "dual infeasible inaccurate",
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
    assert 0.0 <= tau <= 1.0
    np.copyto(out, A2)
    out -= A1
    out *= tau
    out += A1
    return out


def _getM_LmM(M: np.ndarray, lam: np.ndarray, mu: np.ndarray,
              out: np.ndarray) -> np.ndarray:
    """
    Computes: out = M .* (lam - mu).
    """
    np.copyto(out, lam)
    out -= mu
    out *= M
    return out


def _getYY(Y: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    Computes: matrix YY, where YY_{ij} = Y_i * Y_{j+T}^T,
    0 <= i < T, 0 <= j < N.
    """
    assert out.ndim == Y.ndim == 2
    T, N = out.shape
    assert Y.shape[0] == T + N and id(out) != id(Y)

    L = Y[: T, :]
    R = Y[T :, :]
    np.dot(L, R.T, out=out)             # YY = L * R^T
    return out


def _getY2_plus_gradYY(M_LmM: np.ndarray, Y: np.ndarray,
                       out: np.ndarray) -> np.ndarray:
    """
    Computes: 2 * Y + sum(M .* (lam - mu) .* grad(YY)).
    """
    assert out.ndim == M_LmM.ndim == 2
    assert out.shape == Y.shape and id(out) != id(Y)
    T, N = M_LmM.shape
    assert Y.shape[0] == T + N

    L = Y[: T, :]
    R = Y[T :, :]
    np.dot(M_LmM,   R, out=out[: T, :])
    np.dot(M_LmM.T, L, out=out[T :, :])  
    out += Y
    out += Y
    return out


class _AccuracyCriterion:
    def __init__(self, T: int, N: int, rank: int):
        """
        Constructor pre-allocates caches for the temporary variables.
        """
        assert isinstance(T, int) and isinstance(N, int)
        assert isinstance(rank, int) and 0 < rank <= T <= N

        self._tmp1 = np.full((T + N, rank), fill_value=0.0)
        self._tmp2 = np.full((T, N), fill_value=0.0)
        self._tmp3 = np.full((T, N), fill_value=0.0)

    def residual(self, D: np.ndarray, M: np.ndarray, Y: np.ndarray,
                 lam: np.ndarray, mu: np.ndarray, delta: float) -> float:
        """
        Computes the residual.
        """
        assert Y.shape == self._tmp1.shape
        assert D.shape == M.shape == lam.shape == mu.shape
        assert D.shape == self._tmp2.shape == self._tmp3.shape

        # Compute residual A.
        M_LmM = _getM_LmM(M=M, lam=lam, mu=mu, out=self._tmp2)
        res = _getY2_plus_gradYY(M_LmM=M_LmM, Y=Y, out=self._tmp1)
        resA = np.linalg.norm(res.ravel(), np.inf)

        # Compute residual B.
        diff = _getYY(Y=Y, out=self._tmp2)
        diff -= D
        diff -= delta                                   # Q = +YY - delta - D
        res = np.minimum(lam, diff, out=self._tmp3)
        res *= M
        resB = np.linalg.norm(res.ravel(), np.inf)

        # Compute residual C.
        np.negative(diff, out=diff)
        diff -= 2.0 * delta                             # Q = -YY - delta + D
        res = np.minimum(mu, diff, out=self._tmp3)
        res *= M
        resC = np.linalg.norm(res.ravel(), np.inf)

        return max(resA, resB, resC)


class _LinearTerm:
    
    def __init__(self, T: int, N: int, rank: int):
        """
        Constructor pre-allocates caches for the temporary variables.
        """
        assert isinstance(T, int) and isinstance(N, int)
        assert isinstance(rank, int) and 0 < rank <= T <= N

        self._q = np.full((T + N, rank), fill_value=0.0)
        self._M_LmM = np.full((T, N), fill_value=0.0)
        self._diffM = np.full((T, N), fill_value=0.0)

    def compute(self, M1: np.ndarray, M2: np.ndarray,
                Y: np.ndarray, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Computes the linear term of objective function in QP problem.
        """
        assert M1.shape == M2.shape == lam.shape == mu.shape
        assert M1.shape == self._M_LmM.shape == self._diffM.shape
        assert Y.shape == self._q.shape

        np.copyto(self._diffM, M2)
        self._diffM -= M1                       # diffM = M2 - M1
        _getM_LmM(M=self._diffM, lam=lam, mu=mu, out=self._M_LmM)
        _getY2_plus_gradYY(M_LmM=self._M_LmM, Y=Y, out=self._q)
        return self._q.ravel()


class _QuadraticTerm:
    def __init__(self, T: int, N: int, rank: int):
        """
        Constructor pre-allocates and pre-computes persistent data structures.
        N O T E: OSQP solver needs only upper triangular part of the quadratic
        term matrix, thus, we skip the lower one.
        For more readable formulation see the file: test_quadratic_term.py.
        """
        assert isinstance(T, int) and isinstance(N, int)
        assert isinstance(rank, int) and 0 < rank <= T <= N

        noff = T * N * rank     # number of off-diagonal elements (upper)
        nvars = (T + N) * rank  # number of variables in matrix Y
        nnz = noff + nvars      # total number of non-zeros (diagonal + upper)

        self._nvars = nvars
        self._nnz = nnz
        self._rank = rank

        # Allocate caches and index arrays of indices.
        self._tmp = np.full((T, N), fill_value=0.0)
        self._row = np.full((nnz,), fill_value=0, dtype=np.int)
        self._col = np.full((nnz,), fill_value=0, dtype=np.int)
        self._val = np.full((nnz,), fill_value=0.0)

        # Indices of variables in matrix Y.
        idx = np.arange(nvars, dtype=np.int).reshape(n, rank)

        # Add indices of off-diagonal elements (upper triangular submatrix).
        np.copyto(self._row[: noff].reshape(T * N, rank), np.repeat(idx[: T, :], repeats=N, axis=0))
        np.copyto(self._col[: noff].reshape(T * N, rank),
                  np.tile(idx[T : T + N, :], (T, 1)))

        # Add indices of diagonal elements.
        np.copyto(self._row[noff :], np.arange(nvars, dtype=np.int))
        np.copyto(self._col[noff :], self._row[noff :])

        # The diagonal elements of the matrix never changes.
        self._val[noff :].fill(2.0)

    def assemble(self, M: np.ndarray,
                 lam: np.ndarray, mu: np.ndarray) -> csc_matrix:
        """
        Assembles a sparse, quadratic term matrix given current state.
        """
        assert M.shape == lam.shape == mu.shape == self._tmp.shape
        T, N = M.shape
        nvars, rank = self._nvars, self._rank

        # Modify values of off-diagonal elements. Diagonal ones remain the same.
        np.copyto(self._val[: T * N * rank].reshape((T * N, rank)),
                  _getM_LmM(M=M, lam=lam, mu=mu, out=self._tmp).reshape(-1, 1))

        H = csc_matrix((self._val, (self._row, self._col)), shape=(nvars, nvars))
        assert H.nnz == self._nnz
        return H

    def get_structure_unittest(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Get values, row and column indices for unittest.
        """
        return self._val, self._row, self._col


class _Constraints:
    
    def __init__(self, T: int, N: int, rank: int):
        """
        Constructor pre-allocates and pre-computes persistent data structures.
        For more readable formulation see the file: test_constraints.py.
        """
        assert isinstance(T, int) and isinstance(N, int)
        assert isinstance(rank, int) and 0 < rank <= T <= N

        self._T = T
        self._N = N
        self._rank = rank
        nvars = (T + N) * rank  # number of variables in matrix Y
        nconstr = T * N         # number of upper OR lower bound constraints

        row = np.full((2, nconstr, 2 * rank), fill_value=0, dtype=np.int)
        col = np.full((2, nconstr, 2 * rank), fill_value=0, dtype=np.int)
        vin = np.full((2, nconstr, 2 * rank), fill_value=0, dtype=np.int)

        # Placeholders for the values of constraint matrix, lower and upper
        # bounds. We follow the notations adopted for OSQP solver.
        self._A = np.full((2, nconstr, 2 * rank), fill_value=0.0)
        self._l = np.full((2, nconstr), fill_value=0.0)
        self._u = np.full((2, nconstr), fill_value=0.0)

        # Indices of variables in matrix Y.
        idx = np.arange(nvars, dtype=np.int).reshape(T + N, rank)

        # Initialize column indices of constraint matrix A.
        np.copyto(col[0, :, : rank], np.repeat(idx[: T, :], repeats=N, axis=0))
        np.copyto(col[0, :, rank :], np.tile(idx[T : T + N, :], (T, 1)))

        # Initialize positions from where to pull variables Y during
        # the assembling of constraint matrix A. In this particular problem,
        # positions can be taken from column indices (mind the sub-ranges).
        np.copyto(vin[0, :, : rank], col[0, :, rank :])
        np.copyto(vin[0, :, rank :], col[0, :, : rank])

        # Initialize row indices. Broadcasting along the last dimension.
        np.copyto(row, np.arange(2 * nconstr,
                                 dtype=np.int).reshape((2, nconstr, 1)))

        # Upper ("lambda") and lower ("mu") sub-matrices are similar.
        np.copyto(col[1, :, :], col[0, :, :])
        np.copyto(vin[1, :, :], vin[0, :, :])

        self._row = row                 # row indices of constraint matrices
        self._col = col                 # column indices of constraint matrices
        self._vin = vin                 # value indices of constraint matrices
        self._tmp = np.full((T, N), fill_value=0.0)

    def assemble(self,
                 D1: np.ndarray, D2: np.ndarray,
                 M1: np.ndarray, M2: np.ndarray,
                 Y: np.ndarray, lam: np.ndarray, mu: np.ndarray,
                 delta: float, eta1: float, eta2: float
                 ) -> (csc_matrix, csc_matrix, np.ndarray, np.ndarray):
        """ Assembles sparse constraint matrices and vectors given
            current state. """
        NEG_INF = -float(np.power(np.finfo(np.float64).max, 0.1).item())
        T, N, rank = self._T, self._N, self._rank
        assert D1.shape == D2.shape == M1.shape == M2.shape == (T, N)
        assert D1.shape == lam.shape == mu.shape and Y.shape == (T + N, rank)

        YY = _getYY(Y=Y, out=self._tmp)

        # Pull the values from the current variables Y, flip the sign of the
        # left-hand sides of the lower-bound ("mu") constraints, and multiply
        # by the matrix M at time "tau".
        np.copyto(self._A.ravel(), Y.ravel()[self._vin.ravel()])
        np.negative(self._A[1, :, :], out=self._A[1, :, :])
        self._A[0, :, :] *= M1.reshape(-1, 1)
        self._A[1, :, :] *= M1.reshape(-1, 1)

        # Initially the left-hand sides of the constraints are unbounded.
        self._l.fill(NEG_INF)

        # Compute the right-hand sides of the constraints. Note, the entities
        # there are defined at time "tau + delta tau".
        self._u.fill(delta)
        self._u[0, :] += D2.ravel();  self._u[1, :] -= D2.ravel()
        self._u[0, :] -= YY.ravel();  self._u[1, :] += YY.ravel()
        self._u[0, :] *= M2.ravel();  self._u[1, :] *= M2.ravel()

        # Positions where strongly-active (SA) condition is met. Note,
        # data matrix D is taken at time "tau".
        YY -= D1                        # YY <-- YY - D1
        lam_sa = np.logical_and(
            np.isclose(YY, +delta, rtol=0.0, atol=eta1), lam >= eta2).ravel()
        mu_sa = np.logical_and(
            np.isclose(YY, -delta, rtol=0.0, atol=eta1),  mu >= eta2).ravel()
        assert lam_sa.shape == mu_sa.shape and mu_sa.size == T * N
        assert not np.any(np.logical_and(lam_sa, mu_sa)), \
            "'lambda' & 'mu' must never take nonzero values simultaneously"

        # Make equality constraints for the strongly active (SA) ones.
        self._l[0, lam_sa] = self._u[0, lam_sa]
        self._l[1,  mu_sa] = self._u[1,  mu_sa]
        
        # Slightly loose the strict equality. TODO: is that right?
        # self._l[0, lam_sa] -= 0.1 * eta1;  self._u[0, lam_sa] += 0.1 * eta1
        # self._l[1,  mu_sa] -= 0.1 * eta1;  self._u[1,  mu_sa] += 0.1 * eta1

        # Assemble constraint matrix.
        A = csc_matrix((self._A.ravel(), (self._row.ravel(), self._col.ravel())),
                       shape=(2 * T * N, Y.size))

        return A, self._l.ravel(), self._u.ravel()

    def get_structure_unittest(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """ Get values, row and column indices for unittest. """
        return self._A, self._row, self._col


def _SolveQP(P: csc_matrix, q: np.ndarray,
             A: csc_matrix, l: np.ndarray, u: np.ndarray,
             dY: np.ndarray, lam: np.ndarray, mu: np.ndarray,
             iter_counter: int,
             verbose: bool) -> bool:
    """ Solves QP problem using OSQP library. """
    T, N = mu.shape
    rank = dY.shape[1]
    assert dY.shape[0] == T + N
    assert P.shape == ((T + N) * rank, (T + N) * rank)
    assert q.shape == ((T + N) * rank,)
    assert A.shape == (2 * T * N, (T + N) * rank)
    assert lam.shape == mu.shape == (T, N)

    # Create an OSQP object, setup workspace and solve the problem.
    start_time = time.time()
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, verbose=verbose) #, eps_prim_inf=1.0, eps_dual_inf=1.0, eps_abs=1.0)
    res = prob.solve()
    run_time = time.time() - start_time

    # Check status of the solver.
    ok = True
    assert isinstance(res.info.status, str)
    if res.info.status in _ACCEPTABLE_STATUSES:
        print("QP problem status:", res.info.status)
        print("#iter: {:d}, obj_val: {:f}, pri_res: {:f}, dua_res: {:f}, "
              "run_time: {:f}".format(res.info.iter, res.info.obj_val,
                                      res.info.pri_res, res.info.dua_res,
                                      run_time))
    elif res.info.status in _FATAL_STATUSES:
        raise RuntimeError("QP solver fatally failed with status: {:s}"
                           .format(res.info.status))
    else:
        print("WARNING: QP solver failed with status: {:s}"
              .format(res.info.status))
        ok = False

    # Get solution for Y's increment and dual variables.
    if ok:
        f = open(f"{iter_counter}.pkl", "wb")
        pickle.dump(res, f)
        f.close()
        np.copyto(dY, res.x.reshape(T + N, rank))
        np.copyto(lam, res.y[: T * N].reshape(T, N))
        np.copyto( mu, res.y[T * N :].reshape(T, N))
    return ok


class PredictorCorrector:
    def __init__(self, N: int, rank: int, params: dict):
        """ Constructor pre-allocates and pre-computes persistent
            data structures. """
        # assert isinstance(T, int) and 
        assert isinstance(N, int)
        assert isinstance(rank, int) and isinstance(params, dict)

        # self._T = T
        self._N = N
        self._rank = rank

        # Create all necessary modules and pre-allocate the workspace.
        self._accuracy_crit = _AccuracyCriterion(T=T, N=N, rank=rank)
        self._lin_term = _LinearTerm(T=T, N=N, rank=rank)
        self._quad_term = _QuadraticTerm(T=T, N=N, rank=rank)
        self._constraints = _Constraints(T=T, N=N, rank=rank)
        self._currD = np.zeros((T, N))
        self._nextD = np.zeros((T, N))
        self._currM = np.zeros((T, N))
        self._nextM = np.zeros((T, N))
        self._Y = np.zeros((T + N, rank))
        self._dY = np.zeros((T + N, rank))
        self._lam = np.zeros((T, N))
        self._mu = np.zeros((T, N))
        self._lam_new = np.zeros((T, N))
        self._mu_new = np.zeros((T, N))

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
        self._verbose = int(params["verbose"])


    def run(self, D1: np.ndarray, D2: np.ndarray,
                  M1: np.ndarray, M2: np.ndarray,
                  Y0: np.ndarray, lam: np.ndarray, mu: np.ndarray
            ) -> (np.ndarray, np.ndarray, np.ndarray):
        """ Runs the inner loop of predictor-corrector algorithm between
            timestamps t and t+1. """
        assert isinstance(D1, np.ndarray) and isinstance(D2, np.ndarray)
        assert isinstance(M1, np.ndarray) and isinstance(M2, np.ndarray)
        assert isinstance(mu, np.ndarray) and isinstance(lam, np.ndarray)
        assert isinstance(Y0, np.ndarray)

        # Get copies of all problem parameters.
        delta = self._delta
        delta_expand = self._delta_expand
        delta_shrink = self._delta_shrink
        max_delta_expansions = self._max_delta_expansions
        eta1 = self._eta1
        eta2 = self._eta2
        gamma1 = self._gamma1
        gamma2 = self._gamma2
        max_retry_attempts = self._max_retry_attempts
        res_tol = self._res_tol
        min_delta_tau = self._min_delta_tau
        dtau = self._ini_stepsize
        verbose = self._verbose
        assert 0 < min_delta_tau <= dtau <= 1.0

        np.copyto(self._Y, Y0)
        np.copyto(self._lam, lam)
        np.copyto(self._mu, mu)

        attempt = 0
        tau = 0.0
        while dtau > np.sqrt(np.finfo(np.float64).eps).item():
            if verbose >= 1:
                print("+" if attempt == 0 else "a", end="", flush=True)

            _getInterpolatedMatrix(D1, D2, tau, self._currD)
            _getInterpolatedMatrix(M1, M2, tau, self._currM)
            _getInterpolatedMatrix(D1, D2, tau + dtau, self._nextD)
            _getInterpolatedMatrix(M1, M2, tau + dtau, self._nextM)

            q = self._lin_term.compute(M1=self._currM, M2=self._nextM,
                                       Y=self._Y, lam=self._lam, mu=self._mu)

            P = self._quad_term.assemble(M=self._currM,
                                         lam=self._lam, mu=self._mu)

            # Make 'delta' less tight until a feasible solution has been found.
            feasible_ok = False
            for de in range(max_delta_expansions):
                if de > 0:
                    print("Retrying QP with increased delta ...")
                print("delta:", delta)

                A, lower, upper = self._constraints.assemble(
                    D1=self._currD, D2=self._nextD,
                    M1=self._currM, M2=self._nextM,
                    Y=self._Y, lam=self._lam, mu=self._mu,
                    delta=delta, eta1=eta1, eta2=eta2)

                if _SolveQP(P=P, q=q, A=A, l=lower, u=upper,
                            dY=self._dY, lam=self._lam_new, mu=self._mu_new,
                            iter_counter=de,
                            verbose=verbose > 0):
                    feasible_ok = True
                    break

                delta = min(delta * delta_expand, 0.5)
            if not feasible_ok:
                raise RuntimeError("QP solver cannot find a feasible solution")

            self._dY += self._Y             # dY <-- Y + dY
            Y_new = self._dY                # alias to Y + dY
            res = self._accuracy_crit.residual(D=self._nextD, M=self._nextM,
                                               Y=Y_new, lam=self._lam_new,
                                               mu=self._mu_new, delta=delta)
            if res > res_tol and attempt < max_retry_attempts:
                dtau = min(max(gamma1 * dtau, min_delta_tau), 1.0 - tau)
                attempt += 1
                print("Residual is too high res={:f}, repeating ...".format(res))
                continue

            # Update the solution.
            np.copyto(self._Y, Y_new)
            np.copyto(self._lam, self._lam_new)
            np.copyto(self._mu, self._mu_new)

            # Go to the next iteration. Try to shrink 'delta' a little bit.
            tau += dtau
            dtau = min(1.0 - tau, gamma2 * dtau)
            delta = max(delta * delta_shrink, self._base_delta)
            attempt = 0

        # Next outer iteration we'll start with modified 'delta'.
        self._delta = delta
        return self._Y, self._lam, self._mu

