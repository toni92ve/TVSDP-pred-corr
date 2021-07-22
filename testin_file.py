import osqp
import numpy as np
from scipy import sparse

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
        idx = np.arange(nvars, dtype=np.int).reshape((T+N), rank)

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
                 lam: np.ndarray, mu: np.ndarray) -> sparse.csc_matrix:
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

    

_quad_term = _QuadraticTerm(2, 3, 2)
print(_quad_term.__init__.tmp)