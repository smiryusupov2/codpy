from codpy.core import get_matrix
from codpydll import *
import numpy as np

def VanDerMonde(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of the Vandermonde system for a given input array `x` and specified orders.
    Useful for Lagrange polynomial interpolations to design higher order numerical schemes

    The Vandermonde system consists in solving $A x = y$, where $A=(x_i)^j$. 
    If `x` is a one-dimensional array, the function directly computes the Vandermonde matrix.
    If `x` is a multi-dimensional array, the function computes the Vandermonde matrix for each row.

    Args:
        x (array_like): Input one-dimensional matrix, optionally a facility is provided to matrix.
        y (array_like): Input matrix. 

    Returns:
        ndarray: 
        $A^{-1} y$, if x is one-dimensional. $[A_i^{-1} y]$, for $i=1,\cdots,N$, $N$ being the number of row of $x$ , if x is a matrix.

    Examples:
        For a one-dimensional input:
        
        >>> vdm = VanDerMonde(np.array([1, 2, 3]), 3)
        >>> print(vdm)
        # Output:
        [[1 1 1]
         [1 2 4]
         [1 3 9]]

        For a two-dimensional input:
        
        >>> vdm = VanDerMonde(np.array([[1, 2], [3, 4]]), 3)
        >>> print(vdm)
        # Output:
        [[[1 1 1]
          [1 2 4]]
         [[1 1 1]
          [1 4 16]]]
    """
    x = get_matrix(x)
    if x.ndim==1: 
        return cd.tools.VanDerMonde(x,y)
    out = np.array([cd.tools.VanDerMonde(x[n],y) for n in range(0,x.shape[0])])
    return out

class lalg:
    """
    A namespace to grant access to linear algebra tools, using Intel(R) Math Kernel Library (MKL) as backend.
    MKL is a parallelized, highly optimized library for linear algebra and other math tools.
    """
    def prod(x,y): 
        """
        Compute the product of two matrices.

        This method computes the matrix product of two input matrices, x and y.

        Args:
            x: The first matrix.
            y: The second matrix.

        Returns:
            The product of the two matrices.
        """
        return cd.lalg.prod(get_matrix(x),get_matrix(y))
    def transpose(x): 
        """
        Transpose a matrix.

        This method computes the transpose of a given matrix x.

        Args:
            x: The matrix to transpose.

        Returns:
            The transposed matrix.
        """
        return cd.lalg.transpose(x)
    def scalar_product(x,y): 
        """
        Compute the scalar product of two vectors.

        This method calculates the scalar (dot) product of two vectors, x and y.

        Args:
            x: The first vector.
            y: The second vector.

        Returns:
            The scalar product of the two vectors.
        """
        return cd.lalg.scalar_product(x,y)
    def cholesky(x,eps = 0): 
        """
        Compute the inverse of a square symetrical matrix using Cholesky decomposition.

        This method performs the Cholesky decomposition on a given matrix x. It optionally 
        uses a small value eps for numerical stability.

        Args:
            x: The matrix to decompose.
            eps (float, optional): A small value added for numerical stability. Default is 0.

        Returns:
            The inverse of the regularized matrix.
        """
        return cd.lalg.cholesky(x,eps)
    def inverse(x): 
        """
        Compute the inverse of a square matrix using LU decomposition.

        This method performs the LU decomposition on a given matrix x to compute itrs inverse.

        Args:
            x: The matrix to decompose.

        Returns:
            The inverse of the matrix.
        """
        return cd.lalg.inverse(x)
    def polar(x,eps = 0): 
        """
        Compute the polar decomposition of a matrix.

        This method calculates the polar decomposition of a given matrix x. It optionally 
        uses a small value eps for numerical stability.

        Args:
            x: The matrix to decompose.
            eps (float, optional): A small value added for numerical stability. Default is 0.

        Returns:
            The polar decomposition of the matrix.
        """
        return cd.lalg.polar(x, eps)
    def svd(x,eps = 0): 
        """
        Compute the singular value decomposition (SVD) of a matrix.

        This method performs the singular value decomposition on a given matrix x. It optionally 
        uses a small value eps for numerical stability.

        Args:
            x: The matrix to decompose.
            eps (float, optional): A small value added for numerical stability. Default is 0.

        Returns:
        The singular value decomposition of the matrix.
        """
        return cd.lalg.svd(x, eps)
    
    def lstsq(A,b=[],eps = 1e-8):
        """
        Compute the inverse of a rectangular matrix using the least square method.
        This method performs a safe matrix invertion, computing $(A^T A + \epsilon I)^{-1}A^T b$, the inversion relying on a fast cholesky decomposition. 
        For semi-definite matrix and without regularization $\epsilon=0$, the cholesky decomposition might fail, 
        in which case the algorithm rely on SVD decomposition to perform the invertion, a safer, but computationally intensive procedure.

        Args:
            A: The matrix to invert.
            b: an optional matrix as second member.
            eps (float, optional): A small value added for numerical stability. Default is 1e-9 to cope with numerical error.

        Returns:
        $(A^T A + \epsilon I)^{-1}A^T b$ .
        """
        out= cd.lalg.lstsq(get_matrix(A),get_matrix(b),eps)
        return out
    
    def SelfAdjointEigenDecomposition(A): 
        """
        Compute the self adjoint Eigen decomposition $A = U D U^T$.
        """
        return cd.lalg.SelfAdjointEigenDecomposition(A)

    def fix_nonpositive_semidefinite(x,eps=1e-8): 
        eigenvector,eigenvalues = lalg.SelfAdjointEigenDecomposition(get_matrix(x))
        eigenvalues = np.array([max(e,eps) for e in eigenvalues])
        return lalg.prod(eigenvector*eigenvalues,eigenvector.T)
