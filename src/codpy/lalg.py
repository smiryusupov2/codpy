from include_all import *

def VanDerMonde(x: np.ndarray, orders) -> np.ndarray:
    """
    Compute the Vandermonde matrix for a given input array `x` and specified orders.

    The Vandermonde matrix is generated for each element of the input array `x`. 
    If `x` is a one-dimensional array, the function directly computes the Vandermonde matrix.
    If `x` is a multi-dimensional array, the function computes the Vandermonde matrix for each row.

    Args:
        x (array_like): Input array. Can be a one-dimensional or multi-dimensional array.
        orders (int or array_like): The powers to which the elements of `x` are raised. 
        If an integer, it specifies the maximum order. If an array, it contains the specific orders to use.
        **kwargs: Additional keyword arguments to be passed to the underlying Vandermonde computation function.

    Returns:
        ndarray: The computed Vandermonde matrix. If `x` is one-dimensional, the function returns a 2D array.
            If `x` is multi-dimensional, the function returns a 3D array where each 2D array along the first axis
            corresponds to the Vandermonde matrix of a row in `x`.

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
        return cd.tools.VanDerMonde(x,orders)
    out = np.array([cd.tools.VanDerMonde(x[n],orders) for n in range(0,x.shape[0])])
    return out

class lalg:
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
        Compute the Cholesky decomposition of a matrix.

        This method performs the Cholesky decomposition on a given matrix x. It optionally 
        uses a small value eps for numerical stability.

        Args:
            x: The matrix to decompose.
            eps (float, optional): A small value added for numerical stability. Default is 0.

        Returns:
            The Cholesky decomposition of the matrix.
        """
        return cd.lalg.cholesky(x,eps)
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