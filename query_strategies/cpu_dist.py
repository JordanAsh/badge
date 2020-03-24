import numpy as np
from scipy.linalg.blas import dgemm, sgemm

def ext_arrs(A,B, precision="float64"):
    """
    Create extended version of arrays for matrix-multiplication based squared
    euclidean distance between two 2D arrays representing n-dimensional points.

    Parameters
    ----------
    A : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    B : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    precision : str, optional
        Selects the precision type for creating extended arrays.

    Returns
    -------
    A_ext : ndarray
        Extended version of A. The shape of A_ext is such that it has 3 times
        the number of columns in A. The arrangement is described below :
        The first block of dim columns has all 1s.
        The second block of dim columns has A.
        The third block of dim columns has squared elements of A.

    B_ext : ndarray
        Extended version of B. The shape of B_ext is such the the number of rows
        is 3 times the number of columns in B and the number of columns
        is same as the number of rows in B. The arrangement is described below :
        The first block of dim rows has squared B values, but transposed.
        The second block of dim rows has B values scaled by -2 and transposed.
        The third block of dim rows is all 1s.

    """

    nA,dim = A.shape
    A_ext = np.ones((nA,dim*3),dtype=precision)
    A_ext[:,dim:2*dim] = A
    A_ext[:,2*dim:] = A**2

    nB = B.shape[0]
    B_ext = np.ones((dim*3,nB),dtype=precision)
    B_ext[:dim] = (B**2).T
    B_ext[dim:2*dim] = -2.0*B.T
    return A_ext, B_ext

def auto_dtype(A, B):
    """
    Get promoted datatype for A and B combined.

    Parameters
    ----------
    A : ndarray
    B : ndarray

    Returns
    -------
    precision : dtype
        Datatype that would be used after appplying NumPy type promotion rules.
    If its not float dtype, e.g. int dtype, output is `float32` dtype.

    """

    # Datatype that would be used after appplying NumPy type promotion rules
    precision = np.result_type(A.dtype, B.dtype)

    # Cast to float32 dtype for dtypes that are not float
    if np.issubdtype(precision, float)==0:
        precision = np.float32

    return precision

def output_dtype(A,B, precision):
    """
    Get promoted datatype for A and B combined alongwith consideration
    for another input datatype.

    Parameters
    ----------
    A : ndarray

    B : ndarray

    precision : dtype
        This decides whether promoted datatype for A and B combined would be
        outputted or float32.

    Returns
    -------
    out_dtype : dtype
        Datatype that would be used after appplying NumPy type promotion rules.
    If its not float dtype, e.g. int dtype, output is `float32` dtype.

    """
    # Get output dtype
    if precision=="auto":
        out_dtype = auto_dtype(A, B)
    else:
        out_dtype = np.float32

    return out_dtype


def gemm_func(precision):
    """
    Get appropriate blas function

    Parameters
    ----------
    precision : dtype or str
        dtype or string signifying the datatype for which we need an appropriate
        blas function for matrix-multiplication

    Returns
    -------
    gemm_func : function
        Appropriate blas function

    """

    if (precision=="float64") | (precision==np.float64):
        gemm_func = dgemm
    else:
        gemm_func = sgemm
    return gemm_func


def dist_ext(A,B, matmul="dot", precision="auto"):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points using extended arrays based approach.
    For more info on rest of the input parameters and output, please refer to
    function 'dist'.

    """

    # Get output dtype
    out_dtype = output_dtype(A,B, precision)

    # Get extended arrays and then use matrix-multiplication to get distances
    A_ext, B_ext = ext_arrs(A,B, precision=out_dtype)

    if matmul=="dot":
        gemm_function = gemm_func(out_dtype)
        return gemm_function(alpha=1.0, a=A_ext, b=B_ext)
    elif matmul=="gemm":
        return A_ext.dot(B_ext)


def dist_accum(A,B, matmul="dot", precision="auto"):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points using accumulation based approach.
    For more info on rest of the input parameters and output, please refer to
    function 'dist'.

    """


    # Get matrix-multiplication between A and transposed B.
    # Then, accumulate squared row summations of A and B into it along the
    # appropriate axes of the matrix-multiplication result.
    out_dtype = output_dtype(A,B, precision)

    Af = A
    Bf = B
    if matmul=="dot":
        if np.issubdtype(A.dtype, int):
            Af = A.astype('float32')

        if np.issubdtype(B.dtype, int):
            Bf = B.astype('float32')

        out = Af.dot(-2*Bf.T)

    elif matmul=="gemm":
        # Get output dtype and appropriate gemm function for matrix-multiplication
        gemm_function = gemm_func(out_dtype)
        out = gemm_function(alpha=-2, a=Af, b=Bf,trans_b=True)

    out += np.einsum('ij,ij->i',Af,Af)[:,None]
    out += np.einsum('ij,ij->i',Bf,Bf)
    return out


def dist(A,B, matmul="dot", method="ext", precision="auto"):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points.

    Parameters
    ----------
    A : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    B : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    matmul : str, optional
        Selects the method for matrix-multiplication. It can be 'dot' or 'gemm'
        indicating the use of `numpy.dot` and `Scipy's` BLAS based wrapper
        functions- `sgemm/dgemm` respectively.
    method : str, optional
        Selects the method for sum-reductions needed to get those distances.
        It can be 'ext' or 'acc'.
    precision : str, optional
        Selects the precision type for computing distances. It can be 'auto' or
        'float32'.

    Returns
    -------
    out : ndarray
        Squared euclidean distance between two 2D arrays representing
        n-dimensional points. Basically there are two ways -
        First one involves creating extended versions of the input arrays and
        then using matrix-multiplication to get the final distances.
        Second one involves starting off with matrix-multiplication and then
        summing over row-wise squared summations of the input arrays into it
        along the rows and columns respectively.

    Example(s)
    -------
    Find the pairwise euclidean distances between three 2-D coordinates:

    >>> from from eucl_dist.cpu_dist import dist
    >>> coords = np.array([[2,3],[3,4],[2,5]])
    >>> dist(coords, coords)
    array([[ 0.,  2.,  4.],
           [ 2.,  0.,  2.],
           [ 4.,  2.,  0.]], dtype=float32)

    """

    if method=="ext":
        return dist_ext(A,B, matmul=matmul, precision=precision)
    elif method=="accum":
        return dist_accum(A,B, matmul=matmul, precision=precision)
    else:
        raise Exception("Invalid method")
