import pandas as pd
import numpy as np
from core import kernel, distance_labelling, diffops
from data_conversion import get_matrix, get_data

########################################### Partial Differential tools #########################################

def CrankNicolson(A, dt = 0., u0=[], **kwargs):
    """
    Implement the Crank-Nicolson discretization scheme for numerical solution of partial differential equations.

    The Crank-Nicolson method is a time-stepping scheme that averages the explicit (forward Euler) and implicit 
    (backward Euler) methods. This method is known for its stability and accuracy, especially for stiff equations.
    
    Args:
        A (array_like): The coefficient matrix used in the differential equation.
        dt (float, optional): The time step size. Defaults to 0.
        u0 (array_like, optional): The initial condition (state at time t=0). Defaults to an empty list.
        **kwargs: Arbitrary keyword arguments, which may include:
            theta (float, optional): A parameter that balances between explicit and implicit methods. 
                                    Default is 0.5, which gives the Crank-Nicolson scheme.
                                    Values range from 0 (fully explicit) to 1 (fully implicit).

    Returns:
        array_like: The numerical solution of the differential equation at the next time step.

    Example:
        Coefficient matrix A for the differential equation
        
        >>> A = np.array([...])

        Initial condition
        
        >>> u0 = np.array([...])

        Time step size
        
        >>> dt = 0.1

        Compute the next time step solution
        
        >>> u_next = CrankNicolson(A, dt, u0)
    """
    
    kernel.init(**kwargs)
    theta = kwargs.get('theta',0.5)
    return cd.alg.CrankNicolson(get_matrix(A),dt,get_matrix(u0),theta)

def taylor_expansion(x, y, z, fx, nabla = diffops.nabla, hessian = diffops.hessian, **kwargs):
    """
    Perform a Taylor series expansion to approximate the function values at new points.

    This function approximates the values of a given function at new points 'z' based on its values
    and derivatives at points 'x'. It supports first and second order (gradient and Hessian) expansions.

    Args:
        x (array_like): Points where the function values and derivatives are known.
        y (array_like): Additional parameter, typically used in the computation of derivatives.
        z (array_like): Points where the function values are to be approximated.
        fx (array_like): Known function values at points 'x'.
        nabla (function, optional): Function to compute the gradient. Default is diffops.nabla.
        hessian (function, optional): Function to compute the Hessian matrix. Default is diffops.hessian.
        **kwargs: Arbitrary keyword arguments, which may include:
            indices (list, optional): Precomputed indices for mapping 'x' to 'z'. 
                                    If not provided, they will be computed.
            taylor_order (int, optional): The order of the Taylor expansion (1 or 2). Default is 1.
            taylor_explanation (dict, optional): A dictionary to store intermediate results for analysis.

    Returns:
        array_like: Approximated function values at points 'z' using Taylor expansion.

    Example:
        Define points 'x', 'y', 'z', and function values 'fx'
        
        >>> x = np.array([...])
        >>> y = np.array([...])
        >>> z = np.array([...])
        >>> fx = np.array([...])

        Perform Taylor expansion
        
        >>> fz_approx = taylor_expansion(x, y, z, fx)
    """
    # print('######','distance_labelling','######')
    x,y,z,fx = get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx)
    xo, _, _,fxo = x,y,z,fx
    indices = kwargs.get("indices",[])
    if len(indices) != z.shape[0] :
        indices = distance_labelling(**{**kwargs,**{'x':x,'axis':0,'y':z}})
    xo= x[indices]
    fxo= fx[indices]
    deltax = get_data(z - xo)
    taylor_order = int(kwargs.get("taylor_order",1))
    results = kwargs.get("taylor_explanation",None)
    if taylor_order >=1:
        grad = nabla(x=x, y=y, z=x, fx=fx, **kwargs)
        if grad.ndim >= 3:  grad= get_matrix(np.squeeze(grad))
        if grad.ndim == 1:  grad= get_matrix(grad).T
        if len(indices) : grad = grad[indices]
        # print("grad:",grad[0])
        product_ = np.reshape([np.dot(grad[n],deltax[n]) for n in range(grad.shape[0])],(len(grad),1))
        f_z = fxo  + product_
        if isinstance(fx,pd.DataFrame): f_z = pd.DataFrame(f_z, columns = fx.columns)
        if results is not None:
            results["indices"] = indices
            results["delta"] = deltax
            results["nabla"] = grad

    if taylor_order >=2:
        hess = hessian(x=x, y=x, z=x, fx=fx, **kwargs)
        if hess.ndim>3:hess = hess.reshape(hess.shape[0],hess.shape[1],hess.shape[2])
        if len(indices) : hess = hess[indices]
        deltax = np.reshape([np.outer(deltax[n,:],deltax[n,:]) for n in range(deltax.shape[0])], (hess.shape[0],hess.shape[1],hess.shape[2]))
        quadratic_form = np.reshape([ np.trace(hess[n].T@deltax[n])  for n in range(hess.shape[0])], (hess.shape[0],1))
        f_z += 0.5*quadratic_form
        if results is not None:
            results["quadratic"] = deltax
            results["hessian"] = hess
    return f_z