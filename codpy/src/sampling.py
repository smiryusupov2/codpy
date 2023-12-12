import numpy as np
import pandas as pd
import codpydll
import codpypyd as cd
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
from codpy.src.core import op, kernel, kernel_setters, map_setters
from codpy.utils.data_conversion import get_matrix
from codpy.utils.selection import column_selector
from codpy.utils.random import random_select_interface

def kernel_density_estimator(**kwargs):
    """
    Estimate the kernel density of a distribution.

    This function implements a kernel density estimator (KDE), a non-parametric method to estimate
    the probability density function of a random variable. It evaluates the density estimate based on
    two input distributions using a specified kernel function.

   Args:
        **kwargs: Arbitrary keyword arguments, including:
            x (array-like): The first input distribution for which the density estimate is to be computed.
            y (array-like): The second input distribution used in the density estimation process.
            kernel (optional): The kernel function to be used for density estimation. This can be specified
                            as part of the kwargs. If not specified, a default kernel is used.

    Returns:
        array-like: The estimated density values based on the kernel density estimation.

    Example:
        Two sample distributions
        
        >>> x = np.array([...])
        >>> y = np.array([...])

        Compute the kernel density estimation
        
        >>> density = kernel_density_estimator(x=x, y=y)
    """
    # given two distribution x,y evaluate \sum_i K(x^i,y^j)
    #
    x,y = kwargs["x"],kwargs["y"]
    kernel.init(**{**kwargs,**{'z':None}})
    return cd.tools.density_estimator(x,y)

def _get_nadaraya_watson_param(bandwidth=1.):
    kwargs = {
        'rescale_kernel':{'max': 2000, 'seed':42},
        # 'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_invquadratictensor_kernel, 0,1e-8 ,map_setters.set_scale_factor_helper(bandwidth=bandwidth)),
        'set_codpy_kernel' : kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 0,1e-8 ,map_setters.set_scale_factor_helper(bandwidth=bandwidth)), 
        'rescale': False,
        'grid_projection': False}
    return kwargs

def kernel_conditional_density_estimator(x_vals, y_vals, x_data, y_data, kwargs = _get_nadaraya_watson_param()):
    """
    Estimate the conditional density of 'y' given 'x' using the Nadaraya-Watson estimator.

    This function calculates the conditional density of values in 'y_vals' given the values in 'x_vals',
    based on a joint distribution ('x_data', 'y_data'). It uses KDE method for the estimation, with the
    kernel specified in 'kwargs'.

    Args:
        x_vals (array-like): Values of 'x' for which the conditional density of 'y' is estimated.
        y_vals (array-like): Values of 'y' for which the density is to be estimated conditionally on 'x_vals'.
        x_data (array-like): Observed data for 'x' in the joint distribution with 'y'.
        y_data (array-like): Observed data for 'y' in the joint distribution with 'x'.
        kwargs (dict, optional): Parameters for the kernel function used in the Nadaraya-Watson estimator.
                                If not provided, default parameters are used.

    Returns:
        array-like: The estimated conditional density of 'y' given 'x'.

    Example:
        Define joint distribution data for 'x' and 'y'
        
        >>> x_data = np.array([...])
        >>> y_data = np.array([...])

        Values for conditional density estimation
        
        >>> x_vals = np.array([...])
        >>> y_vals = np.array([...])

        Compute the conditional density
        
        >>> conditional_density = kernel_conditional_density_estimator(x_vals, y_vals, x_data, y_data)
    """
    # given a joint distribution (x_data, y_data), return the density y_val | x_val using the Nadaraya-Watson estimate
    # The kernel is input in kwargs
    marginal_x =op.Knm(**{**kwargs, **{'x':x_data,'y':x_vals}})
    marginal_y = op.Knm(**{**kwargs, **{'x':y_data,'y':y_vals}})
    joint_weights = marginal_x * marginal_y
    sum_x = np.sum(marginal_x, axis = 0) 
    joint_weights = np.sum(joint_weights,axis=0) / sum_x
    return joint_weights

def rejection_sampling(proposed_sample, probas,acceptance_ratio = 0.):
    """
    Perform rejection sampling on a set of proposed samples.

    This function implements the rejection sampling algorithm, a technique in Monte Carlo methods.
    It evaluates each proposed sample against an acceptance criterion based on the sample's probability and 
    an acceptance ratio. Samples are accepted with a probability proportional to their probability in the 
    target distribution.

    Args:
        proposed_sample (array-like): An array of proposed samples to be evaluated.
        probas (array-like): An array of probabilities corresponding to each proposed sample.
        acceptance_ratio (float, optional): A threshold ratio for accepting samples. This value can be used 
                                            to control the acceptance rate. Default is 0.

    Returns:
        list: A list of samples that are accepted based on the rejection sampling criterion.

    Example:
        Proposed samples and their probabilities
        
        >>> proposed_samples = np.array([...])
        >>> probabilities = np.array([...])

        Perform rejection sampling
        
        >>> accepted_samples = rejection_sampling(proposed_samples, probabilities)

        Note:
        The function assumes that the proposed samples and their probabilities are of the same length.
    """
    samples= []
    for n in range(proposed_sample.shape[0]):
        acceptance_ratio = max(acceptance_ratio,probas[n])  
        if np.random.uniform(0, acceptance_ratio) < probas[n]:
            samples.append(proposed_sample[n])
    return samples

def get_normals(N,D, **kwargs):
        kernel.init(**kwargs)
        nmax = kwargs.get('nmax',10)
        out = cd.alg.get_normals(N = N,D = D,nmax = nmax)
        # mean,var = np.mean(out,axis=0),np.var(out,axis=0)
        return out

def get_uniform(N,D, **kwargs):
    """
    Generate uniformly distributed random samples from normally distributed samples.

    This function first generates random samples from a normal distribution and then 
    transforms them into a uniform distribution using the error function (erf). It's 
    based on the probability integral transform where the Gaussian CDF is used for this conversion.

    Args:
        N (int): The number of samples to generate.
        D (int): The dimensionality of each sample.
        **kwargs: Additional keyword arguments to be passed to the normal sample generator function.

    Returns:
        ndarray: An array of shape (N, D) containing uniformly distributed random samples.

    Example:
        Generate 100 samples with 2 dimensions
        
        >>> uniform_samples = get_uniform(100, 2)
    """
    out = get_normals(N,D, **kwargs)
    out = np.vectorize(math.erf)(out)/2. + 0.5
    return out

def get_uniform_like(kwargs):return get_uniform(**{**kwargs,**{'N':kwargs['x'].shape[0],'D':kwargs['x'].shape[1]}})
def get_normals_like(kwargs):return get_normals(**{**kwargs,**{'N':kwargs['x'].shape[0],'D':kwargs['x'].shape[1]}})
def get_random_normals_like(kwargs):return np.random.normal(size = kwargs['x'].shape)
def get_random_uniform_like(kwargs):return np.random.uniform(size = kwargs['x'].shape)

def match(x, **kwargs):
    kernel.init(**kwargs)
    x = column_selector(x,**kwargs)
    Ny = kwargs.get('Ny',x.shape[0])
    if Ny >= x.shape[0]: return x
    match_format_switchDict = { pd.DataFrame: lambda x,**kwargs :  match_dataframe(x,**kwargs) }
    def match_dataframe(x, **kwargs):
        out = match(get_matrix(x),**kwargs)
        return pd.DataFrame(out,columns = x.columns)

    def debug_fun(x,**kwargs):
        if 'sharp_discrepancy:xmax' in kwargs: x = random_select_interface(xmaxlabel = 'sharp_discrepancy:xmax', seedlabel = 'sharp_discrepancy:seed',**{**kwargs,**{'x':x}})
        kernel.init(x = x, **kwargs)
        out = cd.alg.match(get_matrix(x),Ny)
        return out
    type_debug = type(x)
    method = match_format_switchDict.get(type_debug,debug_fun)
    out = method(x,**kwargs)
    return out

def kmeans(x, **kwargs):
    """
    Perform K-means clustering on a dataset.

    This function applies the K-means clustering algorithm to partition the input data into 'Ny' clusters.
    It uses the KMeans implementation from Scikit-Learn. The number of clusters, initialization method, and
    other parameters of the KMeans algorithm can be specified via keyword arguments.

    Args:
        x (array-like or DataFrame): Input data for clustering. Should be in a suitable format for clustering (e.g., numerical).
        **kwargs: Arbitrary keyword arguments, which may include:
            Ny (int, optional): The number of clusters to form. Default is the number of rows in 'x'.
            init (str, optional): Method for initialization ('k-means++', 'random', or an ndarray). Default is 'k-means++'.
            max_iter (int, optional): Maximum number of iterations of the k-means algorithm. Default is 300.
            random_state (int, optional): Determines random number generation for centroid initialization. 
            Use an int for reproducibility. Default is 42.

    Returns:
        array-like or DataFrame: Cluster centers if 'Ny' is less than the number of rows in 'x', otherwise returns 'x' as is.

    Example:
        Example with NumPy array
        
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        >>> clusters = kmeans(X, Ny=3)

        Example with pandas DataFrame
        
        >>> import pandas as pd
        >>> df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        >>> clusters = kmeans(df, Ny=3)

        Note:
        This function requires Scikit-Learn's KMeans implementation. Ensure that sklearn.cluster.KMeans is imported.
    """
    x = column_selector(x,**kwargs)
    Ny = kwargs.get('Ny',x.shape[0])
    if Ny >= x.shape[0]: return x
    return KMeans(n_clusters=Ny,
        init=kwargs.get('init','k-means++'),
        n_init=Ny,
        max_iter=kwargs.get('max_iter',300),
        random_state=kwargs.get('random_state',42)).fit(x)

def MiniBatchkmeans(x, **kwargs):
    """
    Perform mini-batch K-means clustering on a dataset.

    This function applies the mini-batch K-means clustering algorithm, an efficient variant of 
    the standard K-means algorithm, to partition the input data into 'Ny' clusters. It is particularly
    useful for large datasets. The function uses the MiniBatchKMeans implementation from Scikit-Learn. 
    The number of clusters, batch size, and other parameters of the MiniBatchKMeans algorithm can be 
    specified via keyword arguments.

    Args:
        x (array-like or DataFrame): Input data for clustering. Should be in a format suitable for clustering (e.g., numerical).
        **kwargs: Arbitrary keyword arguments, which may include:
            Ny (int, optional): The number of clusters to form. Default is the number of rows in 'x'.
            max_iter (int, optional): Maximum number of iterations of the mini-batch k-means algorithm. Default is 300.
            random_state (int, optional): Determines random number generation for centroid initialization. Use an int for reproducibility. Default is 42.
            batch_size (int, optional): Size of the mini-batches. Default is 4352 (256*17).

    Returns:
        array-like or DataFrame: Cluster centers if 'Ny' is less than the number of rows in 'x', otherwise returns 'x' as is.

    Example:
        Example with NumPy array
        
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=10000, centers=5, n_features=2, random_state=42)
        >>> clusters = MiniBatchkmeans(X, Ny=5)

        Example with pandas DataFrame
        
        >>> import pandas as pd
        >>> df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        >>> clusters = MiniBatchkmeans(df, Ny=5)

        Note:
        This function requires Scikit-Learn's MiniBatchKMeans implementation. Ensure that sklearn.cluster.MiniBatchKMeans is imported.
    """
    max_iter,random_state,batch_size=kwargs.get('max_iter',300),kwargs.get('random_state',42),256*17
    x = column_selector(x,**kwargs)
    Ny = kwargs.get('Ny',x.shape[0])
    if Ny >= x.shape[0]: return x
    return MiniBatchKMeans(n_clusters=Ny,
        init="k-means++",
        batch_size = batch_size,
        verbose = 1,
        max_iter=max_iter,
        random_state=random_state).fit(x).cluster_centers_

def sharp_discrepancy(**kwargs):
    x = kwargs['x']
    kernel.init(**kwargs)
    itermax = int(kwargs.get('sharp_discrepancy:itermax',10))
    Ny = kwargs.get("Ny",None)
    if Ny is None:
        if kwargs.get('y',None) is None:return x
        return cd.alg.sharp_discrepancy(x,kwargs['y'],itermax)
    if Ny>=x.shape[0]: return x
    sharp_discrepancy_format_switchDict = { pd.DataFrame: lambda x,**kwargs :  sharp_discrepancy_dataframe(x,**kwargs) }
    def sharp_discrepancy_dataframe(x, **kwargs):
        out = sharp_discrepancy(**{**kwargs,**{'x':get_matrix(x)}})
        return pd.DataFrame(out,columns = x.columns)

    def debug_fun(**kwargs):
        Ny = kwargs.get('Ny',x.shape[0])
        if Ny >= x.shape[0] : return x
        out = cd.alg.sharp_discrepancy(x,Ny,itermax)
        return out
    type_debug = type(x)
    method = sharp_discrepancy_format_switchDict.get(type_debug,debug_fun)
    out = method(**kwargs)
    return out