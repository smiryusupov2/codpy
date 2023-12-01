
def my_fun(x):
    coss = np.cos(2 * x * np.pi)
    if x.ndim == 1 : 
        coss = np.prod(coss, axis=0)
        ress = np.sum(x, axis=0)
    else : 
        coss = np.prod(coss, axis=1)
        ress = np.sum(x, axis=1)
    return ress+coss

def nabla_my_fun(x):
    sinss = np.cos(2 * x * np.pi)
    if x.ndim == 1 : 
        sinss = np.prod(sinss, axis=0)
        D = len(x)
        out = np.ones((D))
        def helper(d) : out[d] += 2.* sinss * np.pi*np.sin(2* x[d] * np.pi) / np.cos(2 * x[d] * np.pi)
        [helper(d) for d in range(0,D)]
    else:
        sinss = np.prod(sinss, axis=1)
        N = x.shape[0]
        D = x.shape[1]
        out = np.ones((N,D))
        def helper(d) : out[:,d] += 2.* sinss * np.pi*np.sin(2* x[:,d] * np.pi) / np.cos(2 * x[:,d] * np.pi)
        [helper(d) for d in range(0,D)]
    return out