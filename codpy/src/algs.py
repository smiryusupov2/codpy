from core import *
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
import xarray
from codpy.utils.data_conversion import get_matrix, get_data
from codpy.utils.utils import lexicographical_permutation, format_32


class lalg:
    def prod(x,y): return cd.lalg.prod(get_matrix(x),get_matrix(y))
    def transpose(x): return cd.lalg.transpose(x)
    def scalar_product(x,y): return cd.lalg.scalar_product(x,y)
    def cholesky(x,eps = 0): return cd.lalg.cholesky(x,eps)
    def polar(x,eps = 0): return cd.lalg.polar(x,eps)
    def svd(x,eps = 0): return cd.lalg.svd(x,eps)

def map_invertion(map,type_in = None):
    if type_in is None: type_in = type(map)
    out = cast(data = map, type_in = type_in, type_out = Dict[int, Set[int]])
    out = cd.alg.map_invertion(out)
    return cast(out,type_out = type_in, type_in = Dict[int, Set[int]])


class alg:
    #set_codpy_kernel = kernel_setters.kernel_helper(kernel_setters.set_gaussian_kernel, 2,1e-8,map_setters.set_min_distance_map)
    #set_sampler_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_tensor_kernel, 3,1e-8 ,map_setters.set_standard_mean_map)

    # def set_sampler_kernel(polynomial_order:int = 2,regularization:float = 0,set_map = map_setters.set_unitcube_map):
    #     cd.kernel.set_polynomial_order(polynomial_order)
    #     cd.kernel.set_regularization(regularization)
    #     cd.set_kernel("tensornorm")
    #     map_setters.set_unitcube_map()
    #     if (polynomial_order > 0):
    #         kern = kernel.get_kernel_ptr()
    #         cd.set_kernel("linear_regressor")
    #         map_setters.set_unitcube_map()
    #         cd.kernel.pipe_kernel_ptr(kern)


    set_sampler_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_tensor_kernel, 0,1e-8 ,map_setters.set_standard_mean_map)
    set_isoprobas_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_tensor_kernel, 0,1e-8 ,map_setters.set_standard_mean_map)
    set_reordering_kernel = kernel_setters.kernel_helper(kernel_setters.set_matern_tensor_kernel, 2,1e-8,map_setters.set_standard_min_map)

    def reordering(x,y,**kwargs):
        reordering_format_switchDict = { pd.DataFrame: lambda x,y,**kwargs :  reordering_dataframe(x,y,**kwargs) }
        def reordering_dataframe(x,y,**kwargs):
            a,b,permutation = alg.reordering(get_matrix(x),get_matrix(y),**kwargs)
            x,y = pd.DataFrame(a,columns = x.columns, index = x.index),pd.DataFrame(b,columns = y.columns, index = y.index)
            return x,y,permutation
        def reordering_np(x,y,permut='source',**kwargs):
            D = op.Dnm(x,y,**kwargs)
            # test = D.trace().sum()
            lsap_fun = kwargs.get("lsap_fun",alg.lsap)
            permutation = lsap_fun(D)
            if permut != 'source': 
                y = y[map_invertion(permutation, type_in = List[int])]
            else: x = x[permutation]
            # D = op.Dnm(x,y,**kwargs)
            # test = D.trace().sum()
            return x,y,permutation
        type_debug = type(x)
        method = reordering_format_switchDict.get(type_debug,reordering_np)
        return method(x,y,**kwargs)

    def grid_projection(x,**kwargs):
        grid_projection_switchDict = { pd.DataFrame: lambda x,**kwargs :  grid_projection_dataframe(x),
                                        xarray.core.dataarray.DataArray: lambda x,**kwargs :  grid_projection_xarray(x) }
        def grid_projection_dataframe(x,**kwargs):
            out = cd.alg.grid_projection(x.values)
            out = pd.DataFrame(out,columns = x.columns, index = x.index)
            return out
        def grid_projection_xarray(x,**kwargs):
            index_string = kwargs.get("index","N")
            indexes = x[index_string]
            out = x.copy()
            for index in indexes:
                mat = x[index_string==int(index)].values
                mat = cd.alg.grid_projection(mat.T)
                out[index_string==int(index)] = mat.T
            return out
        def grid_projection_np(x,**kwargs):
            if x.ndim==3:
                shapes = x.shape
                x = format_32(x)
                out = cd.alg.grid_projection(x,**kwargs)
                out = format_23(out,shapes)
                return out
            else: return cd.alg.grid_projection(get_matrix(x),**kwargs)
        type_debug = type(x)
        method = grid_projection_switchDict.get(type_debug,lambda x : grid_projection_np(x,**kwargs))
        return method(x)


    def scipy_lsap(C):
        N = C.shape[0]
        D = np.min((N,C.shape[1]))
        permutation = linear_sum_assignment(C, maximize=False)
        out = np.array(range(0,D))
        for n in range(0,D):
            out[permutation[1][n]] = permutation[0][n]
        return out
    def lsap(x):
        return cd.alg.LSAP(x)
    def get_x(fx,**kwargs):
        fx = column_selector(fx,**kwargs)
        grid_projection = kwargs.get("grid_projection",False)
        seed = kwargs.get("seed",42)
        rand = kwargs.get("random_sample",np.random.random_sample)
        Nx = fx.shape[0]
        Dx = kwargs.get("Dx",fx.shape[1])
        Dfx = fx.shape[1]
        np.random.seed(seed)
        shape = [Nx,Dx]
        x = rand(shape)
        if grid_projection: x = alg.grid_projection(x)
        if (kwargs.get("reordering",True)): 
            kwargs['x'],kwargs['y'] = x,fx
            if Dx==Dfx:
                kwargs['distance'] = "norm22"
                x,y,permutation = alg.reordering(**kwargs, permut = 'source')
            else:
                def helper(**kwargs): return alg.encoder(**kwargs).params['fx']
                x = kwargs.get("encoder",helper)(**kwargs,permut='source')

        return x
    def get_y(x,**kwargs):
        Ny = kwargs.get("Ny",x.shape[0])
        if Ny < x.shape[0]: return kwargs.get("cluster_fun",alg.sharp_discrepancy)(x,**kwargs)
        return x
    def get_z(x,**kwargs):
        rand = kwargs.get("random_sample",np.random.random_sample)
        Nz = kwargs.get("Nz",x.shape[0])
        Dz = kwargs.get("Dx",x.shape[1])
        shape = [Nz,Dz]
        z = rand(shape)
        return z

    def get_xyz(fx,**kwargs):
        kwargs['fx'] = fx
        kwargs['x'] = kwargs.get("get_x",alg.get_x)(**kwargs)
        kwargs['y'] = kwargs.get("get_y",alg.get_y)(**kwargs)
        z = kwargs.get("get_z",alg.get_z)(**kwargs)

        return kwargs['x'],kwargs['y'],z
        

    def sampler(fx, **kwargs):
        Nz,x,y,z=kwargs.get("Nz",None),kwargs.get("x",None),kwargs.get("y",None),kwargs.get("z",None)
        if isinstance(fx,pd.DataFrame): 
            out,x,y,z = alg.sampler(get_matrix(fx),**kwargs)
            columns = [str(n) for n in range(x.shape[1])]
            return pd.DataFrame(out,columns = fx.columns),pd.DataFrame(x,columns = columns),pd.DataFrame(y,columns = columns),pd.DataFrame(z,columns = columns)
        if x is None or y is None:
            getxyz = kwargs.get('getxyz',alg.get_xyz)
            kwargs['x'],kwargs['y'],kwargs['z'] = getxyz(fx,**kwargs)
            return alg.sampler(fx=fx,**kwargs)
        if z is None: return None,x,y,None
        if kwargs.get('nablainv',False):
            fy = fx[:x.shape[0]]
            z=kwargs['z']
            kwargs['z'] = x
            fz = op.nabla_inv(fz = fy.reshape(fy.shape + (1,)),**kwargs)
            kwargs['z'] = z
            fz = op.nabla(fx=fz,**kwargs)
            fz = fz.squeeze()
        else:
            fz= op.projection(fx=fx,**kwargs)
        return fz,x,y,z



    def iso_probas_projection(x, fx, probas, fun_permutation = lexicographical_permutation, set_codpy_kernel = set_isoprobas_kernel, rescale = True,**kwargs):
        # print('######','iso_probas_projection','######')
        Nx,Dx = np.shape(x)
        Nx,Df = np.shape(fx)
        Ny = len(probas)
        fy,y,permutation = fun_permutation(fx,x)
        out = np.concatenate((y,fy), axis = 1)
        quantile = np.array(np.arange(start = .5/Nx,stop = 1.,step = 1./Nx)).reshape(Nx,1)
        out = op.projection(x = quantile,y = probas,z = probas, fx = out, set_codpy_kernel = set_codpy_kernel, rescale = rescale)
        return out[:,0:Dx],out[:,Dx:],permutation

    def Pi(x, z, fz=[], set_codpy_kernel = set_sampler_kernel, rescale = True,nmax=10,**kwargs):
        # print('######','Pi','######')
        if (set_codpy_kernel): set_codpy_kernel()
        if (rescale): kernel.rescale(x,z)
        out = cd.alg.Pi(x = x,y = x,z = z, fz = fz,nmax = nmax)
        return out
    def get_normals(N,D, **kwargs):
        kernel.init(**kwargs)
        out = cd.alg.get_normals(N = N,D = D,nmax = nmax)
        return out
    def match(x, **kwargs):
        x = column_selector(x,**kwargs)
        Ny = kwargs.get('Ny',x.shape[0])
        if Ny >= x.shape[0]: return x
        match_format_switchDict = { pd.DataFrame: lambda x,**kwargs :  match_dataframe(x,**kwargs) }
        def match_dataframe(x, **kwargs):
            out=alg.match(get_matrix(x),**kwargs)
            return pd.DataFrame(out,columns = x.columns)

        def debug_fun(x,**kwargs):
            if 'sharp_discrepancy:xmax' in kwargs: x= random_select_interface(x,xmaxlabel = 'sharp_discrepancy:xmax', seedlabel = 'sharp_discrepancy:seed',**kwargs)
            kernel.init(x=x,**kwargs)
            out = cd.alg.match(get_matrix(x),Ny)
            return out
        type_debug = type(x)
        method = match_format_switchDict.get(type_debug,debug_fun)
        out = method(x,**kwargs)
        return out
    def kmeans(x, **kwargs):
        x = column_selector(x,**kwargs)
        Ny = kwargs.get('Ny',x.shape[0])
        if Ny >= x.shape[0]: return x
        return KMeans(n_clusters=Ny,
            init=kwargs.get('init','k-means++'), 
            n_init=Ny, 
            max_iter=kwargs.get('max_iter',300), 
            random_state=kwargs.get('random_state',42)).fit(x).cluster_centers_
    def MiniBatchkmeans(x, **kwargs):
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

    def sharp_discrepancy(x, **kwargs):
        itermax = int(kwargs.get('sharp_discrepancy:itermax',10))
        Ny = kwargs.get("Ny",None)
        if Ny is None: 
            if kwargs.get('y',None) is None:return x
            return cd.alg.sharp_discrepancy(x,kwargs['y'],itermax)
        if Ny>=x.shape[0]: return x
        sharp_discrepancy_format_switchDict = { pd.DataFrame: lambda x,**kwargs :  sharp_discrepancy_dataframe(x,**kwargs) }
        def sharp_discrepancy_dataframe(x, **kwargs):
            out=alg.sharp_discrepancy(get_matrix(x),**kwargs)
            return pd.DataFrame(out,columns = x.columns)


        def debug_fun(x,**kwargs):
            out = alg.match(x,**kwargs)
            out = cd.alg.sharp_discrepancy(x,out,itermax)
            return out
        type_debug = type(x)
        method = sharp_discrepancy_format_switchDict.get(type_debug,debug_fun)
        out = method(x,**kwargs)
        return out
    def encoder_cost(x,y,**kwargs):
        kernel.init(x=x,**kwargs)
        Delta = op.nablaT_nabla(x = x,y = x,z=x)
        yyT = lalg.prod(get_matrix(y),get_matrix(y).T)
        return lalg.scalar_product(Delta,yyT)
    
    class encoder:
        params = {}
        def __init__(self,y,x=None,permut='source',**kwargs):
            y,x = column_selector(y,**kwargs),column_selector(x,**kwargs)
            if x is None:
                x = y
                if kwargs.get("Dx",x.shape[1]) != x.shape[1]:
                    x = alg.match(y.T,Ny=kwargs['Dx'],**kwargs).T
                else:return
            self.params= kwargs.copy()
            kernel.init(x=x,**self.params)
            permutation = cd.alg.encoder(get_matrix(x),get_matrix(y))
            self.params['permutation'] = permutation
            if permut == 'source': 
                permut = map_invertion(permutation,type_in = List[int])
                if isinstance(x,pd.DataFrame): x= pd.DataFrame(x.iloc[permut], columns = x.columns)
                else: x=x[permut]
                self.params['x'],self.params['y'],self.params['fx']= y,y,x
            else:
                if isinstance(y,pd.DataFrame): y  = pd.DataFrame(y.iloc[permutation], columns = y.columns)
                else: y=y[permutation]
                self.params['x'],self.params['y'],self.params['fx']= y,y,x

        def __call__(self, **kwargs):
            self.params['z'] = kwargs['z']
            return op.projection(**self.params)

    class decoder:
        encoder = None
        params = None
        def __init__(self,encoder):
            self.encoder = encoder
            self.params= encoder.params.copy()
            self.params['x'],self.params['y'],self.params['fx']= self.params['fx'],self.params['fx'],self.params['x']

        def __call__(self, **kwargs):
            self.params['z'] = kwargs['z']
            return op.projection(**self.params)



########################################### Partial Differential tools
    def taylor_expansion(x, y, z, fx, nabla = op.nabla, hessian = op.hessian, **kwargs):
        # print('######','distance_labelling','######')
        x,y,z,fx = get_matrix(x),get_matrix(y),get_matrix(z),get_matrix(fx)
        xo,yo,zo,fxo = x,y,z,fx
        indices = kwargs.get("indices",[])
        if len(indices) != z.shape[0] :
            indices = alg.distance_labelling(x,z,axis=0,**kwargs)
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
    

def VanDerMonde(x,orders,**kwargs):
    return cd.tools.VanDerMonde(x,orders)