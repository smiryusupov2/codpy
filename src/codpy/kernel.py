import codpy.core as core
from codpydll import *
from codpy.algs import alg
import numpy as np
import pandas as pd
from scipy.special import softmax
from codpy.permutation import lsap
from codpy.data_processing import hot_encoder
from codpy.selection import get_matching_cols
from codpy.lalg import lalg as lalg
from codpy.utils import gather,fill,softmaxindices,softminindices


class Kernel:
    params = {}
    def __init__(self,max_pool=1000,max_nystrom=1000,reg=1e-9,order = None,dim=1,**kwargs):
        self.dim = dim
        self.order = order
        self.reg = reg
        self.max_pool = int(max_pool)
        self.max_nystrom = int(max_nystrom)
        self.set_kernel = kwargs.get("set_kernel",None)
        if self.set_kernel is None:
            # self.set_kernel = lambda :core.kernel_helper2("gaussian","standardmin",0,self.reg)
            self.set_kernel = lambda :core.kernel_helper2("tensornorm","unitcube",0,self.reg)
            # self.set_kernel = lambda :core.kernel_helper2("maternnorm","standardmean",0,self.reg)
        self.x = None
        if kwargs.get("x",None) is not None or kwargs.get("fx",None) is not None: self.set(**kwargs)

    def get_order(self,**kwargs):
        if not hasattr(self, 'order') :self.order= None
        return self.order

    def set_polynomial_regressor(self,x=None,fx=None,**kwargs):
        if x is None or fx is None or self.get_order() is None:
            self.polyvariables,self.polynomial_kernel,self.polynomial_values = None,None,None
            return
        order = self.get_order()
        if order is not None and fx is not None and x is not None:
            self.polyvariables = PolynomialFeatures(order).fit_transform(x)
            self.polynomial_kernel = linear_model.LinearRegression().fit(self.polyvariables,fx)
            self.polynomial_values = self.polynomial_kernel.predict(self.polyvariables)
            self.set_theta(None)
    def get_polynomial_values(self,**kwargs):
        if self.get_order() is None: return None
        if not hasattr(self, 'polynomial_values') or self.polynomial_values is None:
            self.set_polynomial_regressor(self.get_x(),self.get_fx())
        return self.polynomial_values
    def get_polyvariables(self,**kwargs):
        if self.get_order() is None: return None
        if not hasattr(self, 'polyvariables') or self.polyvariables is None:
            self.set_polynomial_regressor(self.get_x(),self.get_fx())
        return self.polyvariables
    def get_polynomial_kernel(self,**kwargs):
        if self.get_order() is None: return None
        if not hasattr(self, 'polynomial_kernel') or self.polynomial_kernel is None:
            self.set_polynomial_regressor(self.get_x(),self.get_fx())
        return self.polynomial_kernel

    def get_polynomial_regressor(self,z,x=None,fx=None,**kwargs):
        if self.get_order() is None : return None
        if x is None:polyvariables = self.get_polyvariables()
        else: polyvariables = PolynomialFeatures(self.order).fit_transform(x)
        if fx is None:polynomial_kernel = self.get_polynomial_kernel()
        else: polynomial_kernel = linear_model.LinearRegression().fit(polyvariables,fx)
        z_polyvariables = PolynomialFeatures(self.order).fit_transform(z)
        if polynomial_kernel is not None: return polynomial_kernel.predict(z_polyvariables)
        return None

    def Knm(self,x,y,fy = [],**kwargs):
        self.set_kernel_ptr()
        return core.op.Knm(x=x,y=y,fy=fy)

    def get_knm_inv(self,**kwargs):
        if not hasattr(self, 'knm_inv') : self.knm_inv = None
        if self.knm_inv is None:
            epsilon = kwargs.get("epsilon",self.reg)
            epsilon_delta = kwargs.get("epsilon_delta",None)
            if epsilon_delta is None: epsilon_delta = []
            else: epsilon_delta = epsilon_delta*self.get_Delta()
            self.set_knm_inv(core.op.Knm_inv(x=self.get_x(),y=self.get_y(),epsilon=epsilon,reg_matrix=epsilon_delta),**kwargs)
        return self.knm_inv
    def get_knm(self,**kwargs):
        if not hasattr(self, 'knm') or self.knm is None:
            self.set_knm(core.op.Knm(x=self.x,y=self.y),**kwargs)
        return self.knm
    def set_knm_inv(self,k,**kwargs):
        self.knm_inv = k
        self.set_theta(None)
    def set_knm(self,k,**kwargs):
        self.knm = k
    def get_x(self,**kwargs):
        if not hasattr(self, 'x'):self.x= None
        return self.x
    def set_x(self,x,set_polynomial_regressor = True,**kwargs):
        self.x = x.copy()
        self.set_y(**kwargs)
        if set_polynomial_regressor:
            self.set_polynomial_regressor(**kwargs)
        self.set_knm_inv(None)
        self.set_knm(None)
        self.rescale()
    def set_y(self,y=None,**kwargs):
        if y is None:self.y = self.get_x()
        else: self.y = y.copy()
    def get_y(self,**kwargs):
        if not hasattr(self, 'y') or self.y is None:self.set_y()
        return self.y
    def get_fx(self,**kwargs):
        if not hasattr(self, 'fx'):self.fx= None
        return self.fx
    def set_fx(self,fx,set_polynomial_regressor = True,**kwargs):
        if fx is not None: self.fx = fx.copy()
        else: self.fx = None
        if set_polynomial_regressor :self.set_polynomial_regressor(**kwargs)
        self.set_theta(None)
    def set_theta(self,theta,**kwargs):
        self.theta = theta
        if theta is None: return
        self.fx = None
        # self.fx =  lalg.prod(self.get_knm(),self.theta)
        # if self.get_order() is not None :
        #     self.fx += self.get_polynomial_regressor(z=self.get_x())
    def get_theta(self,**kwargs):
        if not hasattr(self, 'theta') or self.theta is None:
            if self.get_order() is not None and self.get_fx() is not None:
                fx= self.fx- self.get_polynomial_regressor(z=self.get_x())
            else: fx=self.get_fx()
            if fx is None: self.theta = None
            else: self.theta =  lalg.prod(self.get_knm_inv(),fx)
        return self.theta

    def get_Delta(self,**kwargs):
        if self.Delta is None: self.Delta = codpy.diffops.nablaT_nabla(self.y,self.x)
        return self.Delta
    def select(self,x,N,fx=None,**kwargs):
        if N is None: N = self.max_nystrom
        self.set_x(x,**kwargs)
        self.set_fx(fx,**kwargs)
        self.rescale()
        if self.get_fx() is not None:
            if self.get_polynomial_values() is not None :
                polynomial_values = self.get_polynomial_regressor(z=self.get_x())
                fx= self.fx- polynomial_values
            else: fx=self.fx
            theta,indices = alg.HybridGreedyNystroem(x=self.get_x(),fx=fx,N=N,tol=0.,error_type="frobenius",**kwargs)
            if kwargs.get("all",False):
                self.set(x=self.x,y=self.x[indices],fx=self.fx,set_polynomial_regressor=False)
            else:
                self.set_x(self.x[indices],set_polynomial_regressor=False)
                self.set_fx(self.fx[indices],set_polynomial_regressor=False)
                self.set_theta(theta)
            return indices

        if self.x.shape[0] <= N:
            indices = list(range(self.x.shape[0]))
            return indices
        indices = [0]
        complement_indices = list(range(1,N))
        for n in range(N-1):
            Dnm = core.op.Dnm(x[indices],x[complement_indices])
            indice = np.max(Dnm, axis=0)
            indice = np.argmax(indice)
            indice = complement_indices[indice]
            complement_indices.remove(indice)
            indices.append(indice)
            pass
        self.set_x(self.x[indices])
        return indices



    def set(self,x=None,fx=None,**kwargs):
        if x is None and fx is None: return
        if x is not None and fx is None:
            self.set_x(core.get_matrix(x.copy()),**kwargs)
            self.set_fx(None)
            self.rescale(**kwargs)

        if x is None and fx is not None:
            if self.kernel is None: raise Exception('Please input x at least once')
            if fx.shape[0] != self.x.shape[0] : raise Exception("fx of size " + str(fx.shape[0]) + "must have the same size as x" + str(self.x.shape[0]))
            self.set_fx(core.get_matrix(fx))

        if x is not None and fx is not None:
            self.set_x(x,**kwargs),self.set_fx(fx,**kwargs)
            self.rescale(**kwargs)
            pass
        return self

    def map(self,x,y,**kwargs):
        self.set_x(x,**kwargs),self.set_fx(y,**kwargs)
        self.rescale(**kwargs)

        if x.shape[1] != y.shape[1]:
            self.permutation = cd.alg.encoder(self.get_x(),self.get_fx())
        else:
            D = core.op.Dnm(x = x, y = y, distance = kwargs.get("distance", None))
            self.permutation = lsap(D,bool(kwargs.get("sub", False)))
        # self.set_fx(self.get_fx()[self.permutation])
        self.set_x(self.get_x()[self.permutation])
        return self

    def __len__(self):
        if self.x is None: return 0
        return self.x.shape[0]

    def update_set(self,z,fz,**kwargs):
        return z[-self.max_pool:],fz[-self.max_pool:]


    def update(self,z,fz,**kwargs):
        self.set_kernel_ptr()
        if isinstance(z,list): return  [self.__call__(x,**kwargs) for x in z]
        z = core.get_matrix(z)
        if self.x is None : return None
        Knm= core.op.Knm(x=z,y=self.get_y())
        if self.order is not None:
            fzz = fz - self.get_polynomial_regressor(z)
        else: fzz = fz
        # err = self(z)-fz
        # err= (err**2).sum()
        eps = kwargs.get("eps",self.reg)
        # if self.theta is not None: fzz += eps*lalg.prod(Knm,self.theta)
        self.set_theta(lalg.lstsq(Knm, fzz, eps =eps))
        # err = self(z)-fz
        # err= (err**2).sum()

        if self.order is not None:
            self.fx += self.get_polynomial_regressor(z=self.get_x())

        return
    def add(self,x=None,fx=None,**kwargs):
        x,fx = core.get_matrix(x.copy()),core.get_matrix(fx.copy())
        # if self.x is not None and x is not None: x=np.concatenate([self.x,x.copy()])[-self.max_pool:]
        # if self.fx is not None and fx is not None: fx=np.concatenate([self.fx,fx.copy()])[-self.max_pool:]
        if not hasattr(self, 'x') or self.x is None:
            self.set(x,fx)
            return
        self.Knm,self.Knm_inv,y = alg.add(self.get_knm(),self.get_knm_inv(),self.get_x(),x)
        self.set_x(y)
        if fx is not None and self.get_fx() is not None:
            self.set_fx(np.concatenate([fx,self.get_fx()],axis=0))
        else: self.set_fx(fx)

        self.set_polynomial_regressor()
        pass
    def kernel_distance(self,z):
        return core.op.Dnm(x=z,y=self.x)

    def get_kernel(self):
        if not hasattr(self, 'kernel') :
            self.set_kernel()
            # self.order= None
            self.kernel =  core.kernel.get_kernel_ptr()
        return self.kernel


    def set_kernel_ptr(self,**kwargs):
        core.kernel.set_kernel_ptr(self.get_kernel())
        core.kernel.set_polynomial_order(0)
        core.kernel.set_regularization(self.reg)

    def rescale(self,**kwargs):
        self.set_kernel_ptr(**kwargs)
        if self.get_x() is not None: 
            core.kernel.rescale(self.get_x())
            self.kernel =  core.kernel.get_kernel_ptr()

    def __call__(self, z, **kwargs):
        if isinstance(z,list): return  [self.__call__(x,**kwargs) for x in z]
        z = core.get_matrix(z)
        if self.x is None :
            return None
        self.set_kernel_ptr()
        fy = kwargs.get("fx", self.get_theta())
        if fy is None: fy = self.get_knm_inv()
        Knm= core.op.Knm(x=z,y=self.get_y(),fy=fy)
        if self.order is not None:
            check= self.get_polynomial_regressor(z)
            Knm += check
        return Knm
