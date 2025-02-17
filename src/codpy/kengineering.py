import math
import warnings
import functools
import numpy as np
import abc

import codpy.algs
from codpy.core import _requires_rescale, KerInterface, kernel_setter, get_matrix
from codpy.data_conversion import get_matrix
from codpy.selection import column_selector
from codpy.kernel import Kernel
from codpy.sampling import rejection_sampling
from codpy.lalg import LAlg
from codpy.plot_utils import multi_plot
from codpy.utils import cartesian_outer_product
import pandas as pd

class KernelPair(Kernel):
    def __init__(self,k1,k2,**kwargs):
        self.k1=k1
        self.k2=k2
        super().__init__(**kwargs)
    def split(self,x):
        return x[:,:self.k1.dim()],x[:,self.k1.dim():]

class BitwiseANDKernel(KernelPair):
    def __init__(self,k1,k2,**kwargs):
        x=np.concatenate([k1.get_x(),k2.get_x()],axis=1)
        super().__init__(k1,k2,x=x,**kwargs)
    def knm(
        self, x: np.ndarray, y: np.ndarray, fy: np.ndarray = [], **kwargs
    ) -> np.ndarray:
        x1,x2=self.split(x)
        y1,y2=self.split(y)
        out = self.k1.knm(x1,y1) * self.k2.knm(x2,y2)
        return out
    def get_x(self, **kwargs) -> np.ndarray:
        return np.concatenate([self.k1.get_x(),self.k2.get_x()],axis=1)
    def set_x(self,x, **kwargs) -> np.ndarray:
        x1,x2=self.split(x)
        self.k1.set_x(x1),self.k2.set_x(x2)
        super().set_x(x)


class KEKernel(Kernel):
    def __init__(self,k1,k2,**kwargs):
        self.k1=k1
        self.k2=k2
    def set_custom_kernel(self,**kwargs) -> None:
        raise AssertionError
    def knm(
        self, x: np.ndarray, y: np.ndarray, fy: np.ndarray = [], **kwargs
    ) -> np.ndarray:
        raise NotImplementedError
    def dnm(
        self, x: np.ndarray = None, y: np.ndarray = None, fy: np.ndarray = [], **kwargs
    ) -> np.ndarray:
        raise NotImplementedError
    def get_x(self, **kwargs) -> np.ndarray:
        raise NotImplementedError
    def set_x(
        self, x: np.ndarray, set_polynomial_regressor: bool = True, **kwargs
    ) -> None:
        raise NotImplementedError
    def set_y(self, y: np.ndarray = None, **kwargs) -> None:
        raise NotImplementedError
    def get_y(self, **kwargs) -> np.ndarray:
        raise NotImplementedError
    def get_fx(self, **kwargs) -> np.ndarray:
        raise NotImplementedError
    def set_fx(
        self, fx: np.ndarray, set_polynomial_regressor: bool = True, **kwargs
    ) -> None:
        raise NotImplementedError
    def set_theta(self, theta: np.ndarray, **kwargs) -> None:
        raise NotImplementedError
    def get_theta(self, **kwargs) -> np.ndarray:
        raise NotImplementedError
    def get_Delta(self) -> np.ndarray:
        raise NotImplementedError
    def greedy_select(
        self, N, x=None, fx=None, all=False, n_batch=1, norm="frobenius", **kwargs
    ):
        raise NotImplementedError
    def set(
        self,
        x: np.ndarray = None,
        fx: np.ndarray = None,
        y: np.ndarray = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def map(self,**kwargs,) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def update(self, z: np.ndarray, fz: np.ndarray, eps: float = None, **kwargs) -> None:
        raise NotImplementedError

    def add(self, y: np.ndarray = None, fy: np.ndarray = None) -> None:
        raise NotImplementedError

    def kernel_distance(self, y: np.ndarray, x=None) -> np.ndarray:
        raise NotImplementedError

    def discrepancy(self, z: np.ndarray) -> float:
        raise NotImplementedError

    def get_kernel(self) -> callable:
        raise NotImplementedError

    def set_kernel_ptr(self) -> None:
        raise NotImplementedError

    def set_map(self, map_) -> callable:
        raise NotImplementedError

    def get_map(self) -> callable:
        raise NotImplementedError

    def rescale(self) -> None:
        raise NotImplementedError

    def __call__(self, z: np.ndarray,**kwargs) -> np.ndarray:
        raise NotImplementedError

    def grad(self, z: np.ndarray,**kwargs) -> np.ndarray:
        raise NotImplementedError
    
    
    