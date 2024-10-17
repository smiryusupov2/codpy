import codpy.core as core
from codpydll import *
import numpy as np


class my_kernel(core.cd.kernel):
    def __init__(self,**kwargs) : 

        core.cd.kernel.__init__(self)
        self.bandwidth_=float(kwargs.get("bandwidth",1.))
    @staticmethod
    def create(kwargs={}):
        return my_kernel(**kwargs)
    @staticmethod
    def register(): 
        cd.kernel.register("my_kernel",my_kernel.create)
    def k(self,x,y):
        out = np.linalg.norm((x-y)*self.bandwidth_)
        return out*out*.5
    def grad(self,x,y):
        return y*self.bandwidth_



if __name__ == "__main__":
    core.set_verbose()
    x,y = np.random.randn(3, 2),np.random.randn(3, 2)
    my_kernel_ = my_kernel.create()
    print (my_kernel_.k(x[0],y[0]))

    my_kernel.set_kernel_ptr(my_kernel_)
    my_kernel_ptr = core.kernel_interface.get_kernel_ptr()
    print (my_kernel_.k(x[0],y[0]))
    print(core.op.Knm(x,y))

    my_kernel.register()
    my_kernel_ptr = core.factories.get_kernel_factory()["my_kernel"]({"bandwidth":"2."})
    print(my_kernel_ptr.k(x[0],y[0]))
    my_kernel.set_kernel_ptr(my_kernel_ptr)
    print(core.op.Knm(x,y))

    core.set_kernel("my_kernel",extras = {"bandwidth":"3."})
    print(core.op.Knm(x,y))

    pass
