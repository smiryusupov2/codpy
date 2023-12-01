import torch
class AAD:
    def gradient(fx, x, grad_outputs=None,**kwargs):
        if not isinstance(x,torch.Tensor) : 
            x = torch.tensor(x, requires_grad=True)
        if x.dim() == 1 : 
            out = torch.autograd.functional.jacobian(fx,x)
            return out
        N,D = x.shape[0],x.shape[1]
        out = [get_matrix(torch.autograd.functional.jacobian(fx,y)) for y in x]
        return np.asarray(out)

    def taylor_expansion(x, z, fx, order = False,**kwargs):
        # print('######','distance_labelling','######')
        xo,zo,fxo = get_matrix(x),get_matrix(z),get_matrix(fx(x))
        indices = kwargs.get("indices",[])
        if len(indices) != z.shape[0] :
            indices = alg.distance_labelling(x,z,axis=0,**kwargs)
        xo= xo[indices]
        fxo= fxo[indices]
        deltax = zo - xo
        order = int(kwargs.get("taylor_order",1))
        results = kwargs.get("taylor_explanation",None)
        if order >=1:
            grad = AAD.gradient(fx=fx, x=x, **kwargs)
            if grad.ndim==1:
                out = np.zeros(deltax.shape)
                def helper(n):out[n] = grad 
                [helper(n) for n in range(0,deltax.shape[0])]
                grad = out
            else:
                grad = np.squeeze(grad)
                if len(indices) : grad = grad[indices].reshape(deltax.shape)
            product_ = np.reshape([np.dot(grad[n],deltax[n]) for n in range(grad.shape[0])],(len(grad),1))
            f_z = get_matrix(fxo)  + product_
            if results is not None:
                results["delta"] = deltax
                results["nabla"] = grad

        if order >=2:
            Nx,D,Df = x.shape[0],x.shape[1],fxo.shape[1]
            hess = AAD.hessian(fx =fx, x=x, **kwargs)
            if len(indices): hess = hess[indices]
            deltax = np.reshape([np.outer(deltax[n,:],deltax[n,:]) for n in range(deltax.shape[0])], (hess.shape[0],hess.shape[1],hess.shape[2]))
            quadratic_form = np.reshape([np.trace( hess[n].T@deltax[n] ) for n in range(hess.shape[0])], (hess.shape[0],1))
            f_z += 0.5*quadratic_form
            if results is not None:
                results["quadratic"] = deltax
                results["hessian"] = hess
        return f_z

    def nabla(x,y,z,fx, grad_outputs=None,**kwargs):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
        if x.shape[0] == z.shape[0] : return grad
        indices = alg.distance_labelling(x,z,**kwargs)
        x= x[indices]
        fx= fx[indices]

        return grad[indices]

    def jacobian(y, x):
        jac = torch.zeros(y.shape[0], x.shape[0]) 
        for i in range(y.shape[0]):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[i] = 1
            jac[i] = gradient(y, x, grad_outputs = grad_outputs)
        return jac

    def jacobianBatch(f, wrt):
        jacobian = []
        for i in range(wrt.shape[0]):
            jac = torch.autograd.functional.jacobian(f, wrt[i])
            jacobian.append(jac)
        return torch.stack(jacobian, 0)

    def hessian(fx,x,**kwargs):
        Nx,D = x.shape[0],x.shape[1]
        if not isinstance(x,torch.Tensor) : 
            x = torch.tensor(x, requires_grad=True)
        def hessian_helper(y):
            mat = torch.autograd.functional.hessian(func=fx,inputs= y)
            return get_matrix(mat)
        out = [hessian_helper(y= y.clone().detach().requires_grad_(True)) for y in x]
        return np.asarray(out)

    def divergence(y, x,**kwargs):
        div = 0.
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
        return div

    def laplace(y, x,**kwargs):
        grad = gradient(y, x)
        return divergence(grad, x)