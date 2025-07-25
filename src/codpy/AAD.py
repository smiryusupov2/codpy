import numpy as np
import torch as torch

from codpy import core

class AAD:
    def gradient(fx, x, grad_outputs=None, **kwargs):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, requires_grad=True)
        if x.dim() == 1:
            # Wrapper function to ensure tensor output
            def tensor_fx(t):
                result = fx(t, **kwargs)
                if isinstance(result, np.ndarray):
                    # For gradients, we also want scalar output when input is 1D
                    scalar_result = result.item() if result.size == 1 else result[0]
                    return torch.tensor(scalar_result, dtype=torch.float64)
                elif isinstance(result, (int, float)):
                    return torch.tensor(result, dtype=torch.float64)
                else:
                    return result
            out = torch.autograd.functional.jacobian(tensor_fx, x)
            return out
        N, D = x.shape[0], x.shape[1]
        # Wrapper function to ensure tensor output
        def tensor_fx_wrapper(y_):
            result = fx(y_, **kwargs)
            if isinstance(result, np.ndarray):
                return torch.tensor(result, dtype=torch.float64)
            elif isinstance(result, (int, float)):
                return torch.tensor(result, dtype=torch.float64)
            else:
                return result
        out = [torch.autograd.functional.jacobian(tensor_fx_wrapper, y).detach().numpy() for y in x]
        return np.asarray(out)

    def taylor_expansion(x, z, fx, order=False, **kwargs):
        # print('######','distance_labelling','######')
        fz = fx(x, **kwargs)
        xo, zo, fxo = core.get_matrix(x), core.get_matrix(z), core.get_matrix(fz)
        indices = kwargs.get("indices", [])
        if len(indices) != z.shape[0]:
            indices = core.Misc.distance_labelling(
                **{**kwargs, **{"x": x, "axis": 0, "y": z}}
            )
        xo = xo[indices]
        fxo = fxo[indices]
        deltax = zo - xo
        order = int(kwargs.get("taylor_order", 1))
        results = kwargs.get("taylor_explanation", None)
        if order >= 1:
            grad = AAD.gradient(fx=fx, x=x, **kwargs)
            if grad.ndim == 1:
                out = np.zeros(deltax.shape)

                def helper(n):
                    out[n] = grad

                [helper(n) for n in range(0, deltax.shape[0])]
                grad = out
            else:
                grad = np.squeeze(grad)
                if len(indices):
                    grad = grad[indices].reshape(deltax.shape)
            product_ = np.reshape(
                [np.dot(grad[n], deltax[n]) for n in range(grad.shape[0])],
                (len(grad), 1),
            )
            f_z = core.get_matrix(fxo) + product_
            if results is not None:
                results["delta"] = deltax
                results["nabla"] = grad

        if order >= 2:
            Nx, D, Df = x.shape[0], x.shape[1], fxo.shape[1]
            hess = AAD.hessian(fx=fx, x=x, **kwargs)
            if len(indices):
                hess = hess[indices]
            deltax = np.reshape(
                [np.outer(deltax[n, :], deltax[n, :]) for n in range(deltax.shape[0])],
                (hess.shape[0], hess.shape[1], hess.shape[2]),
            )
            quadratic_form = np.reshape(
                [np.trace(hess[n].T @ deltax[n]) for n in range(hess.shape[0])],
                (hess.shape[0], 1),
            )
            f_z += 0.5 * quadratic_form
            if results is not None:
                results["quadratic"] = deltax
                results["hessian"] = hess
        return f_z

    def nabla(x, y, z, fx, grad_outputs=None, **kwargs):
        import torch as torch

        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(
            y, [x], grad_outputs=grad_outputs, create_graph=True
        )[0]
        if x.shape[0] == z.shape[0]:
            return grad
        indices = core.Misc.distance_labelling(**{**kwargs, **{"x": x, "y": z}})
        x = x[indices]
        fx = fx[indices]

        return grad[indices]

    def jacobian(y, x):
        import torch as torch

        jac = torch.zeros(y.shape[0], x.shape[0])
        for i in range(y.shape[0]):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[i] = 1
            jac[i] = AAD.gradient(y, x, grad_outputs=grad_outputs)
        return jac

    def jacobianBatch(f, wrt):
        import torch as torch

        jacobian = []
        for i in range(wrt.shape[0]):
            jac = torch.autograd.functional.jacobian(f, wrt[i])
            jacobian.append(jac)
        return torch.stack(jacobian, 0)

    def hessian(fx, x, **kwargs):
        import torch as torch

        Nx, D = x.shape[0], x.shape[1]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, requires_grad=True)

        def hessian_helper(y):
            # Wrapper function to ensure tensor output and pass kwargs
            def tensor_fx_hessian(t):
                result = fx(x=t, **kwargs)
                if isinstance(result, np.ndarray):
                    # For hessian, we need a scalar output, so take the first element if it's an array
                    scalar_result = result.item() if result.size == 1 else result[0]
                    return torch.tensor(scalar_result, dtype=torch.float64)
                elif isinstance(result, (int, float)):
                    return torch.tensor(result, dtype=torch.float64)
                else:
                    # If it's already a tensor, ensure it's scalar
                    assert result.numel() == 1, "Hessian function must return a scalar value."
                    # if hasattr(result, 'item'):
                    #     return result.item() if result.numel() == 1 else result[0]
                    return result
            mat = torch.autograd.functional.hessian(func=tensor_fx_hessian, inputs=y)
            return core.get_matrix(mat)

        out = [hessian_helper(y=y.clone().detach().requires_grad_(True)) for y in x]
        return np.asarray(out)
    
    def divergence(y, x, **kwargs):
        import torch as torch

        div = 0.0
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(
                y[..., i], x, torch.ones_like(y[..., i]), create_graph=True
            )[0][..., i : i + 1]
        return div

    def laplace(y, x, **kwargs):
        grad = AAD.gradient(y, x)
        return AAD.divergence(grad, x)
