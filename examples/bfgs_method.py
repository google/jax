import jax
from jax import jit
from typing import Callable, Iterable
from newton_method import wolfe, g, ini, is_float

#@jit
def bfgs(fun: Callable, 
         x_0: Iterable,
         verbose: bool=False,
         epsilon: float=1e-6, 
         k: int=0):
    '''
    Hessian-based BFGS Method.
    '''
    assert all(list(map(is_float, x_0)))
    x_0 = jax.numpy.array(x_0) # same as params
    while 1:
        gradient = jax.grad(fun)(x_0)
        hessian = jax.hessian(fun)(x_0)
        dk = -jax.numpy.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        if verbose:
            print('{}\t{}\t{}'.format(x_0, fun(x_0), k))
        if jax.numpy.linalg.norm(dk) >= epsilon:
            alpha = wolfe(fun, gradient, x_0, dk)
            delta = alpha * dk # sk, same as learning_date * grad(corrected by hessian and jax.grad)
            x_0 += delta[0]
            yk = jax.grad(fun)(x_0) - gradient
            hess_delta = hessian.dot(delta.T)
            if yk.all != 0:
                hessian += (yk.T).dot(yk) / delta.dot(yk.T) - (hess_delta).dot(hess_delta.T) / delta.dot(hess_delta)
            k += 1
        else:
            break
    return x_0

if __name__ =='__main__':
    print(bfgs(g, ini, True))