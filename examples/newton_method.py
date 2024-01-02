import jax
from typing import Callable, Iterable

is_float = lambda x: isinstance(x, (float, int))

ini = [1., -1., 1., -1.]

g = lambda x: (-13 + x[0] + ((5 - x[1])*x[1] - 2)*x[1])**2 + (-29 + x[0] + ((x[1] + 1)*x[1] - 14)*x[1])**2 + (-13 + x[2] + ((5 - x[3])*x[3] - 2)*x[3])**2 + (-29 + x[2] + ((x[3] + 1)*x[3] - 14)*x[3])**2

def wolfe(fun: Callable,
          res0, 
          x_0: Iterable, 
          dk, 
          c1: float=0.3, 
          c2: float=0.5, 
          alphas: float=0., 
          alphae: float=2.,
          eps: float=1e-3) -> float:
    '''
    Step size finder.
    '''
    assert c1 > 0
    assert c1 < 1
    assert c2 > 0
    assert c2 < 1
    assert c1 < c2
    assert alphas >= 0
    assert alphas < alphae
    assert eps > 0
    alpha = 1
    f0 = fun(x_0)
    while alphas < alphae:
        x = x_0 + (alpha*dk)[0]
        f1 = fun(x)
        if f1 <= f0 + c1*alpha*res0.dot(dk.T):
            res1 = jax.grad(fun)(jax.numpy.array(x))
            if res1.dot(dk.T) >= c2*res0.dot(dk.T):
                break
            else:
                alphas = alpha
                alpha = 0.5 * (alphas + alphae)
        else:
            alphae = alpha
            alpha = 0.5 * (alphas + alphae)
        if jax.numpy.abs(alphas - alphae) < eps:
            break
    return alpha


def newton(fun: Callable, 
           x_0: Iterable, 
           alpha: float=0.02, 
           epsilon: float=1e-6, 
           k: int=0):
    assert all(list(map(is_float, x_0)))
    x_0 = jax.numpy.array(x_0)
    while 1:
        gradient = jax.grad(fun)(jax.numpy.array(x_0))
        hessian = jax.hessian(fun)(jax.numpy.array(x_0))
        dk = -jax.numpy.linalg.inv(hessian).dot(gradient.T).reshape(1, -1)
        print('{}\t{}\t{}'.format(x_0, fun(x_0), k))
        if jax.numpy.linalg.norm(dk) >= epsilon:
            alpha = wolfe(fun, gradient, x_0, dk)
            x_0 += alpha * dk[0]
            k += 1
        else:
            break

newton(g, ini)