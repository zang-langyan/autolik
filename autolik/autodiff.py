from typing import Callable
from autolik.Dual.benchmark import Dual

def _F(f:Callable,at):
    """Compute the value of a Dual-type function at specific points"""
    X = tuple(Dual(real) for real in at)

    result = []

    for _ in X:
        result.append(f(*X).real)
    
    return result

def F(f:Callable):
    """Construct a function for Dual-type function computation"""
    return lambda x: _F(f,at=x)

def _gradient(f:Callable,at):
    """Compute the gradient of a provided function at specific points (hidden)"""
    X = tuple(Dual(real) for real in at)

    result = []

    for dual in X:
        dual.eps = 1.
        result.append(f(*X).eps)
        dual.eps = 0.
    
    return result

def grad(f:Callable):
    """Construct a gradient function

    construct a gradient function with respect to x

    Example:

    >>> from autolik import grad
    >>> f = lambda x,y: -x**3 - 2 * x**y + x.exp() + (y ** 2).exp()
    >>> g = grad(f)
    >>> at = [[1,1],[2,2],[3,3]]
    >>> Grad = [grad(f)(x) for x in at]
    
    """
    return lambda x: _gradient(f,x)

def gradient(f:Callable,at):
    """Compute the gradient of a provided function at specific points
    
    Common interface for gradient computation

    Example:
    
    >>> from autolik import grad
    >>> f = lambda x,y: -x**3 - 2 * x**y + x.exp() + (y ** 2).exp()
    >>> at = [[1,1],[2,2],[3,3]]
    >>> Grad = gradient(f,at)

    """
    result = []
    for xs in at:
        result.append(_gradient(f,xs))

    return result







# TODO: fix the hessian computation
def _hessian(f:Callable,at):
    X = tuple(Dual(real) for real in at)

    result = [[0 for _ in at] for _ in at]

    for i, dual0 in enumerate(X):
        for j, dual1 in enumerate(X):
            dual0.eps, dual1.eps = 1., 1.
            result[i][j] = f(*X).eps
            dual0.eps, dual1.eps = 0., 0.
    
    return result


# depreciated #
# def _rawf(f:Callable,wrt:Tuple,at):
#     at_len = len(at)
#     x_len = len(wrt)
#     assert at_len == x_len
    
#     results = [Dual() for _ in range(x_len)]
    
#     for j, dual in enumerate(wrt):
#         dual.real = at[j]
#     for j, dual in enumerate(wrt):
#         results[j] = f(*wrt)

#     return [i.real for i in results]

# def F(f:Callable):
#     return lambda x: _rawf(f,wrt=tuple([Dual() for _ in x]),at=x)

# def _gradient(f:Callable,wrt:Tuple,at):
#     at_len = len(at)
#     x_len = len(wrt)
#     assert at_len == x_len
    
#     results = [Dual() for _ in range(x_len)]
    
#     for j, dual in enumerate(wrt):
#         dual.real = at[j]
#     for j, dual in enumerate(wrt):
#         dual.eps = 1.
#         results[j] = f(*wrt)
#         dual.eps = 0.

#     return [i.eps for i in results]

# def gradient(f:Callable,wrt:Tuple,at):
#     grad_len = len(at)
#     x_len = len(wrt)
    
#     results = [[Dual() for j in range(x_len)] for i in range(grad_len)]
#     for i in range(grad_len):
#         for j, dual in enumerate(wrt):
#             dual.real = at[i][j]
#         for j, dual in enumerate(wrt):
#             dual.eps = 1.
#             results[i][j] = f(*wrt)
#             dual.eps = 0.

#     return [[i.eps for i in res] for res in results]

