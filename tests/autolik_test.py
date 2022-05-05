import math
from autolik import Dual, grad, pdf
from autolik.autodiff import grad, _gradient, gradient, _hessian


###################
# Dual test
###################
def gradient_test():
    f = lambda x,y: -x**3 - 2 * x**y + x.exp() + (y ** 2).exp()

    g1 = lambda x,y: -3 * x**2 - 2 * y * x**(y-1) + math.exp(x)
    g2 = lambda x,y: -2 * x**y * math.log(x) + 2 * y * math.exp(y ** 2)

    at = [[1,1],[2,2],[3,3]]

    Grad = [grad(f)(x) for x in at]
    _Grad = [_gradient(f,x) for x in at]
    Gradient = gradient(f,at)
    analytical_result = [[g1(x[0],x[1]),g2(x[0],x[1])] for x in at]
    assert Grad == analytical_result
    assert _Grad == analytical_result
    assert Gradient == analytical_result

# def hessian_test():
#     f = lambda x,y: -x**3 - 2 * x**y + x.exp() + (y ** 2).exp()

#     g1 = lambda x,y: -3 * x**2 - 2 * y * x**(y-1) + math.exp(x)
#     g2 = lambda x,y: -2 * x**y * math.log(x) + 2 * y * math.exp(y ** 2)

#     h11 = lambda x,y: -6 * x - 2*y*(y-1)*x**(y-2) + math.exp(x)
#     h12 = lambda x,y: -2*x**(y-1) + 2*y*x**(y-1)*math.log(x)
#     h21 = lambda x,y: -2*y*(math.log(x)-1)
#     h22 = lambda x,y: -2*x*math.log(x) + 2*math.exp(y**2) + 2*y*2*y*math.exp(y**2)

#     at = [[1,1],[2,2],[3,3]]

#     print([_hessian(f,x) for x in at])
#     print([[[h11(x[0],x[1]), h12(x[0],x[1])],[h21(x[0],x[1]), h22(x[0],x[1])]] for x in at])


###################
# distribution test
###################

def pdf_test():
    print(pdf.normal(0,0,1))


if __name__ == "__main__":
    gradient_test()
    pdf_test()
    # hessian_test()