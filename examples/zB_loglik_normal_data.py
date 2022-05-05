import autolik
import numpy as np

y = np.random.normal(-1,2,100)

def loglik_normal(mu,sigma):
    return autolik.ll.normal(y, mu, sigma)

g = autolik.grad(loglik_normal)

# set mu and sigma range
mu = np.linspace(-2,2,50)
sig = np.linspace(1,6,50)
Mu,Sigma = np.meshgrid(mu,sig)

# compute loglike function and gradient direction
loglik = np.array([[loglik_normal(i,j) for i in mu] for j in sig])
loglik_grad = np.array([g([i,j]) for i in mu for j in sig])
