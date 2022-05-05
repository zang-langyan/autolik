.. autolik documentation master file, created by
   sphinx-quickstart on Thu May  5 17:09:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to autolik's documentation!
===================================

.. automodule:: autolik

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


autodiff
===================================

Common user interface for the gradient computation and gradient function construction 

.. automodule:: autolik.autodiff
   :members: F, grad, gradient

distributions and likelihood
===================================

Built in distibutions and their likelihood functions API

For probability density function usage, use ``autolik.pdf`` attribute; 
For log-likelihood function usage, use ``autolik.ll`` attribute;

.. automodule:: autolik.distributions.univariate
   :members:

Usage
===================================

Example (self-defined function)::

   >>> from autolik import Dual # for self-defined functions use
   >>> import autolik
   >>> import matplotlib.pyplot as plt
   >>> from matplotlib.pyplot import cm
   >>> import numpy as np
   >>> def z(x,y): # self-defined function
   ...     if isinstance(x,Dual) or isinstance(y,Dual): # autolik.Dual computation
   ...         return x * (-x**2 - y**2).exp()
   ...     else:
   ...         return x * np.exp(-x**2 - y**2)
   >>> z_grad = autolik.grad(z) # gradient function
   >>> x = np.linspace(-2,2,100)
   >>> y = np.linspace(-2,2,100)
   >>> X,Y = np.meshgrid(x,y)
   >>> Z = np.array([[z(i,j) for i in x] for j in y])
   >>> vx = np.linspace(-2,2,50)
   >>> vy = np.linspace(-2,2,50)
   >>> vX,vY = np.meshgrid(vx,vy)
   >>> Z_grad = np.array([z_grad([i,j]) for i in vx for j in vy]) # gradient vectors
   >>> ax = plt.axes()
   >>> ax.contourf(X,Y,Z,levels=50, cmap=cm.jet, alpha = 0.7)
   >>> ax.quiver(vX,vY,Z_grad[:,0],Z_grad[:,1])
   >>> plt.show()

.. figure:: ../examples/figures/contourf_x_exp_x2_y2.png
   :align: center

Example (built in log-likelihood funtion and its gradient)::

   >>> import autolik
   >>> import numpy as np
   >>> y = np.random.normal(-1,2,100)
   >>> def loglik_normal(mu,sigma): # prepare the logliklihood function
   ...    return autolik.ll.normal(y, mu, sigma)
   >>> g = autolik.grad(loglik_normal) # generate the gradient for the logliklihood function
   >>> import matplotlib.pyplot as plt
   >>> from matplotlib.pyplot import cm
   >>> mu = np.linspace(-2,2,50)
   >>> sig = np.linspace(1,6,50)
   >>> Mu,Sigma = np.meshgrid(mu,sig)
   >>> loglik = np.array([[loglik_normal(i,j) for i in mu] for j in sig])
   >>> loglik_grad = np.array([g([i,j]) for i in mu for j in sig])
   >>> ax = plt.axes()
   >>> ax.contourf(Mu,Sigma,loglik, levels=50, cmap=cm.jet, alpha = 0.8) # plot the contour of log-likelihood function
   >>> ax.quiver(Mu,Sigma,loglik_grad[:,0],loglik_grad[:,1]) # plot the gradient vectors of log-likelihood function
   >>> plt.show()

.. figure:: ../examples/figures/contourf_loglik_normal.png
   :align: center