import autolik
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

def tanh(x):
    return (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())

originf = autolik.F(tanh)
g = autolik.grad(tanh)

x = np.linspace(-7,7,200)
plt.plot(x,np.array([originf([i]) for i in x]),
         x,np.array([g([i]) for i in x]))
plt.show()