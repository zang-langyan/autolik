from autolik import Dual
import autolik
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

def z(x,y):
    if isinstance(x,Dual) or isinstance(y,Dual):
        return x * (-x**2 - y**2).exp()
    else:
        return x * np.exp(-x**2 - y**2)

z_grad = autolik.grad(z)

x = np.linspace(-2,2,100)
y = np.linspace(-2,2,100)
X,Y = np.meshgrid(x,y)

# data
Z = np.array([[z(i,j) for i in x] for j in y])

vx = np.linspace(-2,2,50)
vy = np.linspace(-2,2,50)
vX,vY = np.meshgrid(vx,vy)

# gradient vectors
Z_grad = np.array([z_grad([i,j]) for i in vx for j in vy])
# Z_grad = np.array(autolik.gradient(z,at=[[i,j] for i in x for j in y]))
# print(Z_grad)

ax = plt.axes()
ax.contourf(X,Y,Z,levels=50, cmap=cm.jet, alpha = 0.7)
ax.quiver(vX,vY,Z_grad[:,0],Z_grad[:,1])
plt.show()