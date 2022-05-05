from zB_loglik_normal_data import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


ax = plt.axes()
# ax.plot_surface
ax.contourf(Mu,Sigma,loglik, levels=50, cmap=cm.jet, alpha = 0.8)
ax.quiver(Mu,Sigma,loglik_grad[:,0],loglik_grad[:,1])
plt.show()