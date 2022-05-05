import math
from numbers import Real
import scipy.special

class Dual:
    """Autolik Dual type"""
    def __init__(self, real=0., eps=0.) -> None:
        self.real = real
        self.eps = eps

    ############################
    # basic comparison operators
    ############################
    def __lt__(self,x):
        if isinstance(x,Dual):
            return self.real < x.real
        elif isinstance(x,Real):
            return self.real < x

    def __gt__(self,x):
        if isinstance(x,Dual):
            return self.real > x.real
        elif isinstance(x,Real):
            return self.real > x

    def __rlt__(self,x):
        return self.__gt__(x)

    def __rgt__(self,x):
        return self.__lt__(x)

    def __le__(self,x):
        if isinstance(x,Dual):
            return self.real <= x.real
        elif isinstance(x,Real):
            return self.real <= x

    def __ge__(self,x):
        if isinstance(x,Dual):
            return self.real >= x.real
        elif isinstance(x,Real):
            return self.real >= x

    def __rle__(self,x):
        return self.__ge__(x)

    def __rge__(self,x):
        return self.__le__(x)

    ############################
    # basic numeric operators
    ############################
    def __neg__(self):
        return Dual(-self.real,-self.eps)

    def __add__(self,x):
        if isinstance(x,Dual):
            return Dual(self.real + x.real, self.eps + x.eps)
        if isinstance(x,Real):
            return Dual(self.real + float(x), self.eps)

    def __radd__(self,x):
        return self.__add__(x)
    
    def __sub__(self,x): # Dual - Dual or Dual - float
        if isinstance(x,Dual):
            return Dual(self.real - x.real, self.eps - x.eps)
        if isinstance(x,Real):
            return Dual(self.real - float(x) , self.eps)

    def __rsub__(self,x): # Dual - Dual or float - Dual
        if isinstance(x,Dual):
            return Dual(x.real - self.real, x.eps - self.eps)
        if isinstance(x,Real):
            return Dual(float(x) - self.real , -self.eps)

    def __mul__(self,x): # Dual * Dual or Dual * float
        if isinstance(x,Dual):
            return Dual(self.real * x.real, self.real * x.eps + x.real * self.eps)
        if isinstance(x,Real):
            return Dual(float(x) * self.real, float(x) * self.eps)
    
    def __rmul__(self,x): # float * Dual
        return self.__mul__(x)

    def __truediv__(self,x): # Dual / float
        return self.__mul__(1 / x)

    def __rtruediv__(self,x): # float / Dual
        if isinstance(x,Dual):
            return Dual(x.real / self.real, (x.eps * self.real - x.real * self.eps) / self.real ** 2)
        if isinstance(x,Real):
            return Dual(float(x) / self.real, (- float(x) * self.eps) / self.real ** 2)

    def __pow__(self,x): # Dual^()
        if isinstance(x,Dual):
            pow_real = self.real ** x.real
            pow_diriv = pow_real * (x.eps * math.log(self.real) + x.real / self.real * self.eps)
            return Dual(pow_real,pow_diriv)
        if isinstance(x,Real):
            pow_real = self.real ** x
            pow_diriv = x * (self.real ** (x-1)) * self.eps
            return Dual(pow_real,pow_diriv)

    def __rpow__(self,x): # ()^Dual
        if isinstance(x,Dual):
            pow_real = x.real ** self.real
            pow_diriv = pow_real * (self.eps * math.log(x.real) + self.real / x.real * x.eps)
            return Dual(pow_real,pow_diriv)
        if isinstance(x,Real):
            pow_real = float(x) ** self.real
            pow_diriv = pow_real * math.log(float(x))
            return Dual(pow_real,pow_diriv)
        
    ############################
    # special functions
    ############################
    def sin(self):
        return Dual(math.sin(self.real), self.eps * math.cos(self.real))

    def cos(self):
        return Dual(math.cos(self.real), - self.eps * math.sin(self.real))

    def sqrt(self):
        return Dual(math.sqrt(self.real), self.eps / (2 * math.sqrt(self.real)))

    def exp(self):
        return Dual(math.exp(self.real), self.eps * math.exp(self.real))

    def log(self):
        return Dual(math.log(self.real), self.eps * (1 / self.real))

    def gamma(self):
        return Dual(scipy.special.gamma(self.real), self.eps * scipy.special.gamma(self.real) * scipy.special.polygamma(0,self.real))

 
def beta(a:Dual,b:Dual) -> Dual:
    return Dual.gamma(a) * Dual.gamma(b) / Dual.gamma(a + b)