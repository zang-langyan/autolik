from autolik.distributions.univariate import pdf
from autolik.Dual.benchmark import *

class ll:
    """Log-Likelihood functions library"""
    def beta(y,beta,gam):
        length = len(y)
        lik = list(map(pdf.beta, y, [beta]*length, [gam]*length))
        if isinstance(beta,Dual) or isinstance(gam,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def cauchy(y,a,alpha):
        length = len(y)
        lik = list(map(pdf.cauchy, y, [a]*length, [alpha]*length))
        if isinstance(a,Dual) or isinstance(alpha,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def chi(y,n):
        length = len(y)
        lik = list(map(pdf.chi, y, [n]*length))
        if isinstance(n,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def chisqr(y,n):
        length = len(y)
        lik = list(map(pdf.chisqr, y, [n]*length))
        if isinstance(n,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def exponential(y,lam):
        length = len(y)
        lik = list(map(pdf.exponential, y, [lam]*length))
        if isinstance(lam,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def gamma(y,alpha,beta):
        length = len(y)
        lik = list(map(pdf.gamma, y, [alpha]*length, [beta]*length))
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def Ggamma(y,alpha,beta,gam):
        length = len(y)
        lik = list(map(pdf.Ggamma, y, [alpha]*length, [beta]*length),[gam]*length)
        if isinstance(alpha,Dual) or isinstance(beta,Dual) or isinstance(gam,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def Gpareto(y,delta,kappa,gam):
        length = len(y)
        lik = list(map(pdf.Gpareto, y, [delta]*length, [kappa]*length),[gam]*length)
        if isinstance(delta,Dual) or isinstance(kappa,Dual) or isinstance(gam,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def invgaussian(y,lam,mu):
        length = len(y)
        lik = list(map(pdf.invgaussian, y, [lam]*length, [mu]*length))
        if isinstance(lam,Dual) or isinstance(mu,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def invgamma(y,alpha,beta):
        length = len(y)
        lik = list(map(pdf.invgamma, y, [alpha]*length, [beta]*length))
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def laplace(y,alpha1,alpha2):
        length = len(y)
        lik = list(map(pdf.laplace, y, [alpha1]*length, [alpha2]*length))
        if isinstance(alpha1,Dual) or isinstance(alpha2,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def loggamma(y,alpha,beta):
        length = len(y)
        lik = list(map(pdf.loggamma, y, [alpha]*length, [beta]*length))
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def loglogistic(y,lam,kappa):
        length = len(y)
        lik = list(map(pdf.loglogistic, y, [lam]*length, [kappa]*length))
        if isinstance(lam,Dual) or isinstance(kappa,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def lognormal(y,alpha,beta):
        length = len(y)
        lik = list(map(pdf.lognormal, y, [alpha]*length, [beta]*length))
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def logistic(y,lam,kappa):
        length = len(y)
        lik = list(map(pdf.logistic, y, [lam]*length, [kappa]*length))
        if isinstance(lam,Dual) or isinstance(kappa,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def logistic_exp(y,alpha,beta):
        length = len(y)
        lik = list(map(pdf.logistic_exp, y, [alpha]*length, [beta]*length))
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def lomax(y,lam,kappa):
        length = len(y)
        lik = list(map(pdf.lomax, y, [lam]*length, [kappa]*length))
        if isinstance(lam,Dual) or isinstance(kappa,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def makeham(y,delta,kappa,gam):
        length = len(y)
        lik = list(map(pdf.makeham, y, [delta]*length, [kappa]*length, [gam]*length))
        if isinstance(delta,Dual) or isinstance(kappa,Dual) or isinstance(gam,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def minimax(y,beta,gam):
        length = len(y)
        lik = list(map(pdf.minimax, y, [beta]*length, [gam]*length))
        if isinstance(beta,Dual) or isinstance(gam,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def muth(y,kappa):
        length = len(y)
        lik = list(map(pdf.muth, y, [kappa]*length))
        if isinstance(kappa,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def normal(y,mu,sigma):
        length = len(y)
        lik = list(map(pdf.normal, y, [mu]*length, [sigma]*length))
        if isinstance(mu,Dual) or isinstance(sigma,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def pareto(y,lam,kappa):
        length = len(y)
        lik = list(map(pdf.pareto, y, [lam]*length, [kappa]*length))
        if isinstance(lam,Dual) or isinstance(kappa,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def power(y,alpha,beta):
        length = len(y)
        lik = list(map(pdf.power, y, [alpha]*length, [beta]*length))
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)
    
    def std_power(y,beta):
        length = len(y)
        lik = list(map(pdf.std_power, y, [beta]*length))
        if isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def rayleigh(y,alpha):
        length = len(y)
        lik = list(map(pdf.rayleigh, y, [alpha]*length))
        if isinstance(alpha,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def std_wald(y,lam):
        length = len(y)
        lik = list(map(pdf.std_wald, y, [lam]*length))
        if isinstance(lam,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def Tdist(y,n):
        length = len(y)
        lik = list(map(pdf.Tdist, y, [n]*length))
        if isinstance(n,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def uniform(y,a,b):
        length = len(y)
        lik = list(map(pdf.uniform, y, [a]*length, [b]*length))
        if isinstance(a,Dual) or isinstance(b,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)

    def weibull(y,alpha,beta):
        length = len(y)
        lik = list(map(pdf.weibull, y, [alpha]*length, [beta]*length))
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            loglik = list(map(Dual.log, lik))
        else:
            loglik = list(map(math.log, lik))
        return sum(loglik)