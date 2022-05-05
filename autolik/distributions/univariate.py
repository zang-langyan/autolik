from numbers import Real
from autolik.Dual.benchmark import *
import scipy.special
import math

class pdf:
    r"""Univariate distributions probability density functions library"""
    def beta(x,beta,gam):
        r"""Beta distribution :math:`X \sim beta(\beta,\gamma)`
        
        .. math::
            f(x) = \dfrac{\Gamma(\beta+\gamma)x^{\beta-1}(1-x)^{\gamma-1}}{\Gamma{\beta}\Gamma{\gamma}} , 0<x<1

        with positive shape parameters :math:`\beta > 0`, :math:`\gamma > 0`
        """
        assert 0 < x < 1, "'x' out of range"
        assert beta > 0 and gam > 0, "Wrong paramterization"
        if isinstance(beta,Dual) and isinstance(gam,Dual):
            return (beta+gam).gamma() * x ** (beta-1) * (1-x) ** (gam - 1)/(beta.gamma()*gam.gamma())
        elif isinstance(beta,Real) and isinstance(gam,Dual):
            return (gam+beta).gamma() * x ** (beta-1) * (1-x) ** (gam - 1)/(scipy.special.gamma(beta)*gam.gamma())
        elif isinstance(beta,Dual) and isinstance(gam,Real):
            return (beta+gam).gamma() * x ** (beta-1) * (1-x) ** (gam - 1)/(beta.gamma()*scipy.special.gamma(gam))
        elif isinstance(beta,Real) and isinstance(gam,Real):
            return scipy.special.gamma(beta+gam) * x ** (beta-1) * (1-x) ** (gam - 1)/(scipy.special.gamma(beta)*scipy.special.gamma(gam))

    def cauchy(x,a,alpha):
        r"""Cauchy distribution :math:`X \sim Cauchy(a,\alpha)`

        .. math::
            f(x) = \dfrac{1}{\alpha \pi (1+((x-a)/\alpha)^2)} , -\infty<x<\infty

        with location parameter :math:`a \in (-\infty,\infty)`, and positive scale parameter :math:`\alpha > 0`
        """
        assert alpha > 0, "Wrong paramterization"
        denom = alpha * math.pi * (1 + ((x - a)/alpha) ** 2)
        return 1 / denom

    def chi(x,n):
        r"""Chi distribution :math:`X \sim \chi(n)`

        .. math::
            f(x) = \dfrac{1}{2^{\frac{n}{2}-1} \Gamma(\frac{n}{2})}x^{n-1}e^{-\frac{x^2}{2}} , x > 0

        with :math:`n` degrees of freedom 
        """
        assert x > 0, "'x' out of range"
        assert n > 0, "Wrong paramterization"
        if isinstance(n,Dual):
            return 1 / (2**(n/2-1) * (n/2).gamma()) * x**(n-1) * math.exp(-x**2/2)
        elif isinstance(n,Real):
            return 1 / (2**(n/2-1) * scipy.special.gamma(n/2)) * x**(n-1) * math.exp(-x**2/2)

    def chisqr(x,n):
        r"""Chi-square distribution :math:`X \sim \chi^2(n)`

        .. math::
            f(x) = \dfrac{x^{\frac{n}{2}}e^{-\frac{x}{2}}}{2^{\frac{n}{2}} \Gamma(\frac{n}{2})} , x > 0

        with :math:`n` degrees of freedom 
        """
        assert x > 0, "'x' out of range"
        assert n > 0, "Wrong paramterization"
        if isinstance(n,Dual):
            return x**(n/2-1) * math.exp(-x/2) / (2**(n/2) * (n/2).gamma())
        elif isinstance(n,Real):
            return x**(n/2-1) * math.exp(-x/2) / (2**(n/2) * scipy.special.gamma(n/2))

    def exponential(x,lam):
        r"""Exponential distribution :math:`X \sim exponential(\alpha)`

        .. math::
            f(x) = \lambda e^{-\lambda x} , x > 0

        with rate :math:`\lambda`
        """
        assert x > 0, "'x' out of range"
        assert lam > 0, "Wrong paramterization"
        if isinstance(lam,Dual):
            return lam * (-x * lam).exp()
        elif isinstance(lam,Real):
            return lam * math.exp(-x * lam)

    def gamma(x,alpha,beta):
        r"""Gamma distribution :math:`X \sim gamma(\alpha,\beta)`

        .. math::
            f(x) = \dfrac{x^{\beta-1} e^{-\frac{x}{\alpha}}}{\alpha^{\beta} \Gamma{\beta}} , x > 0

        with positive scale parameter :math:`\alpha > 0` and positive shape parameter :math:`\beta > 0`
        """
        assert x > 0, "'x' out of range"
        assert alpha > 0 and beta > 0, "Wrong paramterization"
        if isinstance(alpha,Dual) and isinstance(beta,Dual):
            return x**(beta-1) * (-x/alpha).exp() / (alpha**beta * beta.gamma())
        elif isinstance(alpha,Real) and isinstance(beta,Dual):
            return x**(beta-1) * math.exp(-x/alpha) / (alpha**beta * beta.gamma())
        elif isinstance(alpha,Dual) and isinstance(beta,Real):
            return x**(beta-1) * (-x/alpha).exp() / (alpha**beta * scipy.special.gamma(beta))
        elif isinstance(alpha,Real) and isinstance(beta,Real):
            return x**(beta-1) * math.exp(-x/alpha) / (alpha**beta * scipy.special.gamma(beta))

    def Ggamma(x,alpha,beta,gam):
        r"""Generalized Gamma distribution :math:`X \sim \text{generalized gamma}(\alpha,\beta,\gamma)`

        .. math::
            f(x) = \dfrac{\gamma x^{\gamma \beta-1} e^{-(x/ \alpha)^{\gamma}}}{\alpha^{\gamma \beta} \Gamma{\beta}} , x > 0

        with positive scale parameter :math:`\alpha > 0` and positive shape parameter :math:`\beta > 0` and :math:`\gamma > 0`
        """
        assert x > 0, "'x' out of range"
        assert alpha > 0 and beta > 0 and gam > 0, "Wrong paramterization"
        if isinstance(beta,Dual):
            if isinstance(alpha,Real) and isinstance(gam,Real):
                numer = gam * x**(gam*beta-1) * math.exp(-(x/alpha)**gam)
                denom = alpha**(gam*beta) * beta.gamma()
                return numer / denom
            else:
                numer = gam * x**(gam*beta-1) * (-(x/alpha)**gam).exp()
                denom = alpha**(gam*beta) * beta.gamma()
                return numer / denom
        elif isinstance(beta,Real):
            if isinstance(alpha,Real) and isinstance(gam,Real):
                numer = gam * x**(gam*beta-1) * math.exp(-(x/alpha)**gam)
                denom = alpha**(gam*beta) * scipy.special.gamma(beta)
                return numer / denom
            else:
                numer = gam * x**(gam*beta-1) * (-(x/alpha)**gam).exp()
                denom = alpha**(gam*beta) * scipy.special.gamma(beta)
                return numer / denom

    def Gpareto(x,delta,kappa,gam):
        r"""Generalized Pareto distribution :math:`X \sim \text{generalized pareto}(\delta,\kappa,\gamma)`

        .. math::
            f(x) = (\gamma + \dfrac{\kappa}{x+\delta})(1+\dfrac{x}{\delta})^{-\kappa} e^{-\gamma x} , x > 0

        with shape parameters :math:`\delta > 0`, :math:`\kappa \ge -\delta \gamma` and :math:`\gamma \ge 0`
        """
        assert x > 0, "'x' out of range"
        assert delta > 0 and gam >= 0 and kappa >= -delta*gam, "Wrong paramterization"
        if isinstance(gam,Dual):
            return (gam+kappa/(x+delta)) * (1+x/delta)**(-kappa) * (-gam*x).exp()
        else:
            return (gam+kappa/(x+delta)) * (1+x/delta)**(-kappa) * math.exp(-gam*x)

    def invgaussian(x,lam,mu):
        r"""Inverse Gaussian distribution :math:`X \sim \text{inverse Gaussian}(\lambda, \mu)`

        .. math::
            f(x) = \sqrt{\dfrac{\lambda}{2\pi x^3}} e^{-\dfrac{\lambda(x-\mu)^2}{2x\mu^2}} , x > 0

        with parameters :math:`\lambda > 0` and :math:`\mu > 0`
        """
        assert x > 0, "'x' out of range"
        assert lam > 0 and mu > 0, "Wrong paramterization"
        if isinstance(lam,Dual):
            return (lam / 2 * math.pi * x**3).sqrt() * (-lam*(x-mu)**2/2*x*mu**2).exp()
        elif isinstance(lam,Real):
            if isinstance(mu,Dual):
                return math.sqrt(lam / 2 * math.pi * x**3) * (-lam*(x-mu)**2/2*x*mu**2).exp()
            elif isinstance(mu,Real):
                return math.sqrt(lam / 2 * math.pi * x**3) * math.exp(-lam*(x-mu)**2/2*x*mu**2)

    def invgamma(x,alpha,beta):
        r"""Inverted Gamma distribution :math:`X \sim \text{inverted gamma}(\alpha,\beta)`

        .. math::
            f(x) = \dfrac{x^{-(\alpha+1) e^{-\frac{1}{\beta x}}}}{\Gamma(\alpha)\beta^{\alpha}} , x > 0

        with positive shape parameter :math:`\alpha > 0` and positive scale parameter :math:`\beta > 0`
        """
        assert x > 0, "'x' out of range"
        if isinstance(alpha,Dual) and isinstance(beta,Dual):
            return x**(-(alpha+1)) * (-1/(beta*x)).exp() / (alpha.gamma() * beta**alpha)
        elif isinstance(alpha,Real) and isinstance(beta,Dual):
            return x**(-(alpha+1)) * (-1/(beta*x)).exp() / (scipy.special.gamma(alpha) * beta**alpha)
        elif isinstance(alpha,Dual) and isinstance(beta,Real):
            return x**(-(alpha+1)) * math.exp(-1/(beta*x)) / (alpha.gamma() * beta**alpha)
        elif isinstance(alpha,Real) and isinstance(beta,Real):
            return x**(-(alpha+1)) * math.exp(-1/(beta*x)) / (scipy.special.gamma(alpha) * beta**alpha)            

    def laplace(x,alpha1,alpha2):
        r"""Laplace distribution :math:`X \sim Laplace(\alpha_1,\alpha_2)`

        .. math::
            f(x) = \begin{cases} (1/(\alpha_1 + \alpha_2))e^{\frac{x}{\alpha_1}}, x < 0 \\ (1/(\alpha_1 + \alpha_2))e^{-\frac{x}{\alpha_2}}, x \ge 0 \end{cases}

        with positive scale parameters :math:`\alpha_1 > 0` and :math:`\alpha_2 > 0`
        """
        assert alpha1 > 0 and alpha2 > 0, "Wrong paramterization"
        if x < 0:
            if isinstance(alpha1,Dual):
                return (1/(alpha1+alpha2)) * (x/alpha1).exp()
            elif isinstance(alpha1,Real):
                return (1/(alpha1+alpha2)) * math.exp(x/alpha1)
        else:
            if isinstance(alpha2,Dual):
                return (1/(alpha1+alpha2)) * (-x/alpha2).exp()
            elif isinstance(alpha2,Real):
                return (1/(alpha1+alpha2)) * math.exp(-x/alpha2)

    def loggamma(x,alpha,beta):
        r"""Log-Gamma distribution :math:`X \sim log-gamma(\alpha,\beta)`

        .. math::
            f(x) = \dfrac{e^{\beta x} e^{-\frac{e^x}{\alpha}}}{\alpha^{\beta} \Gamma(\beta)}, -\infty < x < \infty

        with positive scale parameter :math:`\alpha > 0` and positive shape parameter :math:`\beta > 0`
        """
        if isinstance(alpha,Dual) and isinstance(beta,Dual):
            return (beta*x).exp() * (-math.exp(x)/alpha).exp() / (alpha**beta * beta.gamma())
        elif isinstance(alpha,Real) and isinstance(beta,Dual):
            return (beta*x).exp() * math.exp(-math.exp(x)/alpha) / (alpha**beta * beta.gamma())
        elif isinstance(alpha,Dual) and isinstance(beta,Real):
            return math.exp(beta*x) * (-math.exp(x)/alpha).exp() / (alpha**beta * scipy.special.gamma(beta))
        elif isinstance(alpha,Real) and isinstance(beta,Real):
            return math.exp(beta*x) * math.exp(-math.exp(x)/alpha) / (alpha**beta * scipy.special.gamma(beta))

    def loglogistic(x,lam,kappa):
        r"""Log-Logistic distribution :math:`X \sim loglogistic(\lambda,\kappa)`

        .. math::
            f(x) = \dfrac{\lambda \kappa (\lambda x)^{\kappa-1}}{(1+(\lambda x)^{\kappa})^2}, x > 0

        with positive scale parameter :math:`\lambda > 0` and positive shape parameter :math:`\kappa > 0`
        """
        assert x > 0, "'x' out of range"
        assert lam > 0 and kappa > 0, "Wrong paramterization"
        return lam*kappa*(lam*x)**(kappa-1) / (1+(lam*x)**kappa)**2

    def lognormal(x,alpha,beta):
        r"""Log-normal distribution :math:`X \sim log-normal(\alpha,\beta)`

        .. math::
            f(x) = \dfrac{1}{x\beta \sqrt{2\pi}} e^{-\frac{1}{2}(ln(x/ \alpha)/ \beta)^2}, x > 0

        with positive parameters :math:`\alpha > 0`, :math:`\beta > 0`
        """
        assert x > 0, "'x' out of range"
        assert alpha > 0 and beta > 0, "Wrong paramterization"
        if isinstance(alpha,Dual):
            return 1/(x*beta*math.sqrt(2*math.pi)) * (-1/2*((x/alpha).log()/beta)**2).exp()
        elif isinstance(alpha,Real):
            if isinstance(beta,Dual):
                return 1/(x*beta*math.sqrt(2*math.pi)) * (-1/2*(math.log(x/alpha)/beta)**2).exp()
            elif isinstance(beta,Real):
                return 1/(x*beta*math.sqrt(2*math.pi)) * math.exp(-1/2*(math.log(x/alpha)/beta)**2)

    def logistic(x,lam,kappa):
        r"""Logistic distribution :math:`X \sim logistic(\lambda,\kappa)`

        .. math::
            f(x) = \dfrac{\lambda^{\kappa}\kappa e^{\kappa x}}{(1+(\lambda e^x)^{\kappa})^2}, -\infty < x < \infty

        with positive scale parameter :math:`\lambda > 0` and positive shape parameter :math:`\kappa > 0`
        """
        if isinstance(kappa,Dual):
            return lam**kappa * kappa * (kappa*x).exp() / (1+(lam*math.exp(x))**kappa)**2
        elif isinstance(kappa,Real):
            return lam**kappa * kappa * math.exp(kappa*x) / (1+(lam*math.exp(x))**kappa)**2

    def logistic_exp(x,alpha,beta):
        r"""Logistic-Exponential distribution :math:`X \sim log-exponential(\alpha,\beta)`

        .. math::
            f(x) = \dfrac{\alpha \beta (e^{\alpha x} - 1)^{\beta-1} e^{\alpha x}}{(1+(e^{\alpha x} - 1)^{\beta})^2}, x > 0

        with positive scale parameter :math:`\alpha > 0` and positive shape parameter :math:`\bata > 0`
        """
        assert x > 0, "'x' out of range"
        assert alpha > 0 and beta > 0, "Wrong paramterization"
        if isinstance(alpha,Dual):
            return alpha*beta*((alpha*x).exp()-1)**(beta-1)*(alpha*x).exp() / (1+((alpha*x).exp()-1)**beta)**2
        elif isinstance(alpha,Real):
            return alpha*beta*(math.exp(alpha*x)-1)**(beta-1)*math.exp(alpha*x) / (1+(math.exp(alpha*x)-1)**beta)**2

    def lomax(x,lam,kappa):
        r"""Lomax distribution :math:`X \sim lomax(\lambda,\kappa)`

        .. math::
            f(x) = \dfrac{\lambda \kappa}{(1+\lambda x)^{\kappa + 1}}, x > 0

        with positive scale parameter :math:`\lambda > 0` and positive shape parameter :math:`\kappa > 0`
        """
        assert x > 0, "'x' out of range"
        assert lam > 0 and kappa > 0, "Wrong paramterization"
        return lam*kappa / (1+lam*x)**(kappa+1)

    def makeham(x,delta,kappa,gam):
        r"""Makeham distribution :math:`X \sim Makeham(\delta,\kappa,\gamma)`

        .. math::
            f(x) = (\gamma + \delta \kappa^x) e^{-\gamma x -\delta(\kappa^x -1)/ln(\kappa)}, x > 0

        with positive parameters :math:`\delta > 0` :math:`\kappa > 0` :math:`\gamma > 0`
        """
        assert x > 0, "'x' out of range"
        assert delta > 0 and kappa > 1 and gam > 0, "Wrong paramterization"
        if isinstance(kappa,Real):
            if isinstance(delta,Real) and isinstance(gam,Real):
                return (gam+delta*kappa**x) * math.exp(-gam*x-delta*(kappa**x-1)/math.log(kappa))
            else:
                return (gam+delta*kappa**x) * (-gam*x-delta*(kappa**x-1)/math.log(kappa)).exp()
        elif isinstance(kappa,Dual):
            return (gam+delta*kappa**x) * (-gam*x-delta*(kappa**x-1)/kappa.log()).exp()

    def minimax(x,beta,gam):
        r"""Minimax distribution :math:`X \sim minimax(\beta,\gamma)`

        .. math::
            f(x) = \beta \gamma x^{\beta-1} (1-x^{\beta})^{\gamma-1} , 0 < x < 1

        with positive shape parameters :math:`\beta > 0` and :math:`\gamma > 0`
        """
        assert 0 < x < 1, "'x' out of range"
        assert beta > 0 and gam > 0, "Wrong paramterization"
        return beta * gam * x**(beta-1) * (1-x**beta)**(gam-1)

    def muth(x,kappa):
        r"""Muth distribution :math:`X \sim muth(\kappa)`

        .. math::
            f(x) = (e^{\kappa x} - \kappa) e^{-\frac{e^{\kappa x}}{\kappa} + \kappa x + \frac{1}{\kappa}}, x > 0

        with parameter :math:`0 < \kappa \le 1`
        """
        assert x > 0, "'x' out of range"
        assert 0 < kappa <= 1, "Wrong paramterization"
        if isinstance(kappa,Dual):
            return ((kappa*x).exp() - kappa) * (-(kappa*x).exp()/kappa + kappa*x + 1/kappa).exp()
        elif isinstance(kappa,Real):
            return (math.exp(kappa*x) - kappa) * math.exp(-math.exp(kappa*x)/kappa + kappa*x + 1/kappa)

    def normal(x,mu,sigma):
        r"""Gaussian distribution :math:`X \sim N(\mu,\sigma^2)`

        .. math::
            f(x) = \dfrac{1}{\sqrt{2\pi} \sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}, -\infty < x < \infty

        with mean :math:`-\infty < \mu < \infty` and variance :math:`\sigma^2`
        """
        assert sigma > 0, "Wrong paramterization"
        if isinstance(mu,Real) and isinstance(sigma,Real):
            return 1/(math.sqrt(2*math.pi)*sigma) * math.exp(-(x-mu)**2/(2*sigma**2))
        elif isinstance(mu,Dual) or isinstance(sigma,Dual):
            return 1/(math.sqrt(2*math.pi)*sigma) * (-(x-mu)**2/(2*sigma**2)).exp()

    def pareto(x,lam,kappa):
        r"""Pareto distribution :math:`X \sim pareto(\lambda,\kappa)`

        .. math::
            f(x) = \dfrac{\kappa \lambda^{\kappa}}{x^{\kappa+1}}, x > \lambda

        with positive parameters :math:`\lambda > 0` and :math:`\kappa > 0`
        """
        assert x > lam, "'x' out of range"
        assert lam > 0 and kappa > 0, "Wrong paramterization"
        return kappa * lam**kappa / x**(kappa+1)

    def power(x,alpha,beta):
        r"""Power distribution :math:`X \sim power(\alpha,\beta)`

        .. math::
            f(x) = \dfrac{\beta x^{\beta-1}}{\alpha^{\beta}}, 0 < x < \alpha

        with positive scale parameter :math:`\alpha > 0` and positive shape parameter :math:`\beta > 0`
        """
        assert 0 < x < alpha, "'x' out of range"
        assert alpha > 0 and beta > 0, "Wrong paramterization"
        return beta * x**(beta-1) / alpha**beta
    
    def std_power(x,beta):
        r"""Standard Power distribution :math:`X \sim power(1,\beta)`

        .. math::
            f(x) = \beta x^{\beta-1}, 0 < x < 1

        with positive shape parameter :math:`\beta > 0`
        """
        assert 0 < x < 1, "'x' out of range"
        assert beta > 0, "Wrong paramterization"
        return beta * x**(beta-1)

    def rayleigh(x,alpha):
        r"""Rayleigh distribution :math:`X \sim Rayleigh(\alpha)`

        .. math::
            f(x) = \dfrac{2x e^{-\frac{x^2}{\alpha}}}{\alpha}, x > 0

        with positive parameter :math:`\alpha > 0`
        """
        assert x > 0, "'x' out of range"
        assert alpha > 0, "Wrong paramterization"
        if isinstance(alpha,Dual):
            return 2*x * (-x**2/alpha).exp() / alpha
        elif isinstance(alpha,Real):
            return 2*x * math.exp(-x**2/alpha) / alpha

    def std_wald(x,lam):
        r"""Standard Wald distribution :math:`X \sim standard-Wald(\lambda)`

        .. math::
            f(x) = \sqrt{\dfrac{\lambda}{2\pi x^3}} e^{-\frac{\lambda(x-1)^2}{2x}}, x > 0

        with parameter :math:`\lambda > 0`
        """
        assert x > 0, "'x' out of range"
        assert lam > 0, "Wrong paramterization"
        if isinstance(lam, Dual):
            return (lam/(2*math.pi*x**3)).sqrt() * (-lam*(x-1)**2/(2*x)).exp()
        elif isinstance(lam,Real):
            return math.sqrt(lam/(2*math.pi*x**3)) * math.exp(-lam*(x-1)**2/(2*x))

    def Tdist(x,n):
        r"""Student t distribution :math:`X \sim t(n)`

        .. math::
            f(x) = \dfrac{\Gamma(\frac{n+1}{2}) (1+\frac{x^2}{n})^{-\frac{n+1}{2}}}{\sqrt{n\pi} \Gamma{\frac{n}{2}}}, -\infty < x < \infty

        with :math:`n` degrees of freedom
        """
        assert n > 0, "Wrong paramterization"
        if isinstance(n,Dual):
            return ((n+1)/2).gamma() * (1+x**2/n)**(-(n+1)/2) / ((n*math.pi).sqrt() * (n/2).gamma())
        elif isinstance(n,Real):
            return scipy.special.gamma((n+1)/2) * (1+x**2/n)**(-(n+1)/2) / (math.sqrt(n*math.pi) * scipy.special.gamma(n/2))

    def uniform(x,a,b):
        r"""Uniform distribution :math:`X \sim U(a,b)`

        .. math::
            f(x) = \dfrac{1}{b-a}, a < x < b

        with range parameters :math:`-\infty < a < b < \infty`
        """
        assert a < x < b, "'x' out of range"
        return 1 / (b - a)

    def weibull(x,alpha,beta):
        r"""Weibull distribution :math:`X \sim Weibull(\alpha,\beta)`

        .. math::
            f(x) = \dfrac{\beta}{\alpha} x^{\beta-1} e^{-\frac{x^{\beta}}{\alpha}}, x > 0

        with positive scale parameter :math:`\alpha > 0` and positive shape parameter :math:`\beta > 0`
        """
        assert x > 0, "'x' out of range"
        assert alpha > 0 and beta > 0, "Wrong paramterization"
        if isinstance(alpha,Dual) or isinstance(beta,Dual):
            return beta/alpha * x**(beta-1) * (-(1/alpha) * x**beta).exp()
        elif isinstance(alpha,Real) and isinstance(beta,Real):
            return beta/alpha * x**(beta-1) * math.exp(-(1/alpha) * x**beta)

    ####################################
    # not supported for Dual computation
    ####################################
    def std_cauchy(x):
        return 1 / (math.pi * (1+x**2))

    def std_normal(x):
        return math.exp(-x**2/2) / math.sqrt(2*math.pi)

    def std_uniform(x):
        assert 0 < x < 1, "'x' out of range"
        return 1.

# def main():
#     print(pdf.normal(0,0,1))

# if __name__ == "__main__":
#     main()