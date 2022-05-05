|api|Distribution|
|---|---|
|`beta`|[$Beta$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Beta.pdf)|
|`cauchy`|[$Cauchy$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Cauchy.pdf)|
|`chi`|[$Chi$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Chi.pdf)|
|`chisqr`|[$Chi^2$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Chisquare.pdf)|
|`exponential`|[$Exponential$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Exponential.pdf)|
|`gamma`|[$Gamma$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Gamma.pdf)|
|`Ggamma`|[$Generalized \ Gamma$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Generalizedgamma.pdf)|
|`Gpareto`|[$Generalized \ Pareto$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Generalizedpareto.pdf)|
|`invgaussian`|[$Inverse \ Gaussian$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Inversegaussian.pdf)|
|`invgamma`|[$Inverted \ Gamma$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Invertedgamma.pdf)|
|`laplace`|[$Laplace$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Laplace.pdf)|
|`loggamma`|[$Log-Gamma$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Loggamma.pdf)|
|`loglogistic`|[$Log-Logistic$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Loglogistic.pdf)|
|`lognormal`|[$Log-Normal$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Lognormal.pdf)|
|`logistic`|[$Logistic$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Logistic.pdf)|
|`logistic_exp`|[$Logistic-Exponential$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Logisticexponential.pdf)|
|`lomax`|[$Lomax$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Lomax.pdf)|
|`makeham`|[$Makeham$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Makeham.pdf)|
|`minimax`|[$Minimax$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Minimax.pdf)|
|`muth`|[$Muth$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Muth.pdf)|
|`normal`|[$Gaussian$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Normal.pdf)|
|`pareto`|[$Pareto$](math.wm.edu/~leemis/chart/UDR/PDFs/Pareto.pdf)|
|`power`|[$Power$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Power.pdf)|
|`std_power`|[$Standard \ Power$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Standardpower.pdf)|
|`rayleigh`|[$Rayleigh$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Rayleigh.pdf)|
|`std_wald`|[$Standard \ Wald$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Standardwald.pdf)|
|`Tdist`|[$Student  \ t$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/T.pdf)|
|`uniform`|[$Uniform$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Uniform.pdf)|
|`weibull`|[$Weibull$](http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Weibull.pdf)|

**_Usage:_**
```python
>>> import autolik
>>> autolik.pdf.normal(x = 0, mu = 0, sigma = 1)
0.3989422804014327
``` 

```python
>>> y = [0.0,1.5,0.5,-1.,0.8,-2.4]
>>> autolik.ll.normal(y,mu=0,sigma=1)
-10.463631199228036
```