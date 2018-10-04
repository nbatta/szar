import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def f(x):
    integral = 
    return 0.5*np.exp(x**2/2)

g = lambda x: x**2

xs = np.linspace(0.01, 2 ,100)
fs = f(xs)

logderiv = np.gradient(np.log(fs), np.log(xs), edge_order=2)

plt.plot(xs,logderiv, label="numerical log deriv")
plt.plot(xs, g(xs), label="analytical log deriv")
plt.xlabel(r'$x$')
plt.ylabel(r'$\frac{d}{d(\ln(x))} \, e^{x^2/2}$')
plt.legend(loc='upper left')
plt.savefig("python_logderiv_test.svg")

