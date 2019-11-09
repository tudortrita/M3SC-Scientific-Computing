"""Simple example from lecture 16 illustrating
2nd-order centered finite difference method
"""

import numpy as np
import matplotlib.pyplot as plt

# generate grid
L = 5
N = 100
x = np.linspace(0, L, N + 1)
x = x[:-1]
h = x[1] - x[0]
hfac = 1 / (2 * h)

# generate function
k = 2 * np.pi * 4 / L
f = np.sin(k * x)

# compute derivative
df = np.zeros_like(f)
df[1:N - 1] = hfac * (f[2:N] - f[0:N - 2])

df[0] = hfac * (f[1] - f[-1])
df[-1] = hfac * (f[0] - f[-2])

# compute exact derivative
df_exact = k * np.cos(k * x)


# compute error
error = np.abs(df - df_exact)
error_total = error.sum() / N


# display stuff
plt.figure()
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')

plt.figure()
plt.plot(x, df, 'x--', x, df_exact)
plt.legend(('computed', 'exact'))
plt.xlabel('x')
plt.ylabel('df/dx')

plt.figure()
plt.semilogy(x, error)
plt.xlabel('x')
plt.ylabel('error')
plt.show()
