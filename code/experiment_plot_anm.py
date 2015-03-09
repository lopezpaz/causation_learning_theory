from   scipy.interpolate import UnivariateSpline
from   matplotlib import gridspec
import matplotlib.pyplot as plt
from   scipy.stats import gaussian_kde
import numpy as np

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

np.random.seed(0)

x  = np.sort(np.random.rand(1000))
gx = np.sort(np.linspace(min(x),max(x),10))
gy = np.sort(np.random.randn(10))
f  = UnivariateSpline(gx,gy)
y  = f(x)
n  = np.random.randn(1000)*0.1

fig = plt.figure(figsize=(5,4))
plot_main = fig.add_subplot(111)

plot_main.plot(x,y+n,'.',color='0.75',zorder=-100)
plot_main.plot(x,y,linewidth=2, zorder=-99)

i = [10,500,800]
plt.errorbar(x[i],y[i],yerr=[0.25,0.25,0.25], color='red',
linestyle='none',linewidth=2)

plt.errorbar(x[i],y[i],xerr=[0.02,0.25,0.04], color='green',
linestyle='none',linewidth=2)

plot_main.get_xaxis().set_ticks([])
plot_main.get_yaxis().set_ticks([])
plot_main.set_xlim([-0.05,1.05])
plot_main.axes.set_xlabel('$x$')
plot_main.axes.set_ylabel('$y$')

plot_main.plot([x[i[1]]-0.25,x[i[1]]-0.25], [y[i[1]],y[i[1]]+0.75],color='green')
plot_main.plot([x[i[1]]+0.25,x[i[1]]+0.25], [y[i[1]],y[i[1]]+0.75],color='green')
plot_main.plot([x[i[0]],x[i[0]]+0.20], [y[i[0]]-0.25,y[i[0]]-0.25], color='red')
plot_main.plot([x[i[0]],x[i[0]]+0.20], [y[i[0]]+0.25,y[i[0]]+0.25], color='red')

plot_main.text(0.6, -0.70, '$y \\leftarrow f(x)+\\epsilon$', fontsize=12)
plot_main.text(0.265, 0.50, r'residual variance at $x$', fontsize=12,color='green')
plot_main.text(0.21, -1.74, r'residual variance at $y$', fontsize=12,color='red')

plt.savefig('plot_anm.pdf', bbox_inches='tight')
