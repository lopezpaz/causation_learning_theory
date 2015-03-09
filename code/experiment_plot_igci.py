from   scipy.interpolate import UnivariateSpline
from   matplotlib import gridspec
import matplotlib.pyplot as plt
from   scipy.stats import gaussian_kde
import numpy as np

np.random.seed(0)

x  = np.sort(np.random.rand(1000))

gx = np.sort(np.linspace(min(x),max(x),10))
gy = np.sort(np.random.randn(10))
f  = UnivariateSpline(gx,gy)
y  = f(x)
dy = gaussian_kde(y)

plt.figure(figsize=(5,4))

plot_grid = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[1,5],
wspace=0, hspace=0)

plot_main = plt.subplot(plot_grid[1,0])
plot_left = plt.subplot(plot_grid[1,1], sharey=plot_main)
plot_down = plt.subplot(plot_grid[0,0], sharex=plot_main)

ey = np.linspace(min(y)-1,max(y)+1,100)

plot_main.plot(x,y)

plot_left.plot(dy(ey),ey)
plot_left.axis('off')

plot_down.plot(np.hstack((0,0,x,1,1)),np.hstack((0,1,np.ones(x.shape),1,0)))
plot_down.axis('off')

plot_main.axes.get_xaxis().set_ticks([])
plot_main.axes.get_yaxis().set_ticks([])
plot_main.set_xlim([-0.05,1.05])

plot_down.text(-0.15, 0.4, '$p(x)$', fontsize=12)
plot_left.text(1, 0.4, '$p(y)$', fontsize=12)
plot_main.text(0.6, -0.70, '$y \\leftarrow f(x)$', fontsize=12)

plot_main.axes.set_xlabel('$x$')
plot_main.axes.set_ylabel('$y$')

plt.savefig('plot_igci.pdf', bbox_inches='tight')
