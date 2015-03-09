F_X = "/agbs/datasets/causal_pairs_20000/auto.data"

import sys
sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')

import numpy as np
from sklearn.ensemble      import RandomForestClassifier as RFC
from scipy.interpolate     import UnivariateSpline as sp
from sklearn.preprocessing import scale, StandardScaler
from sklearn.mixture       import GMM

def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),
                    2*np.pi*np.random.rand(k*len(s),1))).T
def f1(x,w):
  return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))
def f2(x,y,z):
  return np.hstack((f1(x,wx).mean(0),f1(y,wy).mean(0),f1(z,wz).mean(0)))

def cause(n,k,p1,p2):
  g = GMM(k)
  g.means_   = p1*np.random.randn(k,1)
  g.covars_  = np.power(abs(p2*np.random.randn(k,1)+1),2)
  g.weights_ = abs(np.random.rand(k,1))
  g.weights_ = g.weights_/sum(g.weights_)
  return scale(g.sample(n))

def noise(n,v):
  return v*np.random.rand(1)*np.random.randn(n,1)

def mechanism(x,d):
  g = np.linspace(min(x)-np.std(x),max(x)+np.std(x),d);
  return sp(g,np.random.randn(d))(x.flatten())[:,np.newaxis]

np.random.seed(0)

N = n = 500
K = 100

wx = rp(K,[0.15,1.5,15],1) # 2.60, 0.78
wy = rp(K,[0.15,1.5,15],1)
wz = rp(K,[0.08,0.8,8],2)

def pair(n,k,p1,p2,v,d):
  x = cause(n,k,p1,p2)
  y = scale(scale(mechanism(x,d))+noise(n,v))
  return (x,y)

def embed_dist(p,q):
  return np.mean(np.power(p-q,2));

def obj(p):
  (k,p1,p2,v,d) = p;
  res = 0
  for i in x_te:
    this_res = 1.e3;
    for j in range(N):
      (x,y) = pair(n,k,p1,p2,v,d)
      e1 = f2(x,y,np.hstack((x,y)))
      e2 = f2(y,x,np.hstack((y,x))) 
      e  = np.min(embed_dist(i,e1),embed_dist(i,e2))
      this_res = min(this_res,e)
    res = res + this_res
  return res

def readdata(fx=F_X):
  X  = scale(np.loadtxt(fx));
  d  = X.shape[1]
  R  = np.zeros((d*(d-1)/2,3*wx.shape[1]))
  k  = 0
  for i in range(d):
    for j in range(i+1,d):
      x = X[:,i][:,np.newaxis]
      y = X[:,j][:,np.newaxis]
      R[k,:] = f2(x,y,np.hstack((x,y)))
      k += 1
  return R

x_te = readdata()

k  = int(sys.argv[1])
p1 = int(sys.argv[2])
p2 = int(sys.argv[3])
v  = int(sys.argv[4])
d  = int(sys.argv[5])

print obj((k,p1,p2,v,d))
