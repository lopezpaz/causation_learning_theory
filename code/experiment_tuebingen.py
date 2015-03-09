import sys
sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')

import numpy as np
from sklearn.ensemble      import RandomForestClassifier as RFC
from scipy.interpolate     import UnivariateSpline as sp
from sklearn.preprocessing import scale, StandardScaler
from sklearn.mixture       import GMM
from sklearn.linear_model import LogisticRegression as LR


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

def pair(n=1000,k=3,p1=2,p2=2,v=2,d=5):
  x  = cause(n,k,p1,p2)
  return (x,scale(scale(mechanism(x,d))+noise(n,v)))

def pairset(N):
  z1 = np.zeros((N,3*wx.shape[1]))
  z2 = np.zeros((N,3*wx.shape[1]))
  for i in range(N):
    (x,y)   = pair()
    z1[i,:] = f2(x,y,np.hstack((x,y)))
    z2[i,:] = f2(y,x,np.hstack((y,x)))
  return (np.vstack((z1,z2)),np.hstack((np.ones(N),-np.ones(N))).ravel(),np.ones(2*N))

def tuebingen(fx="data/tuebingen_pairs.csv",fy="data/tuebingen_target.csv"):
  f  = open(fx); pairs = f.readlines(); f.close();
  z1 = np.zeros((len(pairs),3*wx.shape[1]))
  z2 = np.zeros((len(pairs),3*wx.shape[1])) 
  i  = 0
  for row in pairs:
    r       = row.split(",",2)
    x       = scale(np.array(r[1].split(),dtype=np.float))[:,np.newaxis]
    y       = scale(np.array(r[2].split(),dtype=np.float))[:,np.newaxis]
    z1[i,:] = f2(x,y,np.hstack((x,y)))
    z2[i,:] = f2(y,x,np.hstack((y,x)))
    i       = i+1
  y = 2*((np.genfromtxt(fy,delimiter=",")[:,1])==1)-1
  m = np.genfromtxt(fy,delimiter=",")[:,2]
  return (np.vstack((z1,z2)),np.hstack((y,-y)), np.hstack((m,m)))

np.random.seed(0)

N = int(sys.argv[1])
K = int(sys.argv[2])
E = int(sys.argv[3])

# (1.7,1.9,1.9), (1.5)
wx = rp(K,[0.15,1.5,15],1)
wy = rp(K,[0.15,1.5,15],1)
wz = rp(K,[0.15,1.5,15],2)

(x1,y1,m1) = pairset(N)
(x2,y2,m2) = pairset(N)
(x0,y0,m0) = tuebingen()

reg  = RFC(n_estimators=E,random_state=0,n_jobs=16).fit(x1,y1);
print [N,K,E,reg.score(x1,y1,m1),reg.score(x2,y2,m2),reg.score(x0,y0,m0)]

print np.hstack((y0[:,np.newaxis],m0[:,np.newaxis],reg.predict_proba(x0)))
