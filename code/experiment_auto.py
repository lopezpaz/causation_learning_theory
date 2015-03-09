ABALONE_X = "/agbs/datasets/causal_pairs_20000/auto.data"

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

def graph0(n,k,p1,p2,v,d):
  x = cause(n,k,p1,p2)
  y = cause(n,k,p1,p2)
  z = cause(n,k,p1,p2)
  return ((x,y,z),(0,0,0))

def graph1(n,k,p1,p2,v,d):
  x = cause(n,k,p1,p2)
  y = scale(scale(mechanism(x,d))+noise(n,v))
  z = cause(n,k,p1,p2)
  return ((x,y,z),(1,0,0))

def graph2(n,k,p1,p2,v,d):
  x = cause(n,k,p1,p2)
  y = scale(scale(mechanism(x,d))+noise(n,v))
  z = scale(scale(mechanism(y,d))+noise(n,v))
  return ((x,y,z),(1,1,0))

def graph3(n,k,p1,p2,v,d):
  x = cause(n,k,p1,p2)
  z = cause(n,k,p1,p2)
  y = scale(scale(mechanism(x,d)+mechanism(z,d))+noise(n,v))
  return ((x,y,z),(1,-1,0))

def graph4(n,k,p1,p2,v,d):
  y = cause(n,k,p1,p2)
  x = scale(scale(mechanism(y,d))+noise(n,v))
  z = scale(scale(mechanism(y,d))+noise(n,v))
  return ((x,y,z),(-1,1,0))

def graph5(n,k,p1,p2,v,d):
  x = cause(n,k,p1,p2)
  y = scale(scale(mechanism(x,d))+noise(n,v))
  z = scale(scale(mechanism(x,d)+mechanism(y,d))+noise(n,v))
  return ((x,y,z),(1,1,1))

def graph6(n,k,p1,p2,v,d):
  x = cause(n,k,p1,p2)
  z = scale(scale(mechanism(x,d))+noise(n,v))
  y = scale(scale(mechanism(x,d)+mechanism(z,d))+noise(n,v))
  return ((x,y,z),(1,-1,1))

def graph7(n,k,p1,p2,v,d):
  y = cause(n,k,p1,p2)
  x = scale(scale(mechanism(y,d))+noise(n,v))
  z = scale(scale(mechanism(x,d)+mechanism(y,d))+noise(n,v))
  return ((x,y,z),(-1,1,1))

def graph(n=1000,k=2,p1=3,p2=2,v=3,d=5):
  G = [graph0,graph1,graph2,graph3,graph4,graph5,graph6,graph7]
  i = np.random.randint(len(G))
  ((x,y,z),(l1,l2,l3)) = G[i](n,k,p1,p2,v,d)
  return ((x,y,l1),(y,z,l2),(x,z,l3))

def pairset(N):
  Z1 = np.zeros((3*N,3*wx.shape[1]))
  Z2 = np.zeros((3*N,3*wx.shape[1]))
  L1 = np.ones(3*N)
  L2 = np.ones(3*N)
  for i in range(N):
    ((x,y,l1),(y,z,l2),(x,z,l3)) = graph()
    i1       = 3*i+0
    i2       = 3*i+1
    i3       = 3*i+2
    Z1[i1,:] = f2(x,y,np.hstack((x,y,z))) # xy
    Z2[i1,:] = f2(y,x,np.hstack((y,x,z))) # xy
    Z1[i2,:] = f2(y,z,np.hstack((y,z,x))) # yz
    Z2[i2,:] = f2(z,y,np.hstack((z,y,x))) # yz
    Z1[i3,:] = f2(x,z,np.hstack((x,z,y))) # xz
    Z2[i3,:] = f2(z,x,np.hstack((z,x,y))) # xz
    L1[i1]   = +l1
    L2[i1]   = -l1
    L1[i2]   = +l2
    L2[i2]   = -l2
    L1[i3]   = +l3
    L2[i3]   = -l3
    if(np.mod(i,100)==0): print i
  return (np.vstack((Z1,Z2)),np.hstack((L1,L2)).ravel())

def abalone(reg,fx=ABALONE_X):
  X  = scale(np.loadtxt(fx));
  d  = X.shape[1]
  R1 = np.zeros((d,d))
  R2 = np.zeros((d,d))
  R3 = np.zeros((d,d))
  for i in range(d):
    x = X[:,i][:,np.newaxis]
    # print np.sqrt(2./np.median(np.power(x-x[np.random.permutation(x.shape[0]),:],2)))
    for j in range(i+1,d):
      y = X[:,j][:,np.newaxis]
      for k in list(set(range(d)) - set([i,j])):
        z  = X[:,k][:,np.newaxis]
        # zz = np.hstack((x,y,z))
        # print np.sqrt(2./np.median(np.sum(np.power(zz-zz[np.random.permutation(zz.shape[0]),:],2),1)))
        fa = f2(x,y,np.hstack((x,y,z)))
        (p1,p2,p3) = list(reg.predict_proba(fa)[0])
        R1[i,j] = np.sum((R1[i,j],p1))
        R2[i,j] = np.sum((R2[i,j],p2))
        R3[i,j] = np.sum((R3[i,j],p3))

  M = np.zeros((d,d))
  for i in range(d):
    for j in range(d):
      e = np.argmax((R1[i,j],R2[i,j],R3[i,j]))
      if(e==0):
        M[j,i] = R1[i,j]
      if(e==2):
        M[i,j] = R3[i,j] 

  return (R1,R2,R3,M)

np.random.seed(0)

N = 3000
K = 100 
E = 1000

wx = rp(K,[0.15,1.5,15],1) # 2.60, 0.78
wy = rp(K,[0.15,1.5,15],1)
wz = rp(K,[0.08,0.8,8],3)

(x1,y1) = pairset(N)

reg  = RFC(n_estimators=E,random_state=0,n_jobs=16).fit(x1,y1);

(R1,R2,R3,M) = abalone(reg)
M1 = (M>np.percentile(M,10))*M
M2 = (M>np.percentile(M,20))*M
M3 = (M>np.percentile(M,30))*M
M4 = (M>np.percentile(M,40))*M
M5 = (M>np.percentile(M,50))*M
M6 = (M>np.percentile(M,60))*M
M7 = (M>np.percentile(M,70))*M
M8 = (M>np.percentile(M,80))*M
M9 = (M>np.percentile(M,90))*M

import networkx as nx
import matplotlib.pyplot as plt
G=nx.from_numpy_matrix(M,create_using=nx.MultiDiGraph())

labels={}
labels[0]=r'MPG'
labels[1]=r'CYL'
labels[2]=r'DIS'
labels[3]=r'HP'
labels[4]=r'WEI'
labels[5]=r'ACC'
labels[6]=r'YEA'

pos=nx.circular_layout(G)
nx.draw_networkx_nodes(G,pos,node_size=1000,node_color='white')
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos,labels,font_size=12)
print nx.is_directed_acyclic_graph(G)
print labels
print G.edges()
plt.show()

def plot(M):
  G=nx.from_numpy_matrix(M,create_using=nx.MultiDiGraph())
  
  labels={}
  labels[0]=r'MPG'
  labels[1]=r'CYL'
  labels[2]=r'DIS'
  labels[3]=r'HP'
  labels[4]=r'WEI'
  labels[5]=r'ACC'
  labels[6]=r'YEA'
  
  pos=nx.circular_layout(G)
  nx.draw_networkx_nodes(G,pos,node_size=1000,node_color='white')
  nx.draw_networkx_edges(G,pos)
  nx.draw_networkx_labels(G,pos,labels,font_size=12)
  print nx.is_directed_acyclic_graph(G)
  print labels
  print G.edges()
  plt.show()
