import sys
sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import scale

def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),

                    2*np.pi*np.random.rand(k*len(s),1))).T
def f1(x,w):
  return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))

def series_file(fx,N=-1):
  f  = open(fx); pairs = f.readlines(); f.close();
  if(N==-1):
    N  = len(pairs)
  z0 = np.zeros((N,wz.shape[1]))
  z1 = np.zeros((N,wz.shape[1]))
  i  = 0
  for row in pairs:
    r = scale(np.array(row.split(","),dtype=np.float))[:,np.newaxis]
    x = r[:-1]
    y = r[1:]
    z0[i,:] = f1(np.hstack((x,y)),wz).mean(0)
    z1[i,:] = f1(np.hstack((y,x)),wz).mean(0)
    i = i+1
    if(i==N): break
  return (np.vstack((z0,z1)),np.hstack((np.ones(N),-np.ones(N))))

np.random.seed(0)

N = int(sys.argv[1])
K = int(sys.argv[2])
E = int(sys.argv[3])
    
wz = rp(K,[0.15,1.5,15],2) # 0.6

(x1,y1) = series_file('data/series.csv',N)
(x2,y2) = series_file('data/series_va.csv')
(x0,y0) = series_file('data/series_te.csv')

reg = RFC(random_state=0,n_jobs=16,n_estimators=E).fit(x1,y1);
print [N,K,E,reg.score(x1,y1),reg.score(x2,y2),reg.score(x0,y0)]
