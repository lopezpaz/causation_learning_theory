F_X_TR = "/agbs/datasets/causal_pairs_20000/train_pairs.csv"
F_Y_TR = "/agbs/datasets/causal_pairs_20000/train_target.csv"
F_X_TE = "/agbs/datasets/causal_pairs_20000/test_pairs.csv"
F_Y_TE = "/agbs/datasets/causal_pairs_20000/test_target.csv"

import numpy as np
import sys
sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')
from   sklearn.preprocessing import scale
from   sklearn.ensemble      import RandomForestClassifier     as CLF 
from   sklearn.metrics       import roc_auc_score
from   sklearn.grid_search   import GridSearchCV
from   scipy.stats           import skew, kurtosis, rankdata

def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),
                    2*np.pi*np.random.rand(k*len(s),1))).T
def f1(x,w):
  return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))
def score(y,p):
  return (roc_auc_score(y==1,p)+roc_auc_score(y==-1,-p))/2

def featurize_row(row,i,j):
  r  = row.split(",",2)
  x  = scale(np.array(r[i].split(),dtype=np.float))[:,np.newaxis]
  y  = scale(np.array(r[j].split(),dtype=np.float))[:,np.newaxis]
  d  = np.hstack((f1(x,wx).mean(0),f1(y,wy).mean(0),f1(np.hstack((x,y)),wz).mean(0)))
  return d

def featurize(filename):
  f = open(filename);
  pairs = f.readlines();
  f.close();
  return np.vstack((np.array([featurize_row(row,1,2) for row in pairs]),
                    np.array([featurize_row(row,2,1) for row in pairs])))

np.random.seed(0)

K = int(sys.argv[1])
E = int(sys.argv[2])
L = int(sys.argv[3])

wx = rp(K,[0.15,1.5,15],1)
wy = rp(K,[0.15,1.5,15],1)
wz = rp(K,[0.15,1.5,15],2)

x_tr = featurize(F_X_TR)
x_te = featurize(F_X_TE)
y_tr = np.genfromtxt(F_Y_TR, delimiter=",")[:,1]
y_te = np.genfromtxt(F_Y_TE, delimiter=",")[:,1]
d_tr = (np.genfromtxt(F_Y_TR, delimiter=",")[:,2])==4
d_te = (np.genfromtxt(F_Y_TE, delimiter=",")[:,2])==4
y_tr = np.hstack((y_tr,-y_tr))
y_te = np.hstack((y_te,-y_te))
d_tr = np.hstack((d_tr,d_tr))
d_te = np.hstack((d_te,d_te))
x_ab = x_tr[(y_tr==1)|(y_tr==-1)]
y_ab = y_tr[(y_tr==1)|(y_tr==-1)]

params = { 'random_state' : 0, 'n_estimators'      : E, 'max_features' : None, 
           'max_depth'    : 50, 'min_samples_leaf' : 10, 'verbose' : 10 }

params = { 'random_state' : 0, 'n_estimators' : E, 'min_samples_leaf' : L, 'n_jobs' : 16 }

clf0 = CLF(**params).fit(x_tr,y_tr!=0) # causal or confounded?
clf1 = CLF(**params).fit(x_ab,y_ab==1) # causal or anticausal?
clfd = CLF(**params).fit(x_tr,d_tr)    # dependent or independent?

p_te = clf0.predict_proba(x_te)[:,1]*(2*clf1.predict_proba(x_te)[:,1]-1)

print([score(y_te,p_te),clf0.score(x_te,y_te!=0),clfd.score(x_te,d_te)])
