import sys
sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')

import numpy               as np
import networkx            as nx
import matplotlib.pyplot   as plt
from sklearn.ensemble      import RandomForestClassifier as RFC
from scipy.interpolate     import UnivariateSpline as sp
from sklearn.preprocessing import scale, StandardScaler
from sklearn.mixture       import GMM
from sklearn.linear_model  import LogisticRegression as LR

np.random.seed(0)

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
  return (np.vstack((z1,z2)),np.hstack((np.ones(N),-np.ones(N))).ravel())

def boston_housing(fname="../data/housing/housing.data"):
    X = np.genfromtxt(fname)

    num_features = X.shape[1]
    num_pairs = int(num_features*(num_features-1)/2);
    
    pairs = np.zeros((num_pairs,3*wx.shape[1]))

    print 'loading boston housing dataset...please wait!'
    k = 0
    for i in range(num_features-1):
        for j in range(i+1,num_features):
            x = scale(np.array(X[:,i]))[:,np.newaxis]
            y = scale(np.array(X[:,j]))[:,np.newaxis]            
            pairs[k,:] = f2(x,y,np.hstack((x,y)))
            k = k + 1

    return (pairs,num_features,num_pairs)
    
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
  m = np.genfromtxt(fy,delimiter=",")[:,]2gma
  return (np.vstack((z1,z2)),np.hstack((y,-y)), np.hstack((m,m)))


# generate random features
wx = rp(333,[0.2,2,20],1)
wy = rp(333,[0.2,2,20],1)
wz = rp(333,[0.2,2,20],2)

# generate training data
print 'generating training data...'
(x,y) = pairset(10000)
#(x_te,y_te,m_te) = tuebingen()

# load test data
(pairs,num_features,num_pairs) = boston_housing()

# train the classifier and predict the test data
print 'training the random forest classifier...'
reg = RFC(n_estimators=100,random_state=0,n_jobs=4).fit(x,y);
y_prob = reg.predict_proba(pairs)

# save the predictive probability of pairs
np.savetxt('housing_predict.txt',y_prob,fmt='%.5f')

# visualize the directed graph
node_labels = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

desp = """ 
1. CRIM      per capita crime rate by town
2. ZN        proportion of residential land zoned for lots over 
             25,000 sq.ft.
3. INDUS     proportion of non-retail business acres per town
4. CHAS      Charles River dummy variable (= 1 if tract bounds 
             river; 0 otherwise)
5. NOX       nitric oxides concentration (parts per 10 million)
6. RM        average number of rooms per dwelling
7. AGE       proportion of owner-occupied units built prior to 1940
8. DIS       weighted distances to five Boston employment centres
9. RAD       index of accessibility to radial highways
10. TAX      full-value property-tax rate per $10,000
11. PTRATIO  pupil-teacher ratio by town
12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
             by town
13. LSTAT    % lower status of the population
14. MEDV     Median value of owner-occupied homes in $1000's
"""

G = nx.DiGraph()
G.add_nodes_from(node_labels)

k = 0
edges = []
for i in range(num_features-1):
  for j in range(i+1,num_features):
    if (y_prob[k,0] > y_prob[k,1] and y_prob[k,0] > 0.65):
      edges.append((node_labels[i],node_labels[j],{'weight':y_prob[k,0]}))
    elif (y_prob[k,0] < y_prob[k,1] and y_prob[k,1] > 0.65):
      edges.append((node_labels[j],node_labels[i],{'weight':y_prob[k,1]}))

    k = k + 1

G.add_edges_from(edges)

pos = nx.circular_layout(G,scale=3)
nx.draw(G, pos, cmap=plt.get_cmap('jet'), node_size=1700, node_color='b', alpha=0.95)
nx.draw_networkx_labels(G, pos, font_color='w')
nx.draw_networkx_edges(G, pos, arrows=True)
plt.text(1.5, 0.2,desp,fontsize=8)
plt.savefig('boston_housing.eps')
#plt.show()

print 'done.'
