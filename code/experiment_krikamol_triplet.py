##
## Predicting DAG using three random variables

import sys
sys.path.append('/is/ei/dlopez/local/lib/python2.7/site-packages/')

import numpy               as np
import networkx            as nx
import matplotlib.pyplot   as plt
import itertools
from sklearn.ensemble      import RandomForestClassifier as RFC
from sklearn.multiclass    import OneVsRestClassifier as OVR
from scipy.interpolate     import UnivariateSpline as sp
from sklearn.preprocessing import scale, StandardScaler
from sklearn.mixture       import GMM
from sklearn.linear_model  import LogisticRegression as LR

np.random.seed(0)

def rp(k,s,d):
  return np.hstack((np.vstack([si*np.random.randn(k,d) for si in s]),2*np.pi*np.random.rand(k*len(s),1))).T
def f1(x,w):
  return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w))
def f2(x,y,z):
  return np.hstack((f1(x,wx).mean(0),f1(y,wy).mean(0),f1(z,wz).mean(0)))
def f3(x,y,z,xyz):
  return np.hstack((f1(x,wx).mean(0),f1(y,wy).mean(0),f1(z,wz).mean(0),f1(xyz,ww).mean(0)))

# generator of cause variable
def cause(n,k,p1,p2):
  g = GMM(k)
  g.means_   = p1*np.random.randn(k,1)
  g.covars_  = np.power(abs(p2*np.random.randn(k,1)+1),2)
  g.weights_ = abs(np.random.rand(k,1))
  g.weights_ = g.weights_/sum(g.weights_)
  return scale(g.sample(n))

# generator of noise variable
def noise(n,v):
  return v*np.random.rand(1)*np.random.randn(n,1)

# generator of mechanism
def mechanism(x,d):
  g = np.linspace(min(x)-np.std(x),max(x)+np.std(x),d);
  return sp(g,np.random.randn(d))(x.flatten())[:,np.newaxis]

# generator of collider
def collider(x,z,d):
  return mechanism(x,d) + mechanism(z,d)

# triplet generator
def triplet(idx,n=1000,k=2,p1=2,p2=1,v=1,d=5):
  if (idx == 1): # (1) the chain
    x = cause(n,k,p1,p2)
    y = scale(scale(mechanism(x,d))+noise(n,v))
    z = scale(scale(mechanism(y,d))+noise(n,v))
  elif (idx == 2): # (2) the confounder
    y = cause(n,k,p1,p2)
    x = scale(scale(mechanism(y,d))+noise(n,v))
    z = scale(scale(mechanism(y,d))+noise(n,v))
  elif (idx == 3): # (3) the collider
    x = cause(n,k,p1,p2)
    z = cause(n,k,p1,p2)
    y = scale(scale(collider(x,z,d))+noise(n,v))

  return (x,y,z)

def tripletset(N):
  tp = np.zeros((3*N,4*wx.shape[1]))
  for i in range(N):
    (x,y,z) = triplet(1)
    tp[i,:] = f3(x,y,z,np.hstack((x,y,z)))

    (x,y,z) = triplet(2)
    tp[N+i,:] = f3(x,y,z,np.hstack((x,y,z)))

    (x,y,z) = triplet(3)
    tp[2*N+i,:] = f3(x,y,z,np.hstack((x,y,z)))

  return (tp,np.hstack((np.zeros(N),np.ones(N),N*[2])).ravel())
    
def load_dataset(fname="../data/housing/housing.data",cols=(0,)):
  X = np.genfromtxt(fname,usecols=cols,delimiter = ',')
  #X = np.genfromtxt(fname,usecols=cols)

  num_features = X.shape[1]
  num_triplets = int(6*num_features*(num_features-1)*(num_features-2)/6);
    
  triplets = np.zeros((num_triplets,4*wx.shape[1]))

  print ':: loading dataset...please wait!'
  l = 0
  for i in range(num_features-2):
    for j in range(i+1,num_features-1):
      for k in range(j+1,num_features):
              
        permute_idx = itertools.permutations([i,j,k])
        for idx in permute_idx:
          x = scale(np.array(X[:,idx[0]]))[:,np.newaxis]
          y = scale(np.array(X[:,idx[1]]))[:,np.newaxis]
          z = scale(np.array(X[:,idx[2]]))[:,np.newaxis]
          
          triplets[l,:] = f3(x,y,z,np.hstack((x,y,z)))
          l = l + 1
          
  return (triplets,num_features,num_triplets)

def build_dag(node_labels,final_scores,permute_idx,threshold):

  G = nx.DiGraph()
  G.add_nodes_from(node_labels)

  max_scores = np.amax(final_scores,axis=1)
  sorted_idx = np.argsort(-max_scores)

  # build dag
  l = 0
  for i in range(num_features-2):
    for j in range(i+1,num_features-1):
      for k in range(j+1,num_features):                
        
        if (max_scores[sorted_idx[l]] < threshold):
          return G

        p_idx = np.array(list(itertools.permutations([i,j,k])))
        p_new_idx = p_idx[permute_idx[sorted_idx[l]]]

        dag_type = np.argmax(final_scores[sorted_idx[l],:])
        if (dag_type == 0):
          edges = [(node_labels[p_new_idx[0]],node_labels[p_new_idx[1]]),(node_labels[p_new_idx[1]],node_labels[p_new_idx[2]])]
        elif (dag_type == 1):
          edges = [(node_labels[p_new_idx[1]],node_labels[p_new_idx[0]]),(node_labels[p_new_idx[1]],node_labels[p_new_idx[2]])]
        elif (dag_type == 2):        
          edges = [(node_labels[p_new_idx[0]],node_labels[p_new_idx[1]]),(node_labels[p_new_idx[2]],node_labels[p_new_idx[1]])]

        if (G.has_edge(*edges[0]) and G.has_edge(*edges[1])):
          edges = []
        elif (G.has_edge(*edges[0])):
          edges.remove(edges[0])
        elif (G.has_edge(*edges[1])):
          edges.remove(edges[1])

        if edges:
          G.add_edges_from(edges)
          if (not nx.is_directed_acyclic_graph(G)):
            G.remove_edges_from(edges)

        l = l + 1

  return G

###
def infer_boston_housing(reg):
  # load test data
  (triplets,num_features,num_triplets) = load_dataset("../data/housing/housing.data")
  y_prob = reg.predict_proba(triplets)

  # save the predictive probability of pairs
  np.savetxt('housing_triplet_predict.txt',y_prob,fmt='%.5f')

  # post-process the prediction score list
  final_scores = np.zeros((int(num_triplets/6),3))
  permute_idx = np.zeros(int(num_triplets/6))
    
  l = 0
  for p in range(int(num_triplets/6)):
    permute_idx[p] = np.argmax(np.amax(y_prob[l:(l+6),:],axis=1))
    final_scores[p,:] = y_prob[l + permute_idx[p],:]
    l = l + 6

  # save the predictive probability of pairs
  np.savetxt('housing_triplet_predict_final.txt',final_scores,fmt='%.5f')
  np.savetxt('housing_triplet_predict_index.txt',permute_idx,fmt='%d')

  # constract a DAG
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

  return (node_labels,final_scores,permute_idx,desp)

###
def infer_abalone(reg):
  # load test data
  (triplets,num_features,num_triplets) = load_dataset("../data/abalone/abalone.data",range(1,9))
  y_prob = reg.predict_proba(triplets)

  # save the predictive probability of pairs
  np.savetxt('abalone_triplet_predict.txt',y_prob,fmt='%.5f')

  # post-process the prediction score list
  final_scores = np.zeros((int(num_triplets/6),3))
  permute_idx = np.zeros(int(num_triplets/6))
    
  l = 0
  for p in range(int(num_triplets/6)):
    permute_idx[p] = np.argmax(np.amax(y_prob[l:(l+6),:],axis=1))
    final_scores[p,:] = y_prob[l + permute_idx[p],:]
    l = l + 6

  # save the predictive probability of pairs
  np.savetxt('abalone_triplet_predict_final.txt',final_scores,fmt='%.5f')
  np.savetxt('abalone_triplet_predict_index.txt',permute_idx,fmt='%d')

  # constract a DAG
  node_labels = ["LEN","DIA","HEI","WHO","SHU","VIS","SHE","RIN"]

  desp = """ 
Sex / nominal / -- / M, F, and I (infant) 
Length / continuous / mm / Longest shell measurement 
Diameter	/ continuous / mm / perpendicular to length 
Height / continuous / mm / with meat in shell 
Whole weight / continuous / grams / whole abalone 
Shucked weight / continuous	/ grams / weight of meat 
Viscera weight / continuous / grams / gut weight (after bleeding) 
Shell weight / continuous / grams / after being dried 
Rings / integer / -- / +1.5 gives the age in years 
"""

  return (node_labels,final_scores,permute_idx,num_features,desp)
  
###### the main code starts here ################

# generate random features
wx = rp(250,[0.2,2,20],1)
wy = rp(250,[0.2,2,20],1)
wz = rp(250,[0.2,2,20],1)
ww = rp(250,[0.2,2,20],3)

# generate training data
print ':: generating triplet training data...'
(X,Y) = tripletset(5000)

# train the classifier and predict the test data
print ':: training the random forest classifier...'
reg = OVR(RFC(n_estimators=1000,random_state=0,n_jobs=24)).fit(X,Y);

(node_labels,final_scores,permute_idx,num_features,desp) = infer_abalone(reg)

print ':: contructing a DAG...'
G = build_dag(node_labels,final_scores,permute_idx,0.7)

# save dag in dot format
nx.write_dot(G,"abalone.dot")

# draw dag
#pos = nx.circular_layout(G,scale=3)
#nx.draw(G, pos, cmap=plt.get_cmap('jet'), node_size=2000, node_color='b', alpha=0.95)
#nx.draw_networkx_labels(G, pos, font_color='w')
#nx.draw_networkx_edges(G, pos, arrows=True)
#plt.text(1.5, 0.2, desp, fontsize=10)
#plt.savefig('abalone_tmp.eps')
#plt.show()

print 'Done.'
