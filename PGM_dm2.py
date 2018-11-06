
# coding: utf-8

# In[1]:


from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import numpy as np
import math


# In[2]:


df_train = np.array(pd.read_table('./EMGaussian.data')).tolist()
df_test = np.array(pd.read_table('./EMGaussian.test')).tolist()

df_train_x_1 = [float(x[0].split(' ')[0]) for x in df_train]
df_train_x_2 = [float(x[0].split(' ')[1]) for x in df_train]
df_test_x_1 = [float(x[0].split(' ')[0]) for x in df_test]
df_test_x_2 = [float(x[0].split(' ')[1]) for x in df_test]


# In[6]:


#K-mean algorithm
K = 4
mu = [[0,0]]*K
classes = [[]]*K
distances = []
n = len(df_train_x_1)

#initialize the algorithm
for i in range(0,K):
    rand = randint(0,n-1)
    mu[i] = [df_train_x_1[rand],df_train_x_2[rand]]

itermax = 50
it = 0


# In[7]:


while it<itermax:
    #determine new class label for each data point
    for j in range(0,n-1):
        add = []
        for i in range(0,K):
            dist_current = np.sqrt( (df_train_x_1[j] - mu[i][0])**2 + (df_train_x_2[j] - mu[i][1])**2 )
            add.append(dist_current)
        classes[add.index(min(add))] = classes[add.index(min(add))] + [[df_train_x_1[j], df_train_x_2[j]]]

    #compute the new centroïds
    for i in range(0,K):
        mu_x_1 = np.mean([elem[0] for elem in classes[i]])
        mu_x_2 = np.mean([elem[1] for elem in classes[i]])
        mu[i] = [mu_x_1, mu_x_2]
    
    #iterate
    it = it + 1


# In[8]:


colors = ['bs','gs','ys','rs']
for j in range(0,K):
    plt.plot([elem[0] for elem in classes[j]],[elem[1] for elem in classes[j]],colors[j],markersize=2)
    plt.plot([elem[0] for elem in mu],[elem[1] for elem in mu],'rs')
plt.show()


# In[9]:


### EM algorithm - applied to Gaussian mixture ###

### with covariance matrix proportional to identity 

#initialization of the algorithm using k-means 

def init_kmean(tau,mu):
    for i in range(0,n):
        add = []
        for j in range(0,K):
            dist_current = np.sqrt( (df_train_x_1[i] - mu[j][0])**2 + (df_train_x_2[i] - mu[j][1])**2 )
            add.append(dist_current)
        for j in range(0,K):
            if add.index(min(add)) == j:
                    tau[i,j] = 1
            else:
                    tau[i,j] = 0
    return tau
        
# E-steps and M-steps iterations
# means computation
tau_init = init_kmean(np.zeros(shape=(n,K)),mu)
epsilon = 0.001
mu_star = [[0,0]]*K
pi_star = [0]*K
sigma = [0]*K
tau_next, tau_current = np.zeros(shape=(n,K)), np.zeros(shape=(n,K))
tau_current = tau_init

# function to calculate the 'L2' norm of a numpy matrix
def norm2(mat):
    dim = mat.shape
    sum = 0
    for i in range(0,dim[0]):
        for j in range(0,dim[1]):
            sum = sum + mat[i,j]**2
    return np.sqrt(sum)

while ( abs(norm2(tau_next) - norm2(tau_current)) > epsilon ):
        
    pi = [0]*K
    tau = tau_current
    
    # new mean called mu
    for j in range(0,K):
        tau_column = tau[:,j]
        sum_x1 = 0
        sum_x2 = 0
        for i in range(0,n-1): 
            sum_x1 = sum_x1 + tau[i,j]*df_train_x_1[i]
            sum_x2 = sum_x2 + tau[i,j]*df_train_x_2[i]
        mu_star[j] = [sum_x1/tau_column.sum(), sum_x2/tau_column.sum()]
        pi[j] = tau_column.sum()
    
    # new pi (normalized)
    for j in range(0,K):
        pi_star[j] = (1/(np.sum(pi)))*pi[j]

    # covariances computations
    for j in range(0,K):
        tau_column = tau[:,j]
        sum = 0.0
        for i in range(0,n-1): 
            sum = sum + tau[i,j]*( (df_train_x_1[i] - mu_star[j][0])**2 + (df_train_x_2[i] - mu_star[j][1])**2 )
        sigma[j] = np.sqrt(sum/(2*tau_column.sum()))

    # new tau matrix
    for i in range(0,n):
        # normalization factor computation
        sum = 0
        for j in range(0,K):
            sum = sum + pi_star[j]*(1/(sigma[j]**2))*np.exp(-0.5*(1/sigma[j]**2)*((df_train_x_1[i] - mu_star[j][0])**2 + (df_train_x_2[i] - mu_star[j][1])**2))
        # probabilities computations
        for j in range(0,K):
            tau[i,j] = (1/sum)*pi_star[j]*(1/(sigma[j]**2))*np.exp(-0.5*(1/sigma[j]**2)*((df_train_x_1[i] - mu_star[j][0])**2 + (df_train_x_2[i] - mu_star[j][1])**2))

    tau_next = tau                                              
                                                       
                                                       
                                                       


# In[11]:


tau_next


# In[10]:


### plot of data points + covariance matrix
for j in range(0,K):
    mu = np.array(mu_star[j])
    Sigma = np.array([[sigma[j]**2 , 0], [0, sigma[j]**2]])
    N = 60
    X = np.linspace(-10, 10, N)
    Y = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    F = multivariate_normal(mu, Sigma)
    Z = F.pdf(pos)
    plt.plot([elem[0] for elem in classes[j]],[elem[1] for elem in classes[j]],colors[j],markersize=2)
    plt.contour(X,Y,Z)
plt.show()


# In[18]:


### with general covariance matrix

# E-steps and M-steps iterations
# means computation
epsilon_general = 0.01
mu_star_general = [[0,0]]*K
pi_star_general = [0]*K
sigma_general = [None]*K
tau_next_general, tau_current_general = np.zeros(shape=(n,K)), np.zeros(shape=(n,K))
tau_current_general = tau_init

while ( abs(norm2(tau_next_general) - norm2(tau_current_general)) > epsilon_general ):
    
    tau_general = tau_current_general
    
    pi_general = [0]*K
    
    # new mean called mu_general
    for j in range(0,K):
        tau_column_general = tau_general[:,j]
        sum_x1 = 0
        sum_x2 = 0
        for i in range(0,n-1): 
            sum_x1 = sum_x1 + tau_general[i,j]*df_train_x_1[i]
            sum_x2 = sum_x2 + tau_general[i,j]*df_train_x_2[i]
        mu_star_general[j] = [sum_x1/tau_column_general.sum(), sum_x2/tau_column_general.sum()]
        pi_general[j] = tau_column_general.sum()
    
    # new pi_general (normalized)
    for j in range(0,K):
        pi_star_general[j] = (1/(np.sum(pi_general)))*pi_general[j]

    # covariances computations
    for j in range(0,K):
        tau_current_general = tau_general[:,j]
        sum = np.zeros(shape=(2,2))
        for i in range(0,n-1): 
            arr = np.array([df_train_x_1[i], df_train_x_2[i]]) - np.array([mu_star_general[j][0],mu_star_general[j][1]])
            arr.shape = (2,1)
            sum = sum + tau_general[i,j]*arr.dot(arr.transpose())
        sigma_general[j] = (1/tau_current_general.sum())*sum

    # new tau_general matrix
    for i in range(0,n-1):
        # normalization factor computation
        sum = 0
        for j in range(0,K):
            arr = np.array([df_train_x_1[i], df_train_x_2[i]]) - np.array([mu_star_general[j][0],mu_star_general[j][1]])
            arr.shape = (1,2)
            mult = arr.dot(np.linalg.inv(sigma_general[j]))
            sum = sum + pi_star_general[j]*(1/np.sqrt(np.linalg.det(sigma_general[j])))*np.exp(-0.5*mult.dot(arr.transpose()))
        # probabilities computations
        for j in range(0,K):
            arr = np.array([df_train_x_1[i], df_train_x_2[i]]) - np.array([mu_star_general[j][0],mu_star_general[j][1]])
            arr.shape = (1,2)
            mult = arr.dot(np.linalg.inv(sigma_general[j]))
            tau_general[i,j] = (1/sum)*pi_star_general[j]*(1/np.sqrt(np.linalg.det(sigma_general[j])))*np.exp(-0.5*mult.dot(arr.transpose()))

    tau_next_general = tau_general       



# In[ ]:


### plot of data points + covariance matrix
for j in range(0,K):
    mu = np.array(mu_star_general[j])
    Sigma = sigma_general[j]
    N = 60
    X = np.linspace(-10, 10, N)
    Y = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    F = multivariate_normal(mu, Sigma)
    Z = F.pdf(pos)
    plt.contour(X,Y,Z)
    plt.plot([elem[0] for elem in classes[j]],[elem[1] for elem in classes[j]],colors[j],markersize=2)
plt.show()


# In[17]:


# log-likelihood computations
tau_next_general


# In[ ]:


np.transpose(arr).shape

