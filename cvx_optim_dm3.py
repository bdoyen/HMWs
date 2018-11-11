import matplotlib.pyplot as plt
import numpy as np

### Preliminary Functions ###

# computation of the initial f0 objective function
def func(Q,p,A,b,nu):
    n = len(nu)
    nu_transpose = nu.transpose()
    p_transpose = p.transpose()
    result = nu_transpose.dot(Q.dot(nu)) + p_transpose.dot(nu)
    return result[0][0]

# computation of the objective function + log-barrier term for appropriate nu values
def func_objective(Q,p,A,b,t,nu):
    result = 0
    n = len(nu)
    nu_transpose = nu.transpose()
    p_transpose = p.transpose()
    term = nu_transpose.dot(Q.dot(nu)) + p_transpose.dot(nu)
    mult = A.dot(nu)
    sum_log = 0
    for i in range(0,n):
        sum_log = sum_log - np.log(b[i]-mult[i])
    return t*term + sum_log

# computation of the gradient of the objective function + log-barrier term
def grad_func_objective(Q,p,A,b,t,nu):
    n = len(nu)
    grad_phi = np.zeros(shape=(n,1))
    mult = A.dot(nu)
    for i in range(0,n):
        grad_phi = grad_phi + (1/(b[i]-mult[i]))*np.reshape(A[i],(n,1))
    return t*(2*Q.dot(nu) + p) + grad_phi

# computation of the hessian of the objective function + log-barrier term
def hess_func_objective(Q,p,A,b,t,nu):
    n = len(nu)
    hess_phi = np.zeros(shape=(n,n))
    mult = A.dot(nu)
    for i in range(0,n):
        grad_fi = -np.reshape(A[i],(n,1))
        hess_phi = hess_phi + (1/(b[i]-mult[i])**2)*(grad_fi.dot(grad_fi.transpose()))
    return 2*t*Q + hess_phi


### Log-Barrier Method ###

# computing the centering step using backtracking-line search method for descent direction choice
def centering_step(Q,p,A,b,t,v0,eps):
    
    n = len(v0)
    nu_current = v0
    iterations = []
    converged = False
    
    # parameters of backtracking-line search
    alpha = 0.1
    beta = 0.2
    it = 0
    
    # Newton's Method
    
    while not(converged):
        
        # general computations
        grad = grad_func_objective(Q,p,A,b,t,nu_current)
        grad_T = grad.transpose()
        hess_inv = np.linalg.inv(hess_func_objective(Q,p,A,b,t,nu_current))
        
        delta_x = -hess_inv.dot(grad) # newton step
        prod = grad_T.dot(delta_x)
        lambda_2 = -prod # newton decrement
        
        nu_next = np.zeros(shape=(n,1))
        
        ### backtracking-line search method

        # init
        s = 1
        
        # checking for log domain belonging
        while ( (b - A.dot(nu_current + s*delta_x)).min() <= 0 ):
            s = s*beta
        
        # checking for minimality search condition
        while ( func_objective(Q,p,A,b,t,nu_current + s*delta_x)[0][0] > func_objective(Q,p,A,b,t,nu_current)[0][0] + alpha*s*prod[0][0] ):
            s = s*beta
         
        # update
        nu_next = nu_current + s*delta_x     
        nu_current = nu_next
        iterations.append(nu_current)
        it = it + 1
        
        if (lambda_2[0][0]/2 < eps):
            converged = True
    
    return iterations


# log-barrier method
def barr_method(Q,p,A,b,v0,mu,eps):
    
    n = len(v0)
    nu_current = v0
    t = 1
    iterations_barr = []
    newton_iterations = []
    precision = []
    converged = False
    
    while not(converged):
        
        iterations = centering_step(Q,p,A,b,t,v0,eps)
        newton_iterations.append(len(iterations))
        precision.append(n/t)
        nu = iterations[-1] # get best surrogate from centering step
        if (n/t < eps):
            converged = True
        t = mu*t
        iterations_barr.append(nu)
    
    return iterations_barr, newton_iterations, precision
        

### Tests ###

### Input variables

# randomly generated data
n = 100 # number of observations
d = 100 # number of dimensions
lambda_cte = 10
X = np.random.rand(n,d)
y = np.random.rand(n,1)

# input data
Q = 0.5*np.eye(n,n) # Q is symmetric by assumption
A = X.transpose()
b = lambda_cte*np.ones(shape=(d,1))
p = -y
v0 = np.zeros(shape=(n,1)) #feasible point
eps = 1e-6

# mu parameters
mu_values = [2,10,30,50,100,150]
colors = ["gray","blue","purple","red","green","yellow"]

### Output : plots 

y_plot_mu = []

### step plot
fig, ax=plt.subplots()
for mu in mu_values:
    # barrier method computations
    iterations_barr, newton_iterations, precision = barr_method(Q,p,A,b,v0,mu,eps)
    print("-> Log-Barrier method computed for mu :")
    print(str(mu))
    progress = (mu_values.index(mu) + 1)/len(mu_values)
    print("progress of : " + str(int(progress*100)) + '%')
    x_plot = list(range(1,len(newton_iterations) + 1))
    y_plot = [func(Q,p,A,b,elem) - func(Q,p,A,b,iterations_barr[-1]) for elem in iterations_barr[:-1]]
    y_plot_mu.append(x_plot[-1])
    ax.step(x_plot[:-1],y_plot,color=colors[mu_values.index(mu)],label="mu = "+str(mu))
    ax.set_yscale("log")
plt.title("Step plot")
plt.xlabel("Number of Newton Iterations")
plt.ylabel("Duality Gap")
plt.legend(loc='upper right')
plt.show()


### code for mu determination plot
"""
plt.subplot(2,1,2)
x_plot_mu = mu_values
plt.plot(x_plot_mu,y_plot_mu)
plt.title("Parameter mu determination plot")
plt.xlabel("mu")
plt.ylabel("Number of Newton Iterations")
plt.show()
"""

