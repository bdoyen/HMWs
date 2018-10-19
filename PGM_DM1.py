import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

#######################
### ~ import data ~ ###
#######################

df_classificationA = pd.read_table('./classificationA.train')
df_classificationB = pd.read_table('./classificationB.train')
df_classificationC = pd.read_table('./classificationC.train')

df_classificationA_test = pd.read_table('./classificationA.test')
df_classificationB_test = pd.read_table('./classificationB.test')
df_classificationC_test = pd.read_table('./classificationC.test')

############################
### ~ data preparation ~ ###
############################

def pre_processing(df_classification):

	### separation of dataset depending on y value ###
	df_x_1 = df_classification.iloc[:,0]
	df_x_2 = df_classification.iloc[:,1]
	df_y = df_classification.iloc[:,2]
	df_y_0 = df_classification[df_classification.iloc[:,2] == 0]
	df_y_1 = df_classification[df_classification.iloc[:,2] == 1]

	x_1 = np.array(df_x_1)
	x_2 = np.array(df_x_2)
	x = x_1 + x_2
	y = np.array(df_y)

	n = len(df_classification)
	num_0 = len(df_y_0)
	num_1 = len(df_y_1)

	x_1_y_0 = df_y_0.iloc[:,0]
	x_2_y_0 = df_y_0.iloc[:,1]
	x_1_y_1 = df_y_1.iloc[:,0]
	x_2_y_1 = df_y_1.iloc[:,1]

	return [x_1,x_2,x,y,n,num_0,num_1,x_1_y_0,x_2_y_0,x_1_y_1,x_2_y_1]

###############
### ~ LDA ~ ###
###############

def LDA(x_1_y_0,x_2_y_0,x_1_y_1,x_2_y_1,n,num_0,num_1):
	# estimator of pi
	pi_estim_lda = num_1/n

	# estimator of mu_0
	mu_0_estim_1 = (1/num_0)*x_1_y_0.sum()
	mu_0_estim_2 = (1/num_0)*x_2_y_0.sum()

	# estimator of mu_1
	mu_1_estim_1 = (1/num_1)*x_1_y_1.sum()
	mu_1_estim_2 = (1/num_1)*x_2_y_1.sum()

	# estimator of sigma (assumed equal for both classes)
	# computation for y = 0 class (more observations for this class)
	x_1_y_0_mean = [elem - mu_0_estim_1 for elem in np.array(x_1_y_0)]
	x_2_y_0_mean = [elem - mu_0_estim_2 for elem in np.array(x_2_y_0)]
	sigma_1_y_0 = (1/num_0)*sum(elem*elem for elem in x_1_y_0_mean)
	sigma_2_y_0 = (1/num_0)*sum(elem*elem for elem in x_2_y_0_mean)

	a_1 = (mu_1_estim_1 - mu_0_estim_1)/sigma_1_y_0
	a_2 = (mu_1_estim_2 - mu_0_estim_2)/sigma_2_y_0

	b = math.log((( (1 - pi_estim_lda)/(pi_estim_lda) )), 10) + 1/2*(mu_1_estim_1*mu_1_estim_1/sigma_1_y_0 + mu_1_estim_2*mu_1_estim_2/sigma_2_y_0) - 1/2*(mu_0_estim_1*mu_0_estim_1/sigma_1_y_0 + mu_0_estim_2*mu_0_estim_2/sigma_2_y_0)	

	return [a_1,a_2,b]

###############################
### ~ Logistic Regression ~ ###
###############################

def sigmoid(x):
	return (1/(1+math.exp(-x)))

def sigmoid_prim(x):
	return sigmoid(x)*(1-sigmoid(x))

def L(w,b,x,y):
	z = zip(x,y)
	l = [y*math.log(sigmoid(w*x+b),10) + (1-y)*math.log(1-sigmoid(w*x+b),10) for (x,y) in z]
	return sum(l)

def grad_w_L(w,b,x,y):
	z = zip(x,y)
	l = [y*x - x*sigmoid(w*x+b) for (x,y) in z]
	return sum(l)

def grad_b_L(w_1,w_2,b,x_1,x_2,y):
	z_1 = zip(x_1,y)
	z_2 = zip(x_2,y)
	l1 = [y - sigmoid(w_1*x_1+b) for (x_1,y) in z_1]
	l2 = [y - sigmoid(w_2*x_2+b) for (x_2,y) in z_2]
	return sum(l1 + l2)

def hess_w_L(w,b,x):
	l = [-x*x*sigmoid_prim(w*x+b) for x in x]
	return sum(l)

def hess_b_L(w_1,w_2,b,x_1,x_2):
	l1 = [-sigmoid_prim(w_1*x_1+b) for x_1 in x_1]
	l2 = [-sigmoid_prim(w_2*x_2+b) for x_2 in x_2]
	return sum(l1 + l2)

def NewtonRaphson(x_1,x_2,y,epsilon,itermax):

	# initial values
	it = 0
	converged = False
	w_current_1 = random.uniform(-1,1)
	w_current_2 = random.uniform(-1,1)
	b_current = random.uniform(-1,1)
	L_current_1 = L(w_current_1,b_current,x_1,y)

	while (not converged) and (it < itermax):
		it = it+1

		L_current_1 = L(w_current_1,b_current,x_1,y)
		L_current_2 = L(w_current_2,b_current,x_2,y)

		grad_w_L_current_1 = grad_w_L(w_current_1,b_current,x_1,y)
		hess_w_L_current_1 = hess_w_L(w_current_1,b_current,x_1)
		
		grad_w_L_current_2 = grad_w_L(w_current_2,b_current,x_2,y)
		hess_w_L_current_2 = hess_w_L(w_current_2,b_current,x_2)
		
		grad_b_L_current = grad_b_L(w_current_1,w_current_2,b_current,x_1,x_2,y)
		hess_b_L_current = hess_b_L(w_current_1,w_current_2,b_current,x_1,x_2)

		w_new_1 = w_current_1 - grad_w_L_current_1/hess_w_L_current_1
		w_new_2 = w_current_2 - grad_w_L_current_2/hess_w_L_current_2

		b_new = b_current - grad_b_L_current/hess_b_L_current
		
		L_new_1 = L(w_new_1,b_new,x_1,y)
		L_new_2 = L(w_new_2,b_new,x_2,y)

		if (abs(L_new_1 - L_current_1) < epsilon) and (abs(L_new_2 - L_current_2) < epsilon):
			converged = True

		w_current_1 = w_new_1
		w_current_2 = w_new_2
		b_current = b_new

	return [w_current_1,w_current_2,b_current]

#############################
### ~ Linear Regression ~ ###
#############################

def NormalEquation(x_1,x_2,y):

	# Design Matrix X
	coordinates = zip(x_1,x_2)
	X = np.array([ np.array([x_1, x_2, 1]).transpose() for (x_1,x_2) in coordinates])
	Xt = X.transpose()
	Y = np.array(y)
	A = np.dot(Xt,X)
	b = np.dot(Xt,Y)
	w = np.linalg.solve(A, b)
	return w

###############
### ~ QDA ~ ###
###############



#############################
### ~ Error computation ~ ###
#############################

def ModelError(w_1,w_2,b,x_1,x_2,y,n):
	z = zip(x_1,x_2)
	predicted_value = [w_1*x_1 + w_2*x_2 + b for (x_1,x_2) in z]
	predicted_class = [int(pred > 0 or pred == 0) for pred in predicted_value]
	z_class = zip(predicted_class,y)
	result = [int(pred_class != int(y)) for (pred_class,y) in z_class]
	return sum(result)/n

def ErrorsTrainAndTest(df_classification_train,df_classification_test,params):
	[x_1_train,x_2_train,x_train,y_train,n_train,num_0_train,num_1_train,x_1_y_0_train,x_2_y_0_train,x_1_y_1_train,x_2_y_1_train] = pre_processing(df_classification_train)
	[x_1_test,x_2_test,x_test,y_test,n_test,num_0_test,num_1_test,x_1_y_0_test,x_2_y_0_test,x_1_y_1_test,x_2_y_1_test] = pre_processing(df_classification_test)

	# LDA method
	[w_1_estim_lda,w_2_estim_lda,b_estim_lda] = LDA(x_1_y_0_train,x_2_y_0_train,x_1_y_1_train,x_2_y_1_train,n_train,num_0_train,num_1_train)
	lda_error_train = ModelError(w_1_estim_lda,w_2_estim_lda,b_estim_lda,x_1_train,x_2_train,y_train,n_train)
	lda_error_test = ModelError(w_1_estim_lda,w_2_estim_lda,b_estim_lda,x_1_test,x_2_test,y_test,n_test)

	# Logistic Regression
	[epsilon,itermax] = params
	[w_1_estim_nr, w_2_estim_nr, b_estim_nr] = NewtonRaphson(x_1_train,x_2_train,y_train,epsilon,itermax)
	nr_error_train = ModelError(w_1_estim_nr,w_2_estim_nr,b_estim_nr,x_1_train,x_2_train,y_train,n_train)
	nr_error_test = ModelError(w_1_estim_nr,w_2_estim_nr,b_estim_nr,x_1_test,x_2_test,y_test,n_test)

	# Linear Regression 
	[b_estim_lin, w_1_estim_lin, w_2_estim_lin] = NormalEquation(x_1_train,x_2_train,y_train)
	lin_error_train = ModelError(w_1_estim_lin,w_2_estim_lin,b_estim_lin,x_1_train,x_2_train,y_train,n_train)
	lin_error_test = ModelError(w_1_estim_lin,w_2_estim_lin,b_estim_lin,x_1_test,x_2_test,y_test,n_test)

	return [lda_error_train, nr_error_train, lin_error_train], [lda_error_test, nr_error_test, lin_error_test]

params = [10^(-3),1000]

def Plot(df_classification_train,df_classification_test,params):
	[x_1_train,x_2_train,x_train,y_train,n_train,num_0_train,num_1_train,x_1_y_0_train,x_2_y_0_train,x_1_y_1_train,x_2_y_1_train] = pre_processing(df_classification_train)
	[x_1_test,x_2_test,x_test,y_test,n_test,num_0_test,num_1_test,x_1_y_0_test,x_2_y_0_test,x_1_y_1_test,x_2_y_1_test] = pre_processing(df_classification_test)

	# grid
	fig, ax = plt.subplots(2, 4)

	# LDA method
	[w_1_estim_lda,w_2_estim_lda,b_estim_lda] = LDA(x_1_y_0_train,x_2_y_0_train,x_1_y_1_train,x_2_y_1_train,n_train,num_0_train,num_1_train)

	ax[0, 0].plot(x_1_y_0_train,x_2_y_0_train,'bs') # y=0 class
	ax[0, 0].plot(x_1_y_1_train,x_2_y_1_train,'gs') # y=1 class
	ax[0, 0].plot(x_1_train,(1/w_2_estim_lda)*(-b_estim_lda - w_1_estim_lda*x_1_train), linestyle='--', color = 'r') 

	ax[0, 1].plot(x_1_y_0_test,x_2_y_0_test,'bs') # y=0 class
	ax[0, 1].plot(x_1_y_1_test,x_2_y_1_test,'gs') # y=1 class
	ax[0, 1].plot(x_1_test,(1/w_2_estim_lda)*(-b_estim_lda - w_1_estim_lda*x_1_test), linestyle='--', color = 'r') 

	# Logistic Regression
	[epsilon,itermax] = params
	[w_1_estim_nr, w_2_estim_nr, b_estim_nr] = NewtonRaphson(x_1_train,x_2_train,y_train,epsilon,itermax)

	ax[0, 2].plot(x_1_y_0_train,x_2_y_0_train,'bs') # y=0 class
	ax[0, 2].plot(x_1_y_1_train,x_2_y_1_train,'gs') # y=1 class
	ax[0, 2].plot(x_1_train,(1/w_2_estim_nr)*(-b_estim_nr - w_1_estim_nr*x_1_train), linestyle='--', color = 'r') 

	ax[0, 3].plot(x_1_y_0_test,x_2_y_0_test,'bs') # y=0 class
	ax[0, 3].plot(x_1_y_1_test,x_2_y_1_test,'gs') # y=1 class
	ax[0, 3].plot(x_1_test,(1/w_2_estim_nr)*(-b_estim_nr - w_1_estim_nr*x_1_test), linestyle='--', color = 'r') 

	# Linear Regression 
	[b_estim_lin, w_1_estim_lin, w_2_estim_lin] = NormalEquation(x_1_train,x_2_train,y_train)

	ax[1, 0].plot(x_1_y_0_train,x_2_y_0_train,'bs') # y=0 class
	ax[1, 0].plot(x_1_y_1_train,x_2_y_1_train,'gs') # y=1 class
	ax[1, 0].plot(x_1_train,(1/w_2_estim_lin)*(-b_estim_lin - w_1_estim_lin*x_1_train), linestyle='--', color = 'r') 

	ax[1, 1].plot(x_1_y_0_test,x_2_y_0_test,'bs') # y=0 class
	ax[1, 1].plot(x_1_y_1_test,x_2_y_1_test,'gs') # y=1 class
	ax[1, 1].plot(x_1_test,(1/w_2_estim_lin)*(-b_estim_lin - w_1_estim_lin*x_1_test), linestyle='--', color = 'r') 

	plt.show()
	

#Plot(df_classificationA,df_classificationA_test,params)
Plot(df_classificationB,df_classificationB_test,params)
#Plot(df_classificationC,df_classificationC_test,params)



