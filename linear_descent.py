import numpy as np
m=20
x0=np.ones((m,1))
x1=np.arange(1,m+1).reshape(m,1)
X=np.hstack((x0,x1))

Y=np.array([3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(m,1)

alpha=0.01

def cost_function(theta,X,Y):
    diff=np.dot(X,theta)-Y
    return (1/2*m)*(np.dot(diff.transpose(),diff))

def gradient_function(theta,X,Y):
    diff=np.dot(X,theta)-Y
    return (1/m)*(np.dot(X.transpose(),diff))

def gradient_descent(X,Y,alpha):
    theta=np.array([1,1]).reshape(2,1)
    gradient=gradient_function(theta,X,Y)
    while not all(abs(gradient)<=1e-5):
        theta=theta-alpha*gradient
        gradient=gradient_function(theta,X,Y)
    return theta

optimal = gradient_descent(X, Y, alpha)
print('optimal:', optimal)
print('cost function:', cost_function(optimal, X, Y)[0][0])
