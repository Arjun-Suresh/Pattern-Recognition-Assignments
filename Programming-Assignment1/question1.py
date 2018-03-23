from math import *
import matplotlib.pyplot as plt

def phi(x):
    return (1.0 + erf(x/sqrt(2.0))) / 2.0

def func1(d,sigma):
    val = -1.0 * (sqrt(d)/(2.0 * sigma))
    return phi(val)

def func2(d,sigma,rho):
    val = -1.0 * (sqrt(d)/((2.0*sigma)*(sqrt(1.0+rho))))
    return phi(val)

def func2(d,sigma,rho):
    val = -1.0 * (sqrt(d)/((2.0*sigma)*(sqrt(1.0+rho))))
    return phi(val)

def func3(d,sigma,rho):
    val = -1.0 * (sqrt(d)/((2.0*sigma)*(sqrt(1.0+2.0*rho))))
    return phi(val)

def func3(d,sigma,rho):
    val = -1.0 * (sqrt(d)/((2.0*sigma)*(sqrt(1.0+2.0*rho))))
    return phi(val)

x=[]
for i in range(50):
    x.append(i+0.1)

y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
for i in x:
    y1.append(func1(6.0,float(i)))
    y2.append(func2(6.0,float(i),0.2))
    y3.append(func3(6.0,float(i),0.2))
    y4.append(func2(6.0,float(i),0.8))
    y5.append(func3(6.0,float(i),0.8))

	
plt.plot(x, y1, label = "Model1")
plt.plot(x, y2, label = "Model2,rho=0.2")
plt.plot(x, y3, label = "Model3,rho=0.2")
plt.plot(x, y4, label = "Model2,rho=0.8")
plt.plot(x, y5, label = "Model3,rho=0.8")
plt.xlabel('sigma')
plt.ylabel('Bayes error')
plt.title('Question 1')
plt.legend()
plt.show()
