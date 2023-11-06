from copy import copy
import numpy as np
import time
import math
def softmax(X,a=9):
    numerator = 0.0
    denominator = 0.0
    for i in range(len(X)):
        e = math.exp(a*X[i])
        numerator += X[i]*e
        denominator += e

    S = numerator/denominator
    return S

def softmax2(X,a=8):

    e = np.exp(a*np.array(X))

    numerator = 0.0
    denominator = 0.0
    for i in range(len(e)):
        numerator += X[i]*e[i]
        denominator += e[i]

    S = numerator/denominator
    # print(S)
    return S

def softmax_grad(X,a=8):
    numerator = 0.0
    denominator = 0.0
    for i in range(len(X)):
        e = math.exp(a*X[i])
        numerator += X[i]*e
        denominator += e

    S = numerator/denominator
    grad = []
    for x in X:
        grad.append((math.exp(a*x)/denominator)*(1 + a*(x - S)))

    return grad

def softmax_grad_2(X,a=8):
    e = np.exp(a*np.array(X))

    denominator = 0.0
    for i in range(len(e)):
        denominator += e[i]
    grad = []
    S  = softmax(X,a)
    for x in X:
        grad.append((np.exp(a*x)/denominator)*(1 + a*(x - S)))

    # print(grad)
    return grad

def softmin(X,a=8):
    return softmax(np.array(X),-a)

def softmin_grad(X,a=8):
    return softmax_grad(np.array(X),-a)

def softmin_grad_num(X,a=8):
    h = 1e-6
    grad = []
    for i in range(len(X)):
        Xpos = copy(X)
        Xpos[i] += h
        Xneg = copy(X)
        Xneg[i] -= h
        grad.append((softmax(Xpos,-a) - softmax(Xneg,-a))/(2.0*h)) 
    return grad

x = [2,2,2,1]
for a in range(10,11):
    S = softmin(x,a)
    gmin = softmin_grad(x,a)
    gmin_num = softmin_grad_num(x,a)
    # gmax = softmax_grad(x,a)
    print('a = {}'.format(a))
    print('S =  {}'.format(S))
    print('gmin = {}'.format(gmin))
    print('gmin_num = {}'.format(gmin_num))
    print(np.mean(gmin))
    # print('gmax = {}'.format(gmax))


x = np.random.rand(1000,6)
start = time.time()
for i in range(100):
    g = softmax_grad(x[i][:])
end = time.time()
print(end - start)
start = time.time()
for i in range(100):
    g = softmax_grad_2(x[i][:])
end = time.time()
print(end - start)

# u = 1.1
# x = [u, u*u, u + 0.00001, u*u+0.1, u + 0.2*u, u*u + 0.2*u]
# dxdu = [1,2*u,1,0.5*u, 1.2, 2*u + 0.2]
# S = softmin(x,500)
# g = softmin_grad(x,500)
# print('x = {}'.format(x))
# print('dxdu = {}'.format(dxdu))
# print('S = {}'.format(S))
# print('min x = {}'.format(min(x)))
# print('g = {}'.format(g))
# dSdu = np.array(dxdu).dot(np.array(g))

# print('dSdu = {}'.format(dSdu))
