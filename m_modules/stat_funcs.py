import numpy as np

'''Stat functions'''
# least chi squared function
def chi_sq(y,y_pred):
    return np.sum((y-y_pred)**2)

# Rootmean square error
def rmse(y,y_pred):
    return np.sqrt(np.sum((y-y_pred)**2)/len(y))

# max error
def max_error(y,y_pred):
    return np.max(np.abs(y-y_pred))

# mean absolute error
def mae(y,y_pred):
    return np.sum(np.abs(y-y_pred))/len(y)

# standard deviation
def std(y,y_pred):
    return np.sqrt(np.sum((y-y_pred)**2)/(len(y)-1))

# fit functions
def lin_fit(x,m,c):
    return m*x + c

def quad_fit(x, a, b, c):
    return a*x**2 + b*x + c

def quart_fit(x,a,b,c,d,e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def quin_fit(x,a,b,c,d,e,f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def power_sine(x, b,A, f, p):
    return x**b + x*A*np.sin(f*x + p)

# fit function for variable number of terms in fourier series

# def ff1(lam, A,f, p, exp, C):
#     return A*np.exp(lam*exp) * np.sin(lam*f + p) + C

# def ff2(lam, A,f, p, exp, C, A2,f2, p2, exp2, C2):
#     return A*np.exp(lam*exp) * np.sin(lam*f + p) + C + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2) + C2

# def ff3(lam, A,f, p, exp, C, A2,f2, p2, exp2, C2, A3,f3, p3, exp3, C3):
#     return A*np.exp(lam*exp) * np.sin(lam*f + p) + C + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2) + C2 + A3*np.exp(lam*exp3) * np.sin(lam*f3 + p3) + C3

# def ff4(lam, A, f, p, exp, C, A2, f2, p2, exp2, C2, A3, f3, p3, exp3, C3, A4, f4, p4, exp4, C4):
#     return A*np.exp(lam*exp) * np.sin(lam*f + p) + C + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2) + C2 + A3*np.exp(lam*exp3) * np.sin(lam*f3 + p3) + C3 + A4*np.exp(lam*exp4) * np.sin(lam*f4 + p4) + C4

# def ff5(lam, A, f, p, exp, C, A2, f2, p2, exp2, C2, A3, f3, p3, exp3, C3, A4, f4, p4, exp4, C4, A5, f5, p5, exp5, C5):
#     return A*np.exp(lam*exp) * np.sin(lam*f + p) + C + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2) + C2 + A3*np.exp(lam*exp3) * np.sin(lam*f3 + p3) + C3 + A4*np.exp(lam*exp4) * np.sin(lam*f4 + p4) + C4 + A5*np.exp(lam*exp5) * np.sin(lam*f5 + p5) + C5

def ff1(lam, A,f, p, exp):
    return A*np.exp(lam*exp) * np.sin(lam*f + p)

def ff2(lam, A,f, p, exp, A2,f2, p2, exp2):
    return A*np.exp(lam*exp) * np.sin(lam*f + p) + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2)

def ff3(lam, A,f, p, exp,A2,f2, p2, exp2, A3,f3, p3, exp3):
    return A*np.exp(lam*exp) * np.sin(lam*f + p) + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2) + A3*np.exp(lam*exp3) * np.sin(lam*f3 + p3)

def ff4(lam, A, f, p, exp, A2, f2, p2, exp2, A3, f3, p3, exp3, A4, f4, p4, exp4):
    return A*np.exp(lam*exp) * np.sin(lam*f + p) + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2) + A3*np.exp(lam*exp3) * np.sin(lam*f3 + p3) + A4*np.exp(lam*exp4) * np.sin(lam*f4 + p4)

def ff5(lam, A, f, p, exp, A2, f2, p2, exp2, A3, f3, p3, exp3, A4, f4, p4, exp4, A5, f5, p5, exp5):
    return A*np.exp(lam*exp) * np.sin(lam*f + p) + A2*np.exp(lam*exp2) * np.sin(lam*f2 + p2) + A3*np.exp(lam*exp3) * np.sin(lam*f3 + p3) + A4*np.exp(lam*exp4) * np.sin(lam*f4 + p4) + A5*np.exp(lam*exp5) * np.sin(lam*f5 + p5)

def x_n(freq, A,offset,n,base):
    return A*(freq+offset)**n + base

def exp(freq, A, offset,n):
    return A*np.exp((freq+offset)*n)