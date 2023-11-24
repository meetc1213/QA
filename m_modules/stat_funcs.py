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


# extra funcs
def get_lengths(*args):
    # get lengths of arrays
    lengths = []
    for arg in args:
        if type(arg) == np.ndarray:
            lengths.append(len(arg))
        else:
            lengths.append(1)
    return lengths

def reflect(arr, neg=1):
    # for generating symmetric data
    if type(arr[0]) != np.ndarray:
        return np.append(neg*arr[:0:-1],arr)

    if type(arr[0]) == np.ndarray:
        arr_new= []
        for i in range(len(arr)):
            arr_new.append(reflect(arr[i]))
        return arr_new

def check_even(err):
    flipped_array = np.flip(err)
    return np.isclose(flipped_array, err).all()

# def reflect(arr, neg=1):
#     flipped_array = np.flip(arr[1:])
#     even_array = np.append(flipped_array,arr)
#     return even_array

# fit functions
def lin_fit(x,m,c):
    return m*x + c

def quad_fit(x, a, b, c):
    return a*(x-b)**2 + c

def quart_fit(x,a,b,c,d,e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def quin_fit(x,a,b,c,d,e,f):
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

def gaussian_fit(x,a,b,c):
    return a*np.exp(-(x)**2/(2*b**2)) + c

def power_sine(x, b,A, f, p):
    return x**b + x*A*np.sin(f*x + p)

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

def x_n(freq, A,offset,n):
    return A*(freq-offset)**n

def exp(freq, A, offset,n):
    return A*np.exp((freq-offset)*n)

def power(freq, A,b, offset,n):
    return A*b**((freq+offset)*n)

def closed_form(x, h_real_n,f_real_n, A_real,p_real_p, n_real_p):
    # returns an expression of the closed form of my model of fft
    return 2*(h_real_n*np.cos(f_real_n*x) - A_real * (x*np.sin(x*p_real_p) - n_real_p * np.cos(p_real_p*x)) / (n_real_p**2 + x**2))

def beat(x, A, f, f2, p, p2,b):
    return A*np.cos(2*np.pi*f*x + p)*np.cos(2*np.pi*f2*x + p2) + b

def get_values_uc(func, popt, pcov):
    import inspect
    try:
        parameters = inspect.signature(func).parameters
        parameter_names = [param for param in parameters.keys()][1:] # to exclude independent data variable of the fit function
    except TypeError:
        print("Unable to retrieve parameter names for the given function.")

    ucs = np.sqrt(np.diag(pcov))
    print("Parameter values:")
    for i, param in enumerate(parameter_names):
        print(f"{param} = {popt[i]} ± {ucs[i]}")

def pol_fit(d, A, n, C):
    # here U is energy at separation of 1 Bohr
    # A = (1 / (C-U))** (1/n) - 1
    # A = 0
    return A / np.power(d,n) + C


