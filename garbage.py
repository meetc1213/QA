import numpy as np
# write function that calculates area of triangle from coordinates in 3D
def area_triangle(coords):
    # coords is a 3x3 matrix
    # first row is x coordinates
    # second row is y coordinates
    # third row is z coordinates
    # the area is calculated using Heron's formula
    # first calculate the lengths of the sides
    # the sides are calculated using the distance formula
    # the distance formula is sqrt((x2-x1)^2+(y2-y1)^2+(z2-z1)^2)
    # the sides are a,b,c
    # a is the distance between the first and second point
    # b is the distance between the second and third point
    # c is the distance between the third and first point
    # a is the distance between the first and second point
    a = np.sqrt((coords[0,1]-coords[0,0])**2+(coords[1,1]-coords[1,0])**2+(coords[2,1]-coords[2,0])**2)
    # b is the distance between the second and third point
    b = np.sqrt((coords[0,2]-coords[0,1])**2+(coords[1,2]-coords[1,1])**2+(coords[2,2]-coords[2,1])**2)
    # c is the distance between the third and first point
    c = np.sqrt((coords[0,0]-coords[0,2])**2+(coords[1,0]-coords[1,2])**2+(coords[2,0]-coords[2,2])**2)
    # now calculate the semi-perimeter
    s = (a+b+c)/2
    # now calculate the area
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    return area

# write a function that takes 3 vectors and makes them a valid input for area_triangle
def make_coords(a,b,c):
    coords = np.zeros((3,3))
    coords[0,0] = a[0]
    coords[1,0] = a[1]
    coords[2,0] = a[2]
    coords[0,1] = b[0]
    coords[1,1] = b[1]
    coords[2,1] = b[2]
    coords[0,2] = c[0]
    coords[1,2] = c[1]
    coords[2,2] = c[2]
    return coords

x = (44+2*np.sqrt(56))/ 10
p1 = np.array([5,x,5])
p2 = np.array([6,3,8])
p3 = np.array([5,4,6])

# print(area_triangle(make_coords(p1,p2,p3)))
# create a function to compute power of a complex number given exponential form
def complex_power(z,n):
    # z is a complex number in exponential form
    # n is the power
    # z = r*e^(i*theta)
    # z^n = r^n*e^(i*n*theta)
    # r^n is the magnitude of the complex number
    # n*theta is the angle of the complex number
    # r^n*e^(i*n*theta) is the complex number in exponential form
    # the function returns the complex number in exponential form
    # first calculate the magnitude of the complex number
    r = z[0]**n
    # now calculate the angle of the complex number
    theta = n*z[1]
    # now calculate the complex number in exponential form
    z_n = np.array([r,theta])
    return z_n

for i in range(0,4):
    theta = 11*np.pi / 13
    phi = i * 2*np.pi / 4 + theta / 4
    print(complex_power(np.array([1,phi]),4))
print(2*np.pi)

