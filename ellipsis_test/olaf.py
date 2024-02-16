
import numpy as np
import cmath 

def solve_quadratic(a, b, c):
    # Calculate the discriminant
    discriminant = (b**2) - (4*a*c)

    # Find two solutions
    sol1 = (-b - cmath.sqrt(discriminant)) / (2 * a)
    sol2 = (-b + cmath.sqrt(discriminant)) / (2 * a)

    return sol1, sol2

a = 2
m = 1
b = 6

mat = np.array(([a, m],[m,b]))

eig1, eig2 = solve_quadratic(1,-(mat[0,0]+mat[1,1]), mat[0,0]*mat[1,1]-mat[1,0]*mat[0,1])

eigenvalues, eigenvectors = np.linalg.eig(mat)

# m_a = 1
# m_b = -(a+b)
# m_c = a*b-m**2

# test1, test2 = solve_quadratic(m_a, m_b, m_c)

lambda1 = (a+b - np.sqrt( (a+b)**2 -4*(a*b-m**2)))/2
lambda2 = (a+b + np.sqrt( (a+b)**2 -4*(a*b-m**2)))/2 

# calculate eigenvectors
rhs = np.array([0,0])
x1,x2 = np.linalg.solve(mat-np.eye(2)*lambda1, rhs)


e1 = np.array([(a-b-np.sqrt((a-b)**2-4*m**2))/(2*m),1])
e2 = np.array([(a-b+np.sqrt((a-b)**2-4*m*+2))/(2*m),1])