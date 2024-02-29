import numpy as np

'''
Starting with the orientation of the eigenvectors
'''
# def subtract_identity(matrix, scalar):
#     n = len(matrix)
#     result = [[0] * n for _ in range(n)]
#     for i in range(n):
#         for j in range(n):
#             result[i][j] = matrix[i][j] - (scalar if i == j else 0)
#     return result



# def eig(mat):
#     eigenvalues, eigenvectors = np.linalg.eig(mat)
    
#     b = -mat[0,0] - mat[1,1]
#     c = mat[0,0]*mat[1,1] - mat[0,1]**2
    
    
#     eigenval1 = (-b + np.sqrt(b**2 - 4  * c)) / 2
#     eigenval2 = (-b - np.sqrt(b**2 - 4  * c)) / 2
    
#     for ev in [eigenval1, eigenval2]:
#         modified_mat = mat - np.dot(np.eye(2),ev)
#         solution = np.linalg.solve(modified_mat, np.array([0,0]))
    
    
    
    
def get_quadrant(eigenvectors):
    result = np.zeros(2)
    for i, vector in enumerate(eigenvectors.T):
        #   Quadrant I or IV
        if vector[0] > 0:
            if vector[1] > 0:
                result[i] = 1
            else:
                result[i] = 4
        else:
            if vector[1] > 0:
                result[i] = 2
            else:
                result[i] = 3
    # print(result)
    return result
    
    
minl = 10
maxl = [500, 500]
for i in range(10000):
    lx = np.array([np.random.randint(minl, maxl[0]),
                   np.random.randint(minl, maxl[1])])
    if lx[0] == lx[1]:
        lx[0] += 1
    ang = np.random.uniform(0,  2* np.pi)
    angle = np.rad2deg(ang)
    
    D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
    eigenvalues, eigenvectors = np.linalg.eig(M) 
    
    # After 100'000 test and forbidding identical corelation lengths, the second
    # quadrant always comes after the first one
    quadrants = get_quadrant(eigenvectors)
    
    lxmat = 1/np.sqrt(eigenvalues)
    
    if lxmat[0] < lxmat[1]:
        lxmat = np.flip(lxmat)
        eigenvectors = np.flip(eigenvectors, axis = 1)
        
    
    if eigenvectors[0,0] > 0:
        ang_test = np.pi/2 -np.arccos(np.dot(eigenvectors[:,0],np.array([0,1])))    
        case = 1
    else:
        if eigenvectors[1,0] > 0:
            ang_test = np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))
            case = 2
        else:
            ang_test = np.pi -np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))
            case = 3
    
            
    
    if lx[0] < lx[1]:
        lx_target = np.flip(lx)
        ang_target = (ang+ np.pi/2)%np.pi
    else: 
        lx_target = lx
        ang_target = ang
        
    tolerance = 0.1        
    if abs(ang_test - ang_target) < tolerance or abs(ang_test - (ang_target - np.pi)) < tolerance or abs(ang_test - (ang_target + np.pi)) < tolerance:
        pass
    else:
        print(f'wrong angle in case {case}')
        print(ang_test - ang_target)
    
    
        
    if np.round(lxmat[0]) == lx_target[0] and np.round(lxmat[1]) == lx_target[1]:
        pass
        # print(f'correct l extraction in quadrants {quadrants} with {format(angle, ".2f")}°')
    else:
        print(f'wrong extraction in quadrants {quadrants} with {format(angle, ".2f")}° and {lx}')
        
    # print(format(np.arccos(np.dot(eigenvectors[:,0],eigenvectors[:,1])), ".2f"))
    
    
    
    
    