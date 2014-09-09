'''
Created on 03/giu/2013

@author: Christian
'''

from numpy import *
import math

def findBiggestTriangle(points):
    maxA = -1
    maxTrA = [None, None, None]
    for p1 in points:
        for p2 in points:
            for p3 in points:
                area = triangleArea(p1,p2,p3)
                if area > maxA:
                    maxA = area
                    maxTrA = [ p1, p2, p3 ]
    
    return maxTrA

def triangleArea(p1,p2,p3):
    p2p1 = array([ p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] ])
    l_p2p1 = linalg.norm(p2p1)
    p3p1 = array([ p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2] ])
    l_p3p1 = linalg.norm(p3p1)
    p3p2 = array([ p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2] ])
    l_p3p2 = linalg.norm(p3p2)
    s = (l_p2p1 + l_p3p1 + l_p3p2) / 2.0
    if (s * (s - l_p2p1) * (s - l_p3p1) * (s - l_p3p2)) > 0:
        area = math.sqrt( s * (s - l_p2p1) * (s - l_p3p1) * (s - l_p3p2) )
    else:
        area = 1e-6
    
    return area

def center(mtx):
    """translate all data (rows of the matrix) to center on the origin
  
    returns a shifted version of the input data.  The new matrix is such that
    the center of mass of the row vectors is centered at the origin. 
    Returns a float ('d') array
    """
    result = array(mtx, 'd')
    result -= mean(result, 0)
    # subtract each column's mean from each element in that column
    return result
  
def normalize(mtx):
    """change scaling of data (in rows) such that trace(mtx*mtx') = 1
  
    mtx' denotes the transpose of mtx """
    result = array(mtx, 'd')
    num_pts, num_dims = shape(result)
    mag = trace(dot(result, transpose(result)))
    norm = sqrt(mag)
    result /= norm
    return result

def computeNormal(self, temp):
    edge1 = [   temp[0][0] - temp[1][0], 
                temp[0][1] - temp[1][1], 
                temp[0][2] - temp[1][2] ]
                
    edge2 = [   temp[0][0] - temp[2][0], 
                temp[0][1] - temp[2][1], 
                temp[0][2] - temp[2][2] ]
                
    normal = [ edge1[1] * edge2[2] - edge1[2] * edge2[1], 
               edge1[2] * edge2[0] - edge1[0] * edge2[2], 
               edge1[0] * edge2[1] - edge1[1] * edge2[0] ]
                
    length = math.sqrt( normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] )
    if length > 0.0:
        normal = [ normal[0] / length, normal[1] / length, normal[2] / length ]
    else:
        normal = [ 0.0, 0.0, 0.0 ]
    
    return normal

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return R, t

def computeOBB(points, faces):
        vertices = points
        
        A_i = 0.0
        A_m = 0.0        
        
        mu = zeros((1,3), 'f')
        mu_i = zeros((1,3), 'f')
        
        covariance_matrix = zeros((3, 3), 'f')
        
        E_xx = 0.0
        E_xy = 0.0
        E_xz = 0.0
        
        E_yy = 0.0
        E_yz = 0.0
        
        E_zz = 0.0
        
        for f in faces:
            p = vertices[f[0]]
            q = vertices[f[1]]
            r = vertices[f[2]]
            
            mu_i = (p + q + r) / 3.0
            
            temp = cross((q - p), (r - p)) 
            A_i = linalg.norm(temp)
            A_i /= 2.0
            
            mu += mu_i * A_i
            A_m += A_i
    
            E_xx += ( 9.0 * mu_i[0] * mu_i[0] + p[0] * p[0] + q[0] * q[0] + r[0] * r[0] ) * (A_i/12.0)
            E_xy += ( 9.0 * mu_i[0] * mu_i[1] + p[0] * p[1] + q[0] * q[1] + r[0] * r[1] ) * (A_i/12.0)
            E_xz += ( 9.0 * mu_i[0] * mu_i[2] + p[0] * p[2] + q[0] * q[2] + r[0] * r[2] ) * (A_i/12.0)
            
            
            E_yy += ( 9.0 * mu_i[1] * mu_i[1] + p[1] * p[1] + q[1] * q[1] + r[1] * r[1] ) * (A_i/12.0)
            E_yz += ( 9.0 * mu_i[1] * mu_i[2] + p[1] * p[2] + q[1] * q[2] + r[1] * r[2] ) * (A_i/12.0)
            
            E_zz += ( 9.0 * mu_i[2] * mu_i[2] + p[2] * p[2] + q[2] * q[2] + r[2] * r[2] ) * (A_i/12.0)
        
        mu /= A_m
        E_xx /= A_m
        E_xy /= A_m
        E_xz /= A_m
        
        E_yy /= A_m
        E_yz /= A_m

        E_zz /= A_m
        
        E_xx -= mu[0,0] * mu[0,0] 
        E_xy -= mu[0,0] * mu[0,1] 
        E_xz -= mu[0,0] * mu[0,2]
        
        E_yy -= mu[0,1] * mu[0,1] 
        E_yz -= mu[0,1] * mu[0,2]
         
        E_zz -= mu[0,2] * mu[0,2]

        covariance_matrix[0][0] = E_xx
        covariance_matrix[0][1] = E_xy
        covariance_matrix[0][2] = E_xz
        covariance_matrix[1][0] = E_xy
        covariance_matrix[1][1] = E_yy
        covariance_matrix[1][2] = E_yz
        covariance_matrix[2][0] = E_xz
        covariance_matrix[2][1] = E_yz
        covariance_matrix[2][2] = E_zz
        
        covariance_matrix = matrix(covariance_matrix, 'f')

        w, v = linalg.eig(covariance_matrix)

        #Computing needed features from eigenvalues
        eigen_features = []
        s1 = float(w[0])
        s2 = float(w[1])
        s3 = float(w[2])
        s_tot = float(s1 + s2 + s3)
        
        eigen_features.append( s1 / s_tot )
        eigen_features.append( s2 / s_tot )
        eigen_features.append( s3 / s_tot )
        eigen_features.append( (s1 + s2) / s_tot )
        eigen_features.append( (s1 + s3) / s_tot )
        eigen_features.append( (s2 + s3) / s_tot )
        eigen_features.append( s1 / s2 )
        eigen_features.append( s1 / s3 )
        eigen_features.append( s2 / s3 )
        eigen_features.append( s1 / s2 + s1 / s3 )
        eigen_features.append( s1 / s2 + s2 / s3 )
        eigen_features.append( s1 / s3 + s2 / s3 )

        r = array([0.0, 0.0, 0.0], 'f')
        u = array([0.0, 0.0, 0.0], 'f')
        f = array([0.0, 0.0, 0.0], 'f')
        
        r[0] = v[0,0]
        r[1] = v[1,0]
        r[2] = v[2,0]
        
        u[0] = v[0,1]
        u[1] = v[1,1]
        u[2] = v[2,1]
        
        f[0] = v[0,2]
        f[1] = v[1,2]
        f[2] = v[2,2]
        
        r /= linalg.norm(r)
        u /= linalg.norm(u)
        f /= linalg.norm(f)

        transformation_matrix = zeros((3, 3), 'f')
        
        transformation_matrix[0,0] = r[0]
        transformation_matrix[0,1] = u[0]
        transformation_matrix[0,2] = f[0]
        
        transformation_matrix[1,0] = r[1]
        transformation_matrix[1,1] = u[1]
        transformation_matrix[1,2] = f[1]
        
        transformation_matrix[2,0] = r[2]
        transformation_matrix[2,1] = u[2]
        transformation_matrix[2,2] = f[2] 

        p_min = [1e10, 1e10, 1e10]
        p_max = [-1e10, -1e10, -1e10]
        
        for i in range(len(vertices)):
            temp = [0.0, 0.0, 0.0]
            temp[0] = dot( r, vertices[i] )
            temp[1] = dot( u, vertices[i] )
            temp[2] = dot( f, vertices[i] )
            
            if temp[0] > p_max[0]:
                p_max[0] = temp[0]
            if temp[1] > p_max[1]:
                p_max[1] = temp[1]
            if temp[2] > p_max[2]:
                p_max[2] = temp[2]
                
            if temp[0] < p_min[0]:
                p_min[0] = temp[0]
            if temp[1] < p_min[1]:
                p_min[1] = temp[1]
            if temp[2] < p_min[2]:
                p_min[2] = temp[2]
       
        p_max = array(p_max)
        p_min = array(p_min)

        delta = (p_max - p_min) / 2.0
        p_cen = (p_max + p_min) / 2.0
        
        translation = [0.0, 0.0, 0.0]
        translation[0] = dot( transformation_matrix[0], p_cen )
        translation[1] = dot( transformation_matrix[1], p_cen )
        translation[2] = dot( transformation_matrix[2], p_cen )
        translation = array(translation)
        
        p = [0.0,] * 8
        p[0] = (translation - r * delta[0] - u * delta[1] - f * delta[2]).tolist()
        p[1] = (translation + r * delta[0] - u * delta[1] - f * delta[2]).tolist()
        p[2] = (translation + r * delta[0] - u * delta[1] + f * delta[2]).tolist()
        p[3] = (translation - r * delta[0] - u * delta[1] + f * delta[2]).tolist()
        p[4] = (translation - r * delta[0] + u * delta[1] - f * delta[2]).tolist()
        p[5] = (translation + r * delta[0] + u * delta[1] - f * delta[2]).tolist()
        p[6] = (translation + r * delta[0] + u * delta[1] + f * delta[2]).tolist()
        p[7] = (translation - r * delta[0] + u * delta[1] + f * delta[2]).tolist()

        return [p, eigen_features]