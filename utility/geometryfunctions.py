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
    Returns a numpy float ('d') array
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