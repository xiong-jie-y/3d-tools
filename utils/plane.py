# Copyright (c) 2019 s0972456
"""
plane.py
---------------

Functions related to plane segmentation.

"""
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.optimize import leastsq

def apply4x4(vec, mat):
    return np.dot(mat[:3, :3], vec) + mat[:3, 3]

def get_plane(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = -1 * np.dot(cp, p3)

    return np.array([a,b,c,d])

from sympy import Plane
from numpy.testing import assert_, assert_array_equal, assert_array_almost_equal

def plane_to_other_coord(plane, transform):
    xy = np.random.random((3, 2))
    z = (-plane[3] - np.dot(xy, plane[:2])) / plane[2]

    # import IPython; IPython.embed()

    # xyz =  np.concatenate([xy, z], axis=0)
    xyz = np.concatenate([xy, np.reshape(z, (3, 1))], axis=1)

    assert_array_almost_equal(np.dot(xyz, plane[:3]) + plane[3], np.array([0,0,0]))

    points = apply4x4(xyz, transform)
    assert_array_almost_equal(xyz, points)
    return get_plane(*points)

def f_min(X,p):
    plane_xyz = p[0:3]
    distance = (plane_xyz*X.T).sum(axis=1) + p[3]
    return distance / np.linalg.norm(plane_xyz)

def residuals(params, signal, X):
    return f_min(X, params)

def findplane(cad,d):
    p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]
    sol = None
    gray = cv2.cvtColor(cad, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    XYZ = [[],[],[]]
    if np.all(ids != None):
        for index,cornerset in enumerate(corners):
            cornerset = cornerset[0]
            for corner in cornerset:
                if d[int(corner[1])][int(corner[0])][2]!= 0:
                    XYZ[0].append(d[int(corner[1])][int(corner[0])][0])
                    XYZ[1].append(d[int(corner[1])][int(corner[0])][1])
                    XYZ[2].append(d[int(corner[1])][int(corner[0])][2])


        XYZ = np.asarray(XYZ)
        try:
            sol = leastsq(residuals, p0, args=(None, XYZ))[0]
        except TypeError:
            return None

    print(f"{len(XYZ[0])} points are used to detect pane.")

    return sol

def fitplane(p0,points):
  
    XYZ = np.asarray(points.T)
    sol = leastsq(residuals, p0, args=(None, XYZ))[0]

    return sol

def fitplane2(p0, points):
    sol = leastsq(residuals, p0, args=(None, points))[0]

    return sol

def point_to_plane(X,p):
    height,width,dim = X.shape
    X = np.reshape(X,(height*width,dim))
    plane_xyz = p[0:3]
    distance = (plane_xyz*X).sum(axis=1) + p[3]
    distance = distance / np.linalg.norm(plane_xyz)
    distance = np.reshape(distance,(height,width))
    return distance
