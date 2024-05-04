#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
Mathematical and geometrical functions
"""
# standard library
import array as array
import random
import logging
from math import sqrt, pi, sin, cos, floor, atan2
# third-party
import numpy as np
from numpy.random import default_rng
import astropy
import astropy.units as u
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.stats import qmc
from spherical_geometry import polygon as spherical_polygon


def rotate(origin, px, py, angle, clockwise=False):
    """
    Rotate a point counterclockwise by default by
    a given angle around a given origin.

    Parameters
    ----------

    origin : `astropy.units.quantity.Quantity` length (mm)
        the coordinates (x,y) of the point of origin for the rotation

    px : `astropy.units.quantity.Quantity` length (mm)

    py : `astropy.units.quantity.Quantity` length (mm)

    angle : `astropy.units.quantity.Quantity` angle (degrees of radians)
        The angle of rotation

    clockwise : boolean
        if one requires the rotation to be clockwise, set clockwise=True

    Returns
    -------
    Two numpy array of the same length than the input point x, y

    Notes
    -----
    https://en.wikipedia.org/wiki/Rotation_matrix

    Example
    -------
    >>> from geometry_utils import rotate
    >>> import numpy as np
    >>> import astropy.units as u
    >>> origin = np.array([0., 0.])
    >>> point = np.array([3., 4.])
    >>> angle = 13 * u.degree
    >>> rotate(origin, point[0], point[1], angle)
    (<Quantity 2.02330598>, <Quantity 4.57233342>)

    >>> points = np.array([[3., 4.],[2., 3.],[4., 5.]])
    >>> x = points.T[0] * u.mm
    >>> y = points.T[1] * u.mm
    >>> angle = 13.*u.degree
    >>> r = rotate(origin, x, y, angle)
    >>> r[0]
    <Quantity [2.02330598, 1.27388697, 2.77272499] mm>

    >>> r[1]
    <Quantity [4.57233342, 3.3730123 , 5.77165454] mm>

    >>> rotate(origin, point[0], point[1], angle, clockwise=True)
    (<Quantity 3.82291441>, <Quantity 3.2226271>)

    >>> rotate(origin, point[0], point[1], -angle)
    (<Quantity 3.82291441>, <Quantity 3.2226271>)
    """
    ox, oy = origin
    sign = 1 - 2 * clockwise  # -1 if clockwise, 1 if counterclockwise
    angle = sign * angle
    cosa = np.cos(angle)  # angle has to be a Quantity
    sina = np.sin(angle)
    dx = px - ox
    dy = py - oy
    qx = ox + cosa * dx - sina * dy
    qy = oy + sina * dx + cosa * dy
    return qx, qy


def inside_radius(center_x, center_y, radius, px, py):
    """
    Testing if a point is inside a circle

    Parameters
    ----------
    center_x : float or `astropy.units.quantity.Quantity` length
        the x position of the center of the circle

    center_y : float or `astropy.units.quantity.Quantity` length
        the y position of the center of the circle

    radius : float or `astropy.units.quantity.Quantity` length
        the radius of the circle

    px : float or `astropy.units.quantity.Quantity` lenght (mm)
        x values of the positions to be tested

    py : float or `astropy.units.quantity.Quantity` length (mm)
        y values of the positions to be tested

    Returns
    -------
        a boolean or an Numpy array of booleans
        an array with the distance or NaN if the point is outside the
        search radius

    Note: see
    https://stackoverflow.com/questions/481144/
    equation-for-testing-if-a-point-is-inside-a-circle

    Examples
    --------
    >>> import astropy.units as u
    >>> from geometry_utils import inside_radius
    >>> import numpy as np
    >>> inside_radius(0, 0, 1., 0.5, 0.6)
    (True, 0.7810249675906654)

    >>> x = np.array([1, 1.5, 1, 0.7, 0.8, 0.49])
    >>> y = np.array([0, 1.1, 1, 0.6, 0.8, 0.51])
    >>> b, d = inside_radius(0.,0, 1., x, y)
    >>> b
    array([ True, False, False,  True, False,  True])
    >>> d[0]
    1.0
    >>> x *= u.mm
    >>> y *= u.mm
    >>> b, d = inside_radius(0. * u.mm, 0 * u.mm, 1. * u.mm, x, y)
    >>> b
    array([ True, False, False,  True, False,  True])
    >>> d[0]
    <Quantity 1. mm>
    """
    pQuantity = isinstance(px, u.Quantity)
    if pQuantity:
        is_scalar = px.isscalar
    else:
        is_scalar = False
    if isinstance(px, (float, np.float_)) or is_scalar:
        # if a single point
        dx = abs(px - center_x)
        dy = abs(py - center_y)
        if dx > radius:
            return False, np.NaN
        if dy > radius:
            return False, np.NaN
        if dx + dy <= radius:
            return True, np.sqrt(dx**2 + dy**2)
        if dx**2 + dy**2 <= radius**2:
            return True, np.sqrt(dx**2 + dy**2)
        else:
            return False, np.NaN
    else:  # for an array of points
        if isinstance(px, list):
            px = np.array(px)  # to ensure that px and py are Numpy arrays
        if isinstance(py, list):
            py = np.array(py)
        dx = np.abs(px - center_x)
        dy = np.abs(py - center_y)
        dist = np.full(len(px), np.NaN)
        # inside the square surrounding the circle?
        c1 = (dx <= radius) & (dy <= radius)
        c2 = np.copy(c1)
        c3 = dx[c1] + dy[c1] <= radius  # inside the square inside the circle?
        c2[c1] = ~c3
        c3[~c3] = dx[c2]**2 + dy[c2]**2 < radius**2  # Pythagoras test
        c2[c1] = c3
        dist[~c2] = np.NaN
        dist[c2] = np.sqrt(dx[c2]**2 + dy[c2]**2)
        if pQuantity:  # put back the units if px has a unit
            dist *= u.Unit(px.unit)
        return c2, dist


def inside_polygon(x, y, points):
    """
    Test is the point(s) with coordinates x, y are
    within the polygon defined by points

    Parameters
    ----------
    x, y : floats or numpy array of floats
        the coordinates to be checked

    points : floats
        the coodinates of the vertices of the polygone

    Return
    ------
        boolean or numpy array of booleans

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from geometry_utils import inside_polygon, hexagon_vertices
    >>> r = 1
    >>> points = hexagon_vertices(r, 0.)
    >>> pos = (0.3860175015714546, 0.8447416146866873)
    >>> inside_polygon(pos[0], pos[1], points)
    True
    >>> pos_x = np.array([0.3860175015714546, 1.001])
    >>> pos_y = np.array([0.8447416146866873, 0.])
    >>> inside_polygon(pos_x, pos_y, points)
    array([ True, False])
    >>> inside_polygon(pos[0] * u.mm, pos[1] * u.mm, points * u.mm)
    True
    >>> inside_polygon(pos_x * u.mm, pos_y * u.mm, points * u.mm)
    array([ True, False])
    """
    array = False
    if isinstance(x, u.quantity.Quantity):
        if (not x.isscalar):
            array = True
    elif isinstance(x, np.ndarray):
        array = True
    if array:
        inside = []
        for xv, yv in zip(x, y):
            inside.append(inside_polygon_one(xv, yv, points))
        return np.array(inside)
    else:
        return inside_polygon_one(x, y, points)


def quasi_random_disk_sampler(r0=1, n=1):
    """
    Quasi random sampler of points inside a disk
    of radius r0 with defautl r0=1

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.neighbors import BallTree
    >>> from geometry_utils import quasi_random_disk_sampler
    >>> x, y = quasi_random_disk_sampler(n=160, r0=230)
    >>> X = np.transpose([x, y])
    >>> tree = BallTree(X)
    >>> dist, ind = tree.query(X[:-1], k=3)
    >>> d = np.transpose(dist)
    >>> np.mean(d)
    >>> np.std(d)
    >>> rad = 2. * np.mean(d)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> plt.scatter(x, y, s=3)
    >>> for x1, y1 in X:
    ...     circle = plt.Circle((x1, y1), rad, color='r', fill=False)
    ...     ax.add_patch(circle)
    >>> plt.show()
    """
    sampler = qmc.Sobol(d=2, scramble=True)
    m = int(np.log2(n)) + 1
    u, v = np.transpose(sampler.random_base2(m=m))
    u = u[0:n]
    v = v[0:n]
    r = np.sqrt(u)
    theta = 2. * np.pi * v
    x = r0 * r * np.cos(theta)
    y = r0 * r * np.sin(theta)
    return x, y


def latin_hypercube_disk_sampler(r0=1, n=1):
    """
    Latin hypercube sampler of points inside a disk
    of radius r0 with defautl r0=1

    Example
    -------
    >>> from sklearn.neighbors import BallTree
    >>> from geometry_utils import quasi_random_disk_sampler
    >>> x, y = latin_hypercube_disk_sampler(n=160, r0=230)
    >>> X = np.transpose([x, y])
    >>> tree = BallTree(X)
    >>> dist, ind = tree.query(X[:-1], k=3)
    >>> d = np.transpose(dist)
    >>> np.mean(d)
    >>> np.std(d)
    >>> rad = 2. * np.mean(d)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(6, 6))
    >>> plt.scatter(x, y, s=3)
    >>> for x1, y1 in X:
    ...     circle = plt.Circle((x1, y1), rad, color='r', fill=False)
    ...     ax.add_patch(circle)
    >>> plt.show()
    """
    sampler = qmc.LatinHypercube(d=2)
    u, v = np.transpose(sampler.random(n=n))
    u = u[0:n]
    v = v[0:n]
    r = np.sqrt(u)
    theta = 2. * np.pi * v
    x = r0 * r * np.cos(theta)
    y = r0 * r * np.sin(theta)
    return x, y


def inside_polygon_one(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of vertices [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference:
    http://www.ariel.com.au/a/python-point-int-poly.html
    https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
    https://demonstrations.wolfram.com/
    AnEfficientTestForAPointToBeInAConvexPolygon/

    Michael Milgram Does a point lie inside a polygon
        September 1989Journal of Computational Physics 84(1)
        DOI: 10.1016/0021-9991(89)90185-X
    https://www.researchgate.net/publication/
    255536302_Does_a_point_lie_inside_a_polygon

    Parameters
    ----------
    x, y : floats
        the coordinates to be checked

    points : floats
        the coodinates of the vertices of the polygone

    Return
    ------
        boolean

    Examples
    --------
    see inside_polygon
    >>> from geometry_utils import hexagon_vertices, inside_polygon_one
    >>> r = 1
    >>> points = hexagon_vertices(r, 0.)
    >>> pos = (0.3860175015714546, 0.8447416146866873)
    >>> inside_polygon_one(pos[0], pos[1], points)
    True
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def skew_line_distance(start1, end1, start2, end2):
    '''
    Name: skew_line_distance

    Aim: compute the minimum distance between two segments. Use by collision

    Description of the algorithm:
    https://www.quora.com/
    How-do-I-find-the-shortest-distance-between-two-skew-lines

    The shortest distance between a line, L,  and a point, P, is the length
    of the line that is perpendicular to L and goes from a point on L to the
    point P.
    This general idea of perpendicular lines also applies for two lines,
    except our new line must be perpendicular to both.

    This is all a little bit easier to do in vector form so that is how I
    will give it.
    let: L1 be r = a + sb , and L2 be r = c + td.
    where r is the position vector of any point on a line; a and c are the
    position vectors of a point that lies on L1 and L2 respectively; s and
    t are scalars; and b and d are vectors parallel to L1 and L2
    respectively.

    A vector perpendicular to both L1 and L2 is the cross product of the two
    directional vectors:
                            n = b x d
    and the unit vector, n hat, is n divided by the magnitude of n:
                    n hat = (b x d) / |b x d|
    If P is a point on L1 and Q is a point on L2, then we can find the
    vector QP by finding the difference between (a +sb) and (c+td) since
    these give the position vectors of any point on the lines:
                    QP = a - c + sb - td
    The shortest distance between L1 and L2 is the projection of QP on n.
    That is, the dot product of QP and n, all divided by the magnitude of n.
    We already found n divided by the magnitude of n as n hat. So:
    Shortest distance = QP . n hat

    = (a - c + sb - td) . ((b x d)/|b x d|)
    = ((a - c + sb - td) . (b x d))  / |b x d|
    = ((a - c) . (b x d) + sb . (b x d) - td . (b x d))  / |b x d|

    Since b . (b x d) = d . (b x d) = 0  (the projection of x or y on (x x y)
    is actually always zero),
    we can reduce that equation to:
            Shortest distance =| (a - c) . (b x d)  / |b x d| |
    Dot product and division are both linear (as seen in the simplification
    of QP . n hat) so you can use this equation in any order.

    Gellert, W.; Gottwald, S.; Hellwich, M.; Kästner, H.; and Künstner,
    H. (Eds.).
    VNR Concise Encyclopedia of Mathematics, 2nd ed. New York:
    Van Nostrand Reinhold, 1989.
    Hill, F. S. Jr. The Pleasures of Perp Dot Products. Ch. II.5
    in Graphics Gems IV (Ed. P. S. Heckbert). San Diego: Academic Press,
    pp. 138-148, 1994.
    start1 : [xs1,ys1,zs1]
    end1   : [xe1,ye1,ze1]
    start2 : [xs2,ys2,zs2]
    end2   : [xe1,xe2,xe3]
    '''
    SMALL_NUM = 0.00000001
    a = end1 - start1
    b = end2 - start2
    c = start2 - start1
    cross = np.cross(a, b)  # cross product, np.dot: dot product
    norm_cross = np.linalg.norm(cross)
    # the norm of the cross product is null, use the other method in that case
    if (norm_cross < SMALL_NUM):
        # when the segments are almost parallel
        dist = DistBetween2Segments2(start1, end1, start2, end2)
    else:
        # minimum distance
        dist = np.linalg.norm(np.dot(c, cross)) / norm_cross
    return dist


def DistBetween2Segments2(p1, p2, p3, p4):
    '''
    Name: DistBetween2Segments2

    work one input at the time
    strip down version with only the distance as output
    DistBetween2Segments is the full version

    adapted from a Matlab code, which in turn is an adaptation of a C++ code

    Aim: Computes the minimum distance between two line segments.

    Code is adapted for Matlab from Dan Sunday's Geometry Algorithms originally
    written in C++
    http://softsurfer.com/Archive/algorithm_0106/
    algorithm_0106.htm#dist3D_Segment_to_Segment

    Usage: Input the start and end x,y,z coordinates for two line segments.
    p1, p2 are [x,y,z] coordinates of first line segment and p3, p4 are for
    second line segment.

    Output: scalar minimum distance between the two segments.

    Matlab Example:
    P1 = [0 0 0];     P2 = [1 0 0];
    P3 = [0 1 0];     P4 = [1 1 0];
    dist = DistBetween2Segments2(P1, P2, P3, P4)
    dist = 1

    version 0.9 5/10/2018 WFT

    Paramters
    ---------
    p1, p2 : numpy array
         [x, y, z] coordinates of first line segment

    p3, p4 : numpy array
         [x, y, z] coordinates of second line segment

    Returns
    -------
    d : float
        the distance between the two segments

    Example
    -------
    >>> import numpy as np
    >>> from geometry_utils import DistBetween2Segments2
    >>> P1 = np.array([0, 0, 0])
    >>> P2 = np.array([1, 0, 0])
    >>> P3 = np.array([0, 1, 0])
    >>> P4 = np.array([1, 1, 0])
    >>> dist = DistBetween2Segments2(P1, P2, P3, P4)
    >>> dist
    1.0
    '''
    u = p1 - p2
    v = p3 - p4
    w = p2 - p4

    # a,b,c,d,e=map(lambda x,y: np.dot(x,y),[u,u,v,u,v],[u,v,v,w,w]) # slowest

    # a = np.dot(u,u) # slow
    # b = np.dot(u,v)
    # c = np.dot(v,v)
    # d = np.dot(u,w)
    # e = np.dot(v,w)

    a = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]  # explicit: fastest
    b = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    c = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    d = u[0] * w[0] + u[1] * w[1] + u[2] * w[2]
    e = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]

    D = a * c - b * b
    sD = D
    tD = D

    SMALL_NUM = 0.00000001

    # compute the line parameters of the two closest points
    if (D < SMALL_NUM):   # the lines are almost parallel
        sN = 0.0          # force using point P0 on segment S1
        sD = 1.0          # to prevent possible division by 0.0 later
        tN = e
        tD = c
    else:                # get the closest points on the infinite lines
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if (sN < 0.0):   # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif (sN > sD):  # sc > 1 => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c

    if (tN < 0.0):           # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if (-d < 0.0):
            sN = 0.0
        elif (-d > a):
            sN = sD
        else:
            sN = -d
            sD = a
    elif (tN > tD):       # tc > 1 => the t=1 edge is visible
        tN = tD           # This part is not tested yet 2022/2/4/ WFT
        # recompute sc for this edge
        if ((-d + b) < 0.0):
            sN = 0
        elif ((-d + b) > a):
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    # finally do the division to get sc and tc
    if (-1.0 * SMALL_NUM < sN < SMALL_NUM):
        sc = 0.0
    else:
        sc = sN / sD

    if (-1.0 * SMALL_NUM < tN < SMALL_NUM):
        tc = 0.0
    else:
        tc = tN / tD

    # get the difference of the two closest points

    f = w + (sc * u) - (tc * v)
    return sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])  # = S1(sc) - S2(tc)


def closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False,
                                clampA0=False, clampA1=False,
                                clampB0=False, clampB1=False):

    """
    Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    https://stackoverflow.com/questions/2824478/
    shortest-distance-between-two-line-segments

    Parameters
    ----------
    a0, a1 : numpy array
         [x, y, z] coordinates of first line segment

    b0, b1 : numpy array
         [x, y, z] coordinates of second line segment

    Clamp: Optional input
        if true : it means that the segment stops at that point
        clamp = False

    Returns
    -------
    pa : array of 3 floats
        the coordinates of the closest point on segment 1

    pb : array of 3 floats
        the coordinates of the closest point on segment 2

    d : float
        the distance between the two segments

    Example
    -------
    >>> import numpy as np
    >>> from geometry_utils import DistBetween2Segments2
    >>> from geometry_utils import skew_line_distance
    >>> from geometry_utils import closestDistanceBetweenLines
    >>> a1 = np.array([13.43, 21.77, 46.81])
    >>> a0 = np.array([27.83, 31.74, -26.60])
    >>> b0 = np.array([77.54, 7.53, 6.22])
    >>> b1 = np.array([26.99, 12.39, 11.18])
    >>> _, _, d = closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=True)
    >>> # Result: (array([ 20.29994362,  26.5264818 ,  11.78759994]),
    >>> # array([ 26.99,  12.39,  11.18]), 15.651394495590445)
    >>> d
    15.651394495590443
    >>> _, _, d = closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False)
    >>> # Result: (array([ 19.85163563,  26.21609078,  14.07303667]),
    >>> # array([ 18.40058604,  13.21580716,  12.02279907]),
    >>> # 13.240709703623198)
    >>> d
    13.240709703623203
    >>> DistBetween2Segments2(a0, a1, b0, b1)
    15.651394495590445
    >>> skew_line_distance(a0, a1, b0, b1)
    13.240709703623201
    >>> a0 = np.array([-228.8655,  -135.9499, 200])
    >>> a1 = np.array([-228.8655,  -135.9499, 0])  # in the focal plane
    >>> b0 = np.array([-228.8791,  -126.4121, 200])
    >>> b1 = np.array([-228.8655,  -135.9499, 0])  # in the focal plane
    >>> closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=True)
    """

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions,
    # but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:  # No test case here 2022/2/4 WFT
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))  # No test case here 2022/2/4 WFT
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)


def inside_hexagon1(s, pos):
    """
    Check if a point is inside a hexagon of side s centered at (0,0)
    Using symmetry to get into the first quadrant and then simple math
    (Pythagoras is enough) to check whether the point is both "below the
    diagonal" (which has y = sqrt(3) ⋅ (s - x)) and "below the top edge"
    (which has y = sqrt(3)/2 ⋅ s).

    https://stackoverflow.com/questions/42903609/
    function-to-determine-if-point-is-inside-hexagon

    Parameters
    ----------
    s : float
        side of the haxagon

    pos : array like with shape (2, N), where N is the number of positions
        to be tested.

    Returns
    -------
    boolean or an array of booleans

    Examples
    --------
    >>> from geometry_utils import inside_hexagon1
    >>> pos = (0.3860175015714546, 0.8447416146866873)
    >>> inside_hexagon1(1., pos)
    True
    """
    x, y = map(abs, pos)
    return y < 3**0.5 * np.fmin(s - x, s / 2)


def hexagon(theta):
    """
    Calculate the distance from the center of a hexagon at angle theta
    with theta in radians

    It uses the polar definition of a hexagon

    https://stackoverflow.com/questions/42903609/
    function-to-determine-if-point-is-inside-hexagon

    https://mindblownmath.wordpress.com/2014/02/02/polar-equation-of-a-hexagon/

    Parameters
    ----------
    theta : float (radian)
        angle

    Returns
    -------
    d : float
        the distance to the center of the hexagon

    Example
    -------
    >>> import numpy as np
    >>> from numpy.testing import assert_allclose
    >>> from geometry_utils import hexagon
    >>> a = [hexagon(theta * np.pi / 3.) for theta in range(6)]
    >>> assert_allclose(a, np.repeat(1, 6))
    True
    """
    sixty_d = pi / 3.
    return sqrt(3) / (2. * sin(theta + sixty_d -
                      sixty_d * floor(theta / sixty_d)))


def inside_hexagon2(radius, pos, PA):
    """
    Check if a point is inside a hexagon of side s centered at (0,0)
    rotated anti-clockwise by an angle PA using the polar definition
    of a hexagon (squircle)

    The side of an hexagon is half of the diameter, i.e the the distance
    from the center to one of the corner is s

    PA in radian
    The anti-clockwise rotation angle of the hexagon

    Examples
    --------
    >>> from math import pi, atan2
    >>> from geometry_utils import inside_hexagon2
    >>> pos = (0.3860175015714546, 0.8447416146866873)
    >>> inside_hexagon2(1., pos, 0.0)
    True
    """
    x, y = pos
    theta = atan2(y, x) - PA * pi / 180.
    return x**2 + y**2 <= (radius * hexagon(theta))**2


def random_circle(nb, center_x, center_y, radius, seed=2022):
    """
    Generate n random positions inside a circle centered at
    (center_x, center_y)

    Parameters
    ----------
    nb : int
        the number of random points

    center_x : float or `astropy.units.quantity.Quantity` length
        the x position of the center of the circle

    center_y : float or `astropy.units.quantity.Quantity` length
        the y position of the center of the circle

    radius : float or `astropy.units.quantity.Quantity` length

    seed : int, optional
        the seed for the random number generator, default = 2022

    Returns
    -------
    x : numpy array or `astropy.units.quantity.Quantity` array of length n
        the b random x coordinates

    y : numpy array or `astropy.units.quantity.Quantity` array of length n
        the b random y coordinates

    Examples
    --------
    >>> from geometry_utils import *
    >>> import astropy.units as u
    >>> from sklearn.neighbors import BallTree
    >>> nb = 1000
    >>> radius = 10 * u.mm
    >>> center_x = 0 * u.mm
    >>> center_y = 0 * u.mm
    >>> x, y = random_circle(nb, center_x, center_y, radius, seed=2022)
    >>> r = x * x + y * y
    >>> all(r <= radius**2)
    True
    >>> nb1, x1 = np.histogram(y)
    >>> nb2, x2 = np.histogram(y)
    >>> X = np.transpose([x,y])
    >>> tree = BallTree(X)
    >>> dist, ind = tree.query(X, k=2)
    >>> np.transpose(dist)[1].std()
    >>> center_x = 5 * u.mm
    >>> center_y = -2 * u.mm
    >>> x, y = random_circle(nb, center_x, center_y, radius, seed=2022)
    >>> ok, _ = inside_radius(center_x, center_y, radius, x, y)
    >>> all(ok)
    True
    """
    rng = np.random.default_rng(seed=seed)
    if isinstance(radius, astropy.units.quantity.Quantity):
        units = radius.unit
        quantity = True
        r = radius.value
        cx = center_x.value
        cy = center_y.value
    else:
        quantity = False
        r = radius
        cx = center_x
        cy = center_y
    i = 0
    xpoints, ypoints = [], []
    while (i < nb):
        x, y = 2. * (rng.random(2) - 0.5) * [r, r] + [cx, cy]
        ok, _ = inside_radius(cx, cy, r, x, y)
        if ok:
            xpoints.append(x)
            ypoints.append(y)
            i += 1
    if quantity:
        return xpoints * units, ypoints * units
    else:
        return np.array(xpoints), np.array(ypoints)


def random_circle_cluster(nb, center_x, center_y, radius, seed=2022):
    """
    Generate n random positions inside a circle centered at
    (center_x, center_y)

    This method gives a clustering in the center

    Parameters
    ----------
    nb : int
        the number of random points

    center_x : float or `astropy.units.quantity.Quantity` length
        the x position of the center of the circle

    center_y : float or `astropy.units.quantity.Quantity` length
        the y position of the center of the circle

    radius : float or `astropy.units.quantity.Quantity` length

    seed : int, optional
        the seed for the random number generator, default = 2022

    Returns
    -------
    x : numpy array or `astropy.units.quantity.Quantity` array of length n
        the b random x coordinates

    y : numpy array or `astropy.units.quantity.Quantity` array of length n
        the b random y coordinates

    Examples
    --------
    >>> from geometry_utils import *
    >>> import astropy.units as u
    >>> nb = 1000
    >>> radius = 10 * u.mm
    >>> center_x = 0 * u.mm
    >>> center_y = 0 * u.mm
    >>> x, y = random_circle_cluster(nb, center_x, center_y,
    ...                              radius, seed=2022)
    >>> r = x * x + y * y
    >>> all(r <= radius**2)
    True
    >>> center_x = 5 * u.mm
    >>> center_y = -2 * u.mm
    >>> x, y = random_circle_cluster(nb, center_x, center_y,
    ...                              radius, seed=2022)
    >>> ok, _ = inside_radius(center_x, center_y, radius, x, y)
    >>> all(ok)
    True
    >>> X = np.transpose([x,y])
    >>> tree = BallTree(X)
    >>> dist, ind = tree.query(X, k=2)
    >>> np.transpose(dist)[1].std()

    """
    rng = np.random.default_rng(seed=seed)
    r = rng.random(nb) * radius
    theta = rng.random(nb) * 2 * np.pi  # theta between 0 and < 2 * np.pi
    x = r * np.cos(theta) + center_x
    y = r * np.sin(theta) + center_y
    return x, y


def random_hexagon(radius, PA, n=1, outside=False, seed=2022):
    """
    Generate a random position inside a hexagon of radius radius

    Basically, it generates a random position and check if it is
    inside the haxagon.

    radius : float or `astropy.units.quantity.Quantity` length
        the "radius" of the hexagon

    PA : float or  `astropy.units.quantity.Quantity` degree
        the position angle of the haxagon

    outside : bool, optional
        if True, generate points located in the region between the
        hexagon and the circle of radius radius, default = False
        if False, generate points inside the hexagon

    seed : int, optional
        the seed for the random number generator, default = 2022

    n : int, optional
        the number of output points, default = 1

    Returns
    -------

    x : float
        the x position(s)

    y : float
        the y position(s)

    Examples
    --------
    >>> from geometry_utils import *
    >>> import matplotlib.pyplot as plt
    >>> check2 = []
    >>> r = 1.0
    >>> PA = 20.0
    >>> v = hexagon_vertices(r, PA)
    >>> pos = random_hexagon(r, PA, n=10000, outside=False)
    >>> for i in range(10000):
    ...     check2.append(inside_polygon(pos[0][i], pos[1][i], v))
    >>> all(check2)

    plt.scatter(x, y, s=5)
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.show()
    """
    random.seed(seed)
    x, y = [], []
    for i in range(n):
        while True:
            x1 = random.uniform(-radius, radius)
            y1 = random.uniform(-radius, radius)
            if inside_hexagon2(radius, [x1, y1], PA) != outside:
                break
        x.append(x1)
        y.append(y1)
    if n == 1:
        return x[0], y[0]
    else:
        return x, y


def hexagon_vertices(r, PA):
    """
    Return the coordinates for the hexagon vertices with radius r

    Obtained by applying sin(0), sin(60), sin(120), sin(180), sin(240),
    sin(300) when PA = 0 degree

    Parameters
    ----------
    r : float
        the hexagon "radius"

    PA : float
        the position angle of the hexagon

    Returns
    -------
    vertices : list of vertices
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5], [x6, y6]])

    Example
    -------
    >>> from geometry_utils import *
    >>> vertices = hexagon_vertices(1.0, 0)
    >>> inside_hexagon2(1., [0.99, 0.], 0)
    True
    >>> inside_hexagon1(1., [0.99, 0.])
    True
    """
    return [[r * cos(a), r * sin(a)]
            for a in (np.arange(0, 6, 1) * 60 + PA) * pi / 180]


def inside_rectangle(x, y, points):
    """
    Test if points (x, y) are inside a rectangle (square)
    with corner described by points

    Notes
    -----
    This routine is much faster than inside_polygon for rectangles
    and squares.

    Parameters
    ----------
    x, y : floats or array like with length N, where N is the number of
        positions to be tested or `astropy.units.quantity.Quantity` length
        the opposite coodinates of the rectangle

    points : array-like with 4 elements or `astropy.units.quantity.Quantity`
             length
        the coordinates of the rectangle

    Returns
    -------
    boolean if the input is a single point
    Numpy array of booleans for multiple points

    Examples
    --------
    >>> import astropy.units as u
    >>> from geometry_utils import inside_rectangle
    >>> import numpy as np
    >>> import array as array
    >>> points = (0, 0, 1, 1)
    >>> inside_rectangle(0.5, 0.5, points)
    True
    >>> inside_rectangle([0.5, 1.5], [0.8, 0.5], points)
    array([ True, False])
    >>> x = array.array('d', [0.5, 1.5])
    >>> y = array.array('d', [0.8, 0.5])
    >>> inside_rectangle(x, y, points)
    array([ True, False])
    >>> x = [0.5, 1.5] * u.mm
    >>> y = [0.8, 0.5] * u.mm
    >>> points = [0, 0, 1, 1] * u.mm
    >>> inside_rectangle(x, y, points)
    array([ True, False])
    >>> points = [0, 1, 1, 2]
    >>> inside_rectangle(1.5, 1.5, points)
    False
    >>> inside_rectangle(0.5, 1.5, points)
    True
    """
    if (isinstance(points, u.Quantity) &
       isinstance(x, u.Quantity) &
       isinstance(y, u.Quantity)):
        points = points.to(u.mm)
        x = x.to(u.mm)
        y = y.to(u.mm)
    x1, y1, x2, y2 = points
    if isinstance(x, (list, tuple, array.array)):
        x = np.array(x)
        y = np.array(y)
    return (x1 < x) & (x < x2) & (y1 < y) & (y < y2)


def dist2D(x1, x2, y1, y2):
    """
    Compute the distance between two points in
    the plane

    Parameters
    ----------
    x1 : `astropy.units.quantity.Quantity` in mm
        the x coordinate of the first point

    y1 : `astropy.units.quantity.Quantity` in mm
        the y coordinate of the first point

    x2 : `astropy.units.quantity.Quantity` in mm
        the x coordinate of the second point

    y2 : `astropy.units.quantity.Quantity` in mm
        the y coordinate of the second point

    Returns
     -------
    d : `astropy.units.quantity.Quantity` in mm
        the distance

    Examples
    --------
    >>> from geometry_utils import dist2D
    >>> dist2D(2, 0, 2, 0)
    2.8284271247461903
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def weighted_shuffle(weights, seed=2021):
    """
    Name: weigthed_shuffle
    Aim: Weighted Random Sampling (2005; Efraimidis, Spirakis provides a very
    elegant algorithm for this).
    http://utopia.duth.gr/%7Epefraimi/research/data/2007EncOfAlg.pdf

    In weighted random sampling (WRS) the items are weighted and the
    probability of each item to be selected is determined by its relative
    weight.

    The implementation runs in O(n log(n)):
    if the input vector is items with weights, items=[items[i] for i in order]
    is the weighted random shuffle

    References
    ----------

    https://softwareengineering.stackexchange.com/
    questions/233541/how-to-implement-a-weighted-shuffle

    https://link.springer.com/referenceworkentry/10.1007/978-1-4939-2864-4_478

    Efraimidis P, Spirakis P (2006) Weighted random sampling with a reservoir.
    Inf Process Lett J 97(5):181-185

    Parameters
    ----------
    weights : numpy array of floats
        the array of weights

    seed : int, optional
        the seed for the random generator, default = 2021

    Returns
    -------
    order : numpy array of integer indices
        The ordering is base on a random weighted shuffling

    Examples
    --------
    >>> import timeit
    >>> import random
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geometry_utils import weighted_shuffle
    >>> weights = np.array([0.5, 1.2, 0.7, 3., 0.01])
    >>> weighted_shuffle(weights)
    array([1, 0, 3, 2, 4])
    >>> weighted_shuffle(weights, seed=12324)
    array([3, 1, 2, 0, 4])
    >>> setup='''
    ... import numpy as np
    ... from geometry_utils import weighted_shuffle
    ... w = np.array([0.5, 1.2, 0.7, 3., 0.01])
    ... '''
    >>> n = 10000
    >>> t = timeit.timeit(stmt='weighted_shuffle(w,seed=122)',
                          setup=setup, number=n)
    >>> # print(t / n)
    >>> # 1.061536330000763e-05
    >>> # Multiple runs with different seeds
    >>> number = 100
    >>> weights = np.array([0.5, 1.2, 0.7, 3., 0.01])
    >>> output = []
    >>> for seed in range(number):
    ...    random.seed(seed)
    ...    output.append(weighted_shuffle(weights, seed=seed))
    >>> by_column = np.transpose(np.array(output))
    >>> stat = np.histogram(by_column[0], bins=[0, 1, 2, 3, 4, 5])[0]
    >>> stat = stat / stat.sum()
    >>> norm_weights = weights / weights.sum()
    >>> perc = (norm_weights - stat) / norm_weights * 100.
    >>> perc2 = (norm_weights - stat) * 100.
    >>> print(stat)
    >>> print(norm_weights)
    >>> print(perc)
    >>> print(perc2)
    >>> plt.scatter(norm_weights, perc2)
    >>> plt.xlabel('Normalized weight')
    >>> plt.ylabel('(norm_weights - freq.) * 100.')
    >>> plt.title(str(number) + ' realisations ')
    >>> plt.show()
    """
    if isinstance(weights, float):
        weights = [weights]
    if len(weights) == 1:
        return np.array([0])
    random.seed(seed)
    order = sorted(range(len(weights)),
                   key=lambda i: -random.random() ** (1.0 / weights[i]))
    return np.array(order)


def random_weighted_choice_std(weights, k=1, seed=2021):
    """
    Use random weighted choice using random

    Parameters
    ----------
    weights : numpy array of floats
        the array of weights

    k : int, optional, default = 1
        the number of draws

    seed : int, optional
        the seed for the random generator, default = 2021

    Returns
    -------
    : numpy array of integer indices
        The rank is equal to k

    Examples
    --------
    >>> import timeit
    >>> import numpy as np
    >>> from geometry_utils import random_weighted_choice_std
    >>> weights = np.array([0.5, 1.2, 0.7, 3., 0.01])
    >>> number = 10000
    >>> sample = random_weighted_choice_std(weights, k=100000)
    >>> stat = np.histogram(sample, bins=[0, 1, 2, 3, 4, 5])[0]
    >>> stat = stat / stat.sum()
    >>> norm_weights = weights / weights.sum()
    >>> perc = (norm_weights - stat) / norm_weights * 100.
    >>> perc
    >>> number = 100000
    >>> sample1 = random_weighted_choice_std(weights, k=number, seed=321)
    >>> sample2 = random_weighted_choice_std(weights, k=number, seed=213)
    >>> stat1 = np.histogram(sample1, bins=[0, 1, 2, 3, 4, 5])[0]
    >>> stat2 = np.histogram(sample2, bins=[0, 1, 2, 3, 4, 5])[0]
    >>> stat1 = stat1 / stat1.sum()
    >>> stat2 = stat2 / stat2.sum()
    >>> norm_weights = weights / weights.sum()
    >>> stat3 = (stat1 - stat2) * norm_weights * 100.
    >>> stat3
    >>> setup='''
    ... import numpy as np
    ... from geometry_utils import random_weighted_choice_std
    ... w = np.array([0.5, 1.2, 0.7, 3., 0.01])
    ... n = 10000
    ... '''
    >>> t = timeit.timeit(stmt='random_weighted_choice_std(w, seed=122, k=n)',
                          setup=setup, number=1)
    >>> # print(t / n)
    >>> # 5.201025000005189e-07
    """
    random.seed(seed)
    return random.choices(population=np.arange(0, len(weights), 1),
                          weights=weights, k=k)


def random_weighted_choice(weights, seed=2021):
    """
    Use numpy random weighted choice

    Parameters
    ----------
    weights : numpy array of floats
        the array of weights

    seed : int, optional
        the seed for the random generator, default = 2021

    Returns
    -------
    order : numpy array of integer indices
        The ordering is base on a random weighted shuffling

    Examples
    --------
    >>> import timeit
    >>> import numpy as np
    >>> from geometry_utils import random_weighted_choice
    >>> weights = np.array([0.5, 1.2, 0.7, 3., 0.01])
    >>> random_weighted_choice(weights)
    array([3, 2, 0, 1, 4])
    >>> random_weighted_choice(weights, seed=122)
    array([3, 1, 2, 0, 4])
    >>> setup='''
    ... import numpy as np
    ... from geometry_utils import random_weighted_choice
    ... w = np.array([0.5, 1.2, 0.7, 3., 0.01])
    ... '''
    >>> n = 10000
    >>> t = timeit.timeit(stmt='random_weighted_choice(w,seed=122)',
                          setup=setup, number=n)
    >>> # print(t / n)
    """
    rng = default_rng(seed)
    p = len(weights)
    return rng.choice(p, p, p=weights / np.sum(weights), replace=False)


def PolyArea(x, y):
    """
    Area of a polygon using the Shoelace formula

    https://en.wikipedia.org/wiki/Shoelace_formula

    https://stackoverflow.com/
    questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Test against
    https://www.omnicalculator.com/math/irregular-polygon-area

    Parameters
    ----------
    x : `numpy.array` or `astropy.quantity.Quantity`
        the array containig the x coordinates of the vertices

    y : `numpy.array` or `astropy.quantity.Quantity`
        the array containig the y coordinates of the vertices

    Returns
    -------
    float of `astropy.quantity.Quantity`
        the area of the polygon

    Examples
    --------
    >>> import numpy as np
    >>> from geometry_utils import PolyArea
    >>> x = np.array([0, 1, 1, 0])
    >>> y = np.array([0, 0, 1, 1])
    >>> PolyArea(x, y)
    1.0
    >>> v = [(0, -2), (6, -2), (9, -0.5), (6, 2),
    ...      (9, 4.5), (4, 7), (-1, 6), (-3, 3)]
    >>> x, y = np.transpose(v)
    >>> PolyArea(x, y)  # the result should be 77.0
    77.0
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def random_polygon(nb, xp, yp, seed=42):
    """
    Given the vertices xp, yp, compute nb random points
    located inside the polygon

    Parameters
    ----------
    nb : int
        the number of requested points

    xp : list or numpy.array or `astropy.quantity`
        the x coordinates of the polygon

    yp : list of numpy.array
        the y coordinates of the polygon

    seed : int, optional
        the seed for the random number generator, default = 42

    Returns
    -------
    xpoints: `numpy.array` or 'astropy.quantity.Quantity'
        the x-coordinates of nb random points inside the polygon

    ypoints: `numpy.array` or 'astropy.quantity.Quantity'
        the y-coordinates of nb random points inside the polygon

    Examples
    --------
    >>> import astropy.units as u
    >>> from geometry_utils import *
    >>> vertices = hexagon_vertices(100, 0)
    >>> xp, yp = np.transpose(vertices)
    >>> nb = 100
    >>> xr, yr = random_polygon(nb, xp, yp)
    >>> inside_hexagon2(100, [xr[10], yr[10]], 0.)
    True
    >>> xp *= u.mm
    >>> yp *= u.mm
    >>> xr, yr = random_polygon(nb, xp, yp)
    >>> xr.unit.to_string()
    'mm'
    """
    if isinstance(xp, astropy.units.quantity.Quantity):
        units = xp.unit
        quantity = True
        xp = xp.value
        yp = yp.value
    else:
        quantity = False
    vertices = np.transpose([xp, yp])
    rnd = np.random.RandomState(seed)
    xpmin = xp.min()
    ypmin = yp.min()
    xpmax = xp.max()
    ypmax = yp.max()
    deltaX = xpmax - xpmin
    deltaY = ypmax - ypmin
    i = 0
    xpoints, ypoints = [], []
    while (i < nb):
        x, y = rnd.rand(2) * [deltaX, deltaY] + [xpmin, ypmin]
        if inside_polygon(x, y, vertices):
            xpoints.append(x)
            ypoints.append(y)
            i += 1
    if quantity:
        return xpoints * units, ypoints * units
    else:
        return np.array(xpoints), np.array(ypoints)


def rectangular_to_spherical(x, y, z):
    """
    Transform cartesian coordiantes x, y, z to
    the spehrical coordiantes theta, phi

    Parameters
    ----------
    x : float, `numpy.array` or `astropy.quantity.Quantity`
        the x coordinate

    y : float, `numpy.array` or `astropy.quantity.Quantity`
        the y coordinate

    z : float, `numpy.array` or `astropy.quantity.Quantity`
        the z coordinate

    Returns
    -------
    theta : `astropy.units.quantity.Quantity` in degrees
        the theta angle in deg

    phi : `astropy.units.quantity.Quantity` in degrees
        the phi angle in deg

    Examples
    --------
    >>> import astropy.units as u
    >>> from geometry_utils import *
    >>> x = 2
    >>> y = 2 * np.sqrt(3)
    >>> z = 4 * np.sqrt(3)
    >>> r, t, p = rectangular_to_spherical(x, y, z)
    >>> np.round(r)
    8.0
    >>> phi.to(u.rad)*6 / (np.pi * u.rad)
    <Quantity 1.>
    >>> theta.to(u.rad)*3 / (np.pi * u.rad)
    <Quantity 1.>

    Notes
    -----
    rectangular (x,y,z) <-> (r,phi,theta)
    This will use tarfoc in the future (2021/01/24)
        x = r sin(phi)cos(theta)
        y = r sin(phi)sin(theta)
        z = r cos(phi)

        r^2 = x^2 + y^2 + z^2
        theta = arctan(y/x)
        phi = arcos(z/r)
    The angles are defined as follows:

    theta is in the x-y plane
    theta is 0 on the x-axis x
    theta is 90 deg on the y-axis

    phi is 0 on the z axis
    phi is 90 degrees on the x-y plane

    if the alt-az ESO coordinates is used
    alt = 90 - phi -> cos(phi) = cos(90 - alt) = sin(alt)
    sin(phi) = cos(alt)

    examples from
    https://math.libretexts.org/Bookshelves/Calculus/
    Book%3A_Calculus_(OpenStax)/
    12%3A_Vectors_in_Space/12.7%3A_Cylindrical_and_Spherical_Coordinates
    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = (np.arctan2(y, x) * u.rad).to(u.deg)
    phi = (np.arccos(z / r) * u.rad).to(u.deg)
    return r, theta, phi


def convert_angle_360_180_north(ang):
    """
    Convert angles from 0 to 360 degrees to
    -180/180 with North=0 degrees

    PA       conversion
    N=0            0
    E=90          90
    S=180      -180/180
    W=270        -90

    Parameter
    ---------
    ang : `astropy.quantity` deg or float or numpy array, list
        the angle(s) to be converted

    Returns
    -------
    : the same format than the input apart from a list when the
      output is a numpy array
        the corresponding angle(s) between -180 and +180 deg

    Examples
    --------
    >>> import astropy.units as u
    >>> from geometry_utils import *
    >>> convert_angle_360_180_north(90.)
     -90.0
    >>> convert_angle_360_180_north([90., 270.])
    array([ 90., -90.])
    """
    if isinstance(ang, list):
        ang = np.array(ang)

    q = type(ang) is astropy.units.quantity.Quantity

    if q:
        ang = ang.to(u.deg).value

    c = ang > 180

    ang_converted = ~c * ang - c * (360. - ang)

    if q:
        ang_converted *= u.deg

    return ang_converted


def angle_in_range(alpha, lower, upper):
    """
    Test if the angle alpha is between lower and upper
    All angles are in degrees between 0 and 360

    Parameters
    ----------
    alpha : `astropy.quantity` deg, range [0...360]
        the test angles

    lower : `astropy.quantity` deg, range [0...360]
        the lower angle

    upper : `astropy.quantity` deg, range [0...360]
        the upper angle

    Returns
    -------
    : bool of array of booleans
    """
    a0 = 360 * u.deg
    return (alpha - lower) % a0 <= (upper - lower) % a0


def convexHull(xpoly, ypoly, xpoints, ypoints):
    """
    Fraction of overlapping area between
    a polygon and the convex hull contaning
    all the points (xpoints, ypoints)

    The routine uses ConvexHull from scipy and
    Polygon from the shapely package

    The optimal value is one if all the objects are
    uniformely distributed inside the polygon.

    The routine does not check if points are outside
    the polygon. If there are points outside the polygon,
    the output is greater than one.

    https://en.wikipedia.org/wiki/Convex_hull

    The routine works in the cartesian metrics

    Parameters
    ----------
    xpoly : array of length m
        the x values of the polygon (same units as
        ypoly, xpoints, and ypoints)

    ypoly : array of length m
        the y values of the polygon

    xpoints : array of length n
        the x positions of the points formning the
        convex hull

    ypoints : array of length n
        the y positions

    Return
    ------
     :  float
        the fracton of overlapping area between the convex hull
        of the focal point objects and the field of view area

    Example
    -------
    >>> import numpy as np
    >>> from geometry_utils import *
    >>> r = 1.0
    >>> PA = 10.0
    >>> vertices = hexagon_vertices(r, PA)
    >>> polygon = np.transpose(vertices)
    >>> pos = random_hexagon(r, PA, n=5000, outside=False, seed=1231)
    >>> frac = convexHull(polygon[0], polygon[1], pos[0], pos[1])
    >>> frac > 0.99
    True
    >>> pos = random_hexagon(1.05, PA, n=5000, outside=False, seed=1652)
    >>> convexHull(polygon[0], polygon[1], pos[0], pos[1])
    1.0978807184148143
    """
    points = np.transpose([xpoints, ypoints])
    hull = ConvexHull(points)
    p = Polygon(np.transpose([xpoly, ypoly]))
    q = Polygon(points[hull.vertices, :])
    point_area = q.intersection(q).area
    polygon_area = p.intersection(p).area
    return point_area / polygon_area


def sky_polygon(ra_inside, dec_inside, ra, dec):
    """
    Return a SphericalPolygon object from
    the RA an Dec values that define a polygon
    on the sky. A position inside the polygon is
    needed to let the code knows the convexity of that
    polygon.

    Parameters
    ----------
    ra_inside : `astropy.quantity` deg, range [0...360]
        RA of a point inside the polygon

    dec_inside : `astropy.quantity` deg, range [-90 ... 90]
        Dec of a point inside the polygon

    ra : `astropy.quantity` deg, range [0...360]
        list of RA that define the polygon on the sky

    dec : `astropy.quantity` deg, range [0...360]
        list of Dec that define the polygon on the sky

    Return
    ------
    : a spherical polygon object

    Example
    -------
    >>> import astropy.units as u
    >>> from geometry_utils import *
    >>> ra = [269.68801204, 268.92337738, 268.97811349, 268.56809053,
    ...       268.5139373 , 267.77579256, 267.84493982, 267.87461843,
    ...       267.91805004, 267.88813334, 267.9901644 , 268.87100668,
    ...       268.89488363, 269.38034859, 269.35725318, 270.3361496 ,
    ...       271.14025336, 271.07984028, 271.49918335, 271.55942634,
    ...       272.29481341, 272.15614326, 272.12667696, 272.05178445,
    ...       272.08105381, 271.92300848, 271.06795968, 271.04699799,
    ...       270.58365698, 270.60449132] * u.deg
    >>> dec = [-56.98076329, -57.26703811, -57.30799119, -57.46595617,
    ...        -57.42469166, -57.71232092, -58.25971182, -58.25837275,
    ...        -58.52903946, -58.53031808, -59.05463457, -59.26997159,
    ...        -59.24285545, -59.35394608, -59.38024039, -59.58523182,
    ...        -59.29024991, -59.25027263, -59.08418589, -59.12411191,
    ...        -58.81474832, -58.27095297, -58.27328127, -58.00454653,
    ...        -58.00230536, -57.48208027, -57.28748279, -57.31498353,
    ...        -57.21008625, -57.18357722] * u.deg
    >>> ra_inside = 270. * u.deg
    >>> dec_inside = -58.2825256 * u.deg
    >>> sky_poly = sky_polygon(ra_inside, dec_inside, ra, dec)
    """
    center = (ra_inside, dec_inside)
    sky_polygon = spherical_polygon.SphericalPolygon.from_radec(ra,
                                                                dec,
                                                                center=center)
    return sky_polygon


def angle_difference(ang1, ang2):
    """
    Angular distance/difference between two angles on the circle

    https://math.stackexchange.com/questions/185831/
    difference-between-degrees-on-a-circle

    Parameters
    ----------
    ang1 : float array of floats or astropy.units.quantity.Quantity
        angle(s) in degrees

    ang2 : float array of floats or astropy.units.quantity.Quantity
        angle(s) in degrees

    Return
    ------
    : astropy.units.quantity.Quantity (degrees)
        the angles in degrees

    Example
    -------
    >>> import astropy.units as u
    >>> from geometry_utils import *
    >>> ang1 = [ 0., 24.] * u.deg
    >>> ang2 = [ 12., 356.] * u.deg
    >>> angle_difference(ang1, ang2)
    <Quantity [12., 28.] deg>
    >>> ang1 = 345.
    >>> ang2 = 0.
    >>> angle_difference(ang1, ang2)
    <Quantity 15. deg>
    """
    if not isinstance(ang1, astropy.units.quantity.Quantity):
        ang1 *= u.deg
    if not isinstance(ang2, astropy.units.quantity.Quantity):
        ang2 *= u.deg

    ang1 = ang1.to(u.deg)
    ang2 = ang2.to(u.deg)
    a1 = ang1.isscalar
    a2 = ang2.isscalar

    err_mssg = "Non compatible array length"
    if a1 or a2:
        if a1 is not a2:
            logging.error(err_mssg)
            return None
    if not a1 and not a2:
        if len(ang1) != len(ang2):
            logging.error(err_mssg)
            return None
    ang360 = 360.
    if a1 and a2:
        a = np.min([ang1.value, ang2.value])
        b = np.max([ang1.value, ang2.value])
        diff_ang = np.min([b - a, (ang360 + a - b) % ang360])
    else:
        a = np.min([ang1, ang2], 0)
        b = np.max([ang1, ang2], 0)
        diff_ang = np.min([b - a, (ang360 + a - b) % ang360], 0)
    return diff_ang * u.deg


def interpolate_angles(x, angles, values):
    """
    Bounded circular interpolation

    https://stackoverflow.com/questions/
    27295494/bounded-circular-interpolation-in-python

    Wrap aroung 360 degrees

    Parameters
    ----------
    x: 1D array of float of length n
        the x independent values for the interpolation

    angles: 1D array of float of of Astropy Quantity angles of lenght n
        the depdendent angles (in degrees)

    values: float or array of floats
        the values for the interpolation

    Returns
    -------
    : astropy.units.quantity.Quantity in degrees of lenght n
        the interpolated angles in degrees

    Example
    -------
    >>> import astropy.units as u
    >>> from geometry_utils import *
    >>> x = np.array([0, 2, 4, 6, 8])
    >>> angles = [1, 179, 211, 359, 1] * u.deg
    >>> val = np.arange(9)  # 0, 1, 2, 3, 4, 5, 6, 7, 8
    >>> interpolate_angles(x, angles, val)
    array([1., 90., 179., 195., 211., 285., 359., 0., 1.])
    """
    if isinstance(angles, astropy.units.quantity.Quantity):
        ang = angles.to(u.deg).value
    else:
        ang = angles  # if float, the values have to be in deg
    complement360 = np.rad2deg(np.unwrap(np.deg2rad(ang)))
    f = interp1d(x, complement360, kind='linear',
                 bounds_error=False, fill_value=None)
    return (f(values) % 360) * u.deg

