# Highly inspired by the work of:
# Source: https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.1, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n, min_width, max_width, min_height, max_height, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(a,axis=0)**2, axis=1))
    if np.any(d<mindst):
        if rec > 10:
            raise ValueError("could not generate points")
        return get_random_points(n, min_width, max_width, min_height, max_height, mindst=mindst, rec=rec+1)
    print(a)
    a = a * np.array([max_width - min_width, max_height - min_height]) + np.array([min_width, min_height])
    print(a)
    return a


def get_random_outline(n, center, min_width, max_width, min_height, max_height, edgy=0):
    """ create a random outline of *n* points, with a random
    bezier curve through them. """
    a = get_random_points(n=n, min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height)
    a = a - np.mean(a, axis=0)
    x,y, a = get_bezier_curve(a, edgy=edgy)

    # Makes sure the dimensions are correct
    x /= np.max(abs(x))
    y /= np.max(abs(y))
    x = x * (max_width - min_width) + min_width
    y = y * (max_height - min_height) + min_height

    # Makes sure the center is correct
    x += center[0] - np.mean(x)
    y += center[1] - np.mean(y)

    return x,y

def get_random_shapely_outline(n, center, min_width, max_width, min_height, max_height, edgy=0):
    """ create a random outline of *n* points, with a random
    bezier curve through them."""
    x,y = get_random_outline(n, center, min_width, max_width, min_height, max_height, edgy)
    return Polygon(list(zip(x,y)))


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    rad = 0.2
    edgy = 0.5

    x,y = get_random_outline(12, [40,40], 50, 80, 30, 50, edgy)
    plt.plot(x,y, "k")
    plt.show()