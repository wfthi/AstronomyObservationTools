#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
Collects a number of general projection routines

Copyright (C) 2024  Wing-Fai Thi

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
# standard library
import warnings
# third-party
import numpy as np
import astropy.units as u


def skyCoord_gnomonic(center, position):
    """
    Project the the sky coorinates on a planes

    Parameters
    ----------
    center : astropy SkyCoord or skyPosition
        coordiantes of the center
    position : astropy SkyCoord or skyPosition
        positions on the sky to be projected

    Returns
    -------
    x, y : 'astropy.units.quantity.Quantity' degrees
        angles on a plane

    Example
    -------
    >>> from projection import *
    >>> from astropy.coordinates import SkyCoord
    >>> c0 = SkyCoord(ra=3.133 * u.degree, dec=-4 * u.degree)
    >>> c = SkyCoord(ra=3.4 * u.degree, dec=-5 * u.degree)
    >>> skyCoord_gnomonic(c0, c)
    >>> #(<Quantity -0.26602641 deg>, <Quantity -1.00015558 deg>)
    """
    pc = center.ra
    tc = center.dec
    p = position.ra
    t = position.dec
    x, y = sky_gnomonic(pc, tc, p, t)
    return x.to(u.degree), y.to(u.degree)


def deproject_gnomonic(pc, tc, x, y):
    """
    Deproject the position in degree in a place to the sky centrered at
    (pc, tc)

    https://lsstdesc.org/Coord/_build/html/_modules/coord/celestial.html

    The inverse equations are also given at the same web sites:

    sin(dec) = cos(c) sin(dec0) + v sin(c) cos(dec0) / r
    tan(ra-ra0) = u sin(c) / (r cos(dec0) cos(c) - v sin(dec0) sin(c))

    where
        r = sqrt(u^2+v^2)
        c = tan^(-1)(r)     for gnomonic
        c = 2 tan^(-1)(r/2) for stereographic
        c = 2 sin^(-1)(r/2) for lambert
        c = r               for postel

    Note that we can rewrite the formulae as:
    sin(dec) = cos(c) sin(dec0) + v (sin(c)/r) cos(dec0)
    tan(ra-ra0) = u (sin(c)/r) / (cos(dec0) cos(c) - v sin(dec0) (sin(c)/r))

    which means we only need cos(c) and sin(c)/r.  For most of the projections,
    this saves us from having to take sqrt(rsq).

    Parameter
    ---------
    pc : 'astropy.units.quantity.Quantity' degrees
        phi angle of the center

    tc : 'astropy.units.quantity.Quantity' degrees
        theta angle of the center

    x : 'astropy.units.quantity.Quantity' angle
        the projected x position

    y : 'astropy.units.quantity.Quantity' angle
        projected y position

    Notes
    -----
    The distance between the center and the points should not be too large

    Example
    -------
    >>> from projection import deproject_gnomonic
    >>> import astropy.units as u
    >>> pc = 10. / 24.*360. * u.degree
    >>> p  = 10.5 / 24.*360. * u.degree
    >>> tc = 30 * u.degree
    >>> t = 31 * u.degree
    >>> x, y = utils.sky_gnomonic(pc, tc, p, t)
    >>> print(x.to(u.degree), y.to(u.degree))
    >>> utils.deproject_gnomonic(pc, tc, x, y)
    >>> (<Quantity 157.5 deg>, <Quantity 31. deg>)
    >>> x = 0.11261512 * u.radian
    >>> y = 0.02125724 * u.radian
    >>> deproject_gnomonic(pc, tc, x, y)
    >>> from palpy import dtp2s
    >>> pc1 = pc.to(u.rad).value
    >>> tc1 = tc.to(u.rad).value
    >>> dtp2s(x.value, y.value, pc1, tc1)
    """
    x = x.to(u.radian).value
    y = y.to(u.radian).value
    rsq = x * x
    rsq += y * y
    cosdec0 = np.cos(tc).value
    sindec0 = np.sin(tc).value
    cosc = 1. / np.sqrt(1. + rsq)
    # Compute sindec, tandra
    # Note: more efficient to use numpy op= as much
    # as possible to avoid temporary arrays.
    # sindec = cosc * self._sindec + v * sinc_over_r * self._cosdec
    sindec = y * cosc
    sindec *= cosdec0
    sindec += cosc * sindec0

    tandra_num = x
    # tandra_num = x sinc
    # tandra_denom = r * cosc * cosdec - v * sinc * sindec (cosc = sinc / r)
    #              = sinc cosdec * v sinc sindec
    #              = sinc (cosdec + v sindec)
    tandra_denom = -y * sindec0
    tandra_denom += cosdec0
    dec = (np.arcsin(sindec) * u.rad).to(u.deg)
    ra = (pc.to(u.radian).value + np.arctan2(tandra_num, tandra_denom)) * u.rad
    ra = ra.to(u.deg)
    return ra % (360 * u.deg), dec


def palDtp2s(raz, decz, xi, eta):
    """
    Transform tangent plane coordinates into spherical using the
    inverse gnomonic projection.

    Parameters
    ----------
    raz : `astropy.units.quantity.Quantity` degrees or Angle
        RA spherical coordinate of tangent point (radians)

    decz : `astropy.units.quantity.Quantity` degrees or Angle
        Dec spherical coordinate of tangent point (radians)

    xi : `astropy.units.quantity.Quantity` degrees or Angle (rad)
        First rectangular coordinate on tangent plane

    eta : `astropy.units.quantity.Quantity` degrees or Angle (rad)
        Second rectangular coordinate on tangent plane

    Returns
    -------
    ra : `astropy.units.quantity.Quantity` degrees or Angle
        RA spherical coordinate of point to be projected

    dec : `astropy.units.quantity.Quantity` degrees or Angle
        Dec spherical coordinate of point to be projected

    Notes
    -----
    The code has been adapted from the C code palDtp2s fromt the PAL.
    It is only used as comparison code

    Given:
    XI,ETA    d       rectangular coordinates of star image (Note 2)
    A0,B0     d       tangent point's spherical coordinates

    Returned:
    A,B       d       star's spherical coordinates

    All angular arguments are in radians

    The SOFA 2021 Fortran code is:
    SB0 = SIN(B0)
    CB0 = COS(B0)
    D = CB0 - ETA*SB0
    A = iau_ANP(ATAN2(XI,D)+A0)
    B = ATAN2(SB0+ETA*CB0,SQRT(XI*XI+D*D))

    The iau_ANP routine:
    W = MOD(A,D2PI)
    IF ( W .LT. 0D0 ) W = W + D2PI
    iau_ANP = W

    Example
    -------
    >>> import astropy.units as u
    >>> from sts.utils import palDtp2s
    >>> # (0.11261512331584518, 0.02125723540611328)
    >>> x = 0.11261512 * u.radian
    >>> y = 0.02125724 * u.radian
    >>> pc = 10. / 24.*360. * u.degree
    >>> p  = 10.5 / 24.*360. * u.degree
    >>> tc = 30 * u.degree
    >>> t = 31 * u.degree
    >>> ra, dec = palDtp2s(pc, tc, x, y)
    >>> # 157.4999998 deg, 31.00000027 deg
    >>> import palpy as pal
    >>> ra, dec = pal.dtp2s(x.value, y.value,
                            pc.to(u.rad).value, tc.to(u.rad).value)
    >>> ra * u.rad.to(u.deg)
    >>> dec * u.rad.to(u.deg)
    """
    x = xi.to(u.rad).value
    y = eta.to(u.rad).value
    ra0 = raz.to(u.rad).value
    dec0 = decz.to(u.rad).value
    sdec0 = np.sin(dec0)
    cdec0 = np.cos(dec0)
    denom = (cdec0 - y * sdec0)
    d = np.arctan2(x, denom) + ra0
    ra = d % (2 * np.pi)
    dec = np.arctan2(sdec0 + y * cdec0,
                     np.sqrt(x * x + denom * denom))
    ra *= u.rad
    dec *= u.rad
    return ra.to(u.deg) % (360 * u.deg), dec.to(u.deg)


def palDs2tp(raz, decz, ra, dec):
    """
    Projection of spherical coordinates onto tangent plane:
    "gnomonic" projection - "standard coordinates"

    Reference
    ---------
    https://sourcecodequery.com/example-method/palpy.ds2tp
    http://star-www.rl.ac.uk/docs/sun67.htx/sun67.html
    https://github.com/Starlink/pal
    Python wrapper: https://github.com/Starlink/palpy

    This is used by TPoint

    Notes
    -----
    The C version can be found at
    https://github.com/Starlink/pal/blob/master/palDs2tp.c
    A python binding palpy : https://github.com/Starlink/palpy
    The online manual : http://www.starlink.ac.uk/docs/sun267.htx/sun267.html

    Example
    -------
    >>> from sts import utils
    >>> import astropy.units as u
    >>> raz = 10. / 24.*360. * u.degree
    >>> ra  = 10.5 / 24.*360. * u.degree
    >>> decz = 30 * u.degree
    >>> dec = 31 * u.degree
    >>> utils.palDs2tp(raz, decz, ra, dec)
    (<Quantity 0.11261512>, <Quantity 0.02125724>)

    >>> # if palpy is installed
    >>> import palpy as pal
    >>> pal.ds2tp(ra.to(u.rad).value, dec.to(u.rad).value,
                  raz.to(u.rad).value, decz.to(u.rad).value)
    >>> # (0.11261512331584518, 0.02125723540611328)

    >>> dr0 = 3.1
    >>> dd0 = -0.9
    >>> dr1 = dr0 + 0.2
    >>> dd1 = dd0 - 0.1
    >>> dx, dy = pal.ds2tp(dr1, dd1, dr0, dd0)
    >>> print(dx, dy)
    0.10861123015904033 -0.10955062007114535
    >>> palDs2tp(dr0, dd0, dr1, dd1)
    (0.10861123015904033, -0.10955062007114535)
    """
    TINY = 1e-6
    #  Trig functions
    sdecz = np.sin(decz)
    sdec = np.sin(dec)
    cdecz = np.cos(decz)
    cdec = np.cos(dec)
    radif = ra - raz
    sradif = np.sin(radif)
    cradif = np.cos(radif)
    # Reciprocal of star vector length to tangent plane
    denom = sdec * sdecz + cdec * cdecz * cradif

    # Handle vectors too far from axis
    w = (denom >= 0) & (denom <= TINY)
    if np.count_nonzero(w) > 0:
        denom[w] = TINY
        warnings.warn('Source(s) too far from the axis')
    w = (denom < 0) & (denom > -TINY)
    if np.count_nonzero(w) > 0:
        denom[w] = -TINY
        warnings.warn('Source(s) on tangent plane')
    w = (denom <= -TINY)
    if np.count_nonzero(w) > 0:
        warnings.warn('Source(s) too far from the axis')

    xi = cdec * sradif / denom
    eta = (sdec * cdecz - cdec * sdecz * cradif) / denom
    return xi * u.radian, eta * u.radian


def astrd2sn(racen, deccen, ra, dec):
    """
    Standard coordinate conversion (see Kovalevsky)

    Notes
    -----
    Kovalevsky 2002 Book of Modern Astrometry p. 72
    Translated from astrd2sn.pro by Marc Buie

    To check
    --------
    LSST DESC has a minus sign for the xi value

    Example
    -------
    >>> from projection import astrd2sn
    >>> import astropy.units as u
    >>> pc = 10. / 24.*360. * u.degree
    >>> p  = 10.5 / 24.*360. * u.degree
    >>> tc = 30 * u.degree
    >>> t = 31 * u.degree
    >>> astrd2sn(pc, tc, p, t)
    """
    tiny = 1e-6
    beta = ra - racen
    cosbeta = np.cos(beta)
    tandec = np.tan(dec)
    tandeccen = np.tan(deccen)
    d = cosbeta + tandec * tandeccen

    # Handle vectors too far from axis
    w = (d >= 0) & (d <= tiny)
    if np.count_nonzero(w) > 0:
        d[w] = tiny
        warnings.warn('Source(s) too far from the axis')
    w = (d < 0) & (d > -tiny)
    if np.count_nonzero(w) > 0:
        d[w] = -tiny
        warnings.warn('Source(s) on tangent plane')
    w = (d <= -tiny)
    if np.count_nonzero(w) > 0:
        warnings.warn('Source(s) too far from the axis')

    xi = np.sin(beta) / np.cos(deccen) / d
    eta = (tandec - tandeccen * cosbeta) / d
    return xi * u.radian, eta * u.radian


def sky_gnomonic(pc, tc, p, t):
    """
    Local tangent plane projections of an area of the sky using
    the gnomonic projection
        https://mathworld.wolfram.com/GnomonicProjection.html
        https://lsstdesc.org/Coord/_build/html/_modules/coord/celestial.html

    Note: - sign here is to make +x correspond to -ra,
            so x increases for decreasing ra.
            East is to the left on the sky!

    Parameters
    ----------
    pc : 'astropy.units.quantity.Quantity' degrees
        phi angle of the center

    tc : 'astropy.units.quantity.Quantity' degrees
        theta angle of the center

    p : 'astropy.units.quantity.Quantity' degrees
        phi angle of the position on the sky

    t : 'astropy.units.quantity.Quantity' degrees
        theta angle of the position on the sky

    Returns
    -------
    x : 'astropy.units.quantity.Quantity'
        projected x position

    y : 'astropy.units.quantity.Quantity'
        projected y position

    Example
    -------
    >>> from sts.utils import *
    >>> import astropy.units as u
    >>> pc = 10. / 24.*360. * u.degree
    >>> p  = 10.5 / 24.*360. * u.degree
    >>> tc = 30 * u.degree
    >>> t = 31 * u.degree
    >>> sky_gnomonic(pc, tc, p, t)
    (<Quantity 0.11261512 rad>, <Quantity 0.02125724 rad>)
    >>> # -0.11261512 in the LSST code
    >>> palDs2tp(pc, tc, p, t)

    Comparison
    ----------
    >>> from  coord import CelestialCoord
    >>> from coord.angle import Angle
    >>> from coord.angleunit import radians, degrees, hours, arcsec
    >>> center = CelestialCoord(ra=10 * hours, dec=30 * degrees)
    >>> sky_coord = CelestialCoord(ra=10.5 * hours, dec=31 * degrees)
    >>> x, y = center.project(sky_coord)
    >>> print(x / degrees, y / degrees)
    -6.45237127534 1.21794987289
    """
    TINY = 1e-6
    cosdecc = np.cos(tc)
    sindecc = np.sin(tc)
    cosdec = np.cos(t)
    sindec = np.sin(t)
    cosppc = np.cos(p - pc)
    # cosc = np.sin(tc)*np.sin(t) + np.cos(tc)*np.cos(t)*np.cos(p-pc)
    denom = (sindecc * sindec + cosdecc * cosdec * cosppc)

    # Handle vectors too far from axis
    w = (denom >= 0) & (denom <= TINY)
    if np.count_nonzero(w) > 0:
        denom[w] = TINY
        print(denom[w])
        warnings.warn('Source(s) too far from the axis')
    w = (denom < 0) & (denom > -TINY)
    if np.count_nonzero(w) > 0:
        denom[w] = -TINY
        warnings.warn('Source(s) on tangent plane')
    w = (denom <= -TINY)
    if np.count_nonzero(w) > 0:
        warnings.warn('Source(s) too far from the axis')

    # x =  - np.cos(t)*np.sin(p-pc)/cosc
    # minus sign due to the orientation on the sky
    # x = -cosdec * np.sin(p - pc) * inv_cosc
    x = cosdec * np.sin(p - pc)
    x /= denom
    # y = (np.cos(tc)*np.sin(t) - np.sin(tc)*np.cos(t)*np.cos(p-pc))/cosc
    y = (cosdecc * sindec - sindecc * cosdec * cosppc)
    y /= denom
    return x * u.radian, y * u.radian
