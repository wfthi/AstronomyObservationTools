#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import numpy as np
import astropy.units as u
from astropy.coordinates import get_sun
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body, get_body_barycentric

"""
Module containing routines related to the Planet brightness

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

Ref.
    For the brigthness of the planets V-band observations
    Mallama, A. ;  Hilton, J. L.
    Computing apparent planetary magnitudes for The Astronomical Almanac

    Astronomy and Computing, Volume 25, p. 10-24.
    10.1016/j.ascom.2018.08.002

https://arxiv.org/pdf/1808.01973.pdf
https://www.sciencedirect.com/science/article/abs
/pii/S221313371830009X?via%3Dihub

https://sourceforge.net/projects/planetary-magnitudes/

The apparent magnitude depends upon the planet's distance from the Sun, r,
and from the Earth, d, in accordance with the inverse square law.
Another important factor is the illumination phase angle, alpha α,
which is defined as the arc between the Sun and the sensor with
its vertex at the planetocenter.

            V = 5log10(rd) + V1(0)+ C1 α + C2 α^2 + ...

V is the apparent visual magnitude, and V1 (0) is the magnitude when observed
at α = 0 and when the planet is at a distance of one au from both the Sun and
the observer. V1 (0) is sometimes referred to as the planet's absolute
magnitude or geometric magnitude and it may also be thought of as C0 α0.
The sum Σ_n C_n α^n is called the phase function. The phase function generally
increases the planet's apparent magnitude with increasing phase angle.

Simple formulae are taken from
http://stjarnhimlen.se/comp/ppcomp.html#15

    Mercury:   -0.36 + 5*log10(r*R) + 0.027 * FV + 2.2E-13 * FV**6
    Venus:     -4.34 + 5*log10(r*R) + 0.013 * FV + 4.2E-7  * FV**3
    Mars:      -1.51 + 5*log10(r*R) + 0.016 * FV
    Jupiter:   -9.25 + 5*log10(r*R) + 0.014 * FV
    Saturn:    -9.0  + 5*log10(r*R) + 0.044 * FV + ring_magn
    Uranus:    -7.15 + 5*log10(r*R) + 0.001 * FV
    Neptune:   -6.90 + 5*log10(r*R) + 0.001 * FV

    Moon:      +0.23 + 5*log10(r*R) + 0.026 * FV + 4.0E-9 * FV**4

    See also Meeus p. 286

    Meeus j. 1988 Astronomical Algorithms, Atlantic Books, 2nd edition
"""


def illumination_fraction(r, d, dSunEarth):
    """
    Compute the planet illumination fraction

    Ref. Meeus Chap 41 p. 283

    Parameters
    ----------
    r : 'astropy.units.quantity.Quantity' au
        the planet-Sun distance in Astronomical Units

    d: 'astropy.units.quantity.Quantity' au
        the planet-Earth distance in Astronomical Units

    dSunEarth : 'astropy.units.quantity.Quantity' au
        the Sun-Earth distance in Astronomical Units

    Returns
    -------
     : float
        the illumination fraction

    Example
    -------
    Meeus Example 41.a
    Find the illumination fraction k of the disk of Venus on 1992
    December 20, at 0h TD
    >>> import numpy as np
    >>> import astropy.units as u
    >>> import planets
    >>> r = 0.724604 * u.au
    >>> d = 0.910947 * u.au
    >>> dSunEarth = 0.983824 * u.au
    >>> planets.illumination_fraction(r, d, dSunEarth)
    <Quantity 0.64656109>
    """
    cos_phase_angle = (r**2 + d**2 - dSunEarth**2) / (2 * r * d)
    return 0.5 * (1 + cos_phase_angle)


def phase_angle(r, d, dSunEarth):
    """
    Given the planet-Sun distance, the planet-Earth, and
    the Sun-Earth distance, compute the phase-angle of the
    planet as seen from the Earth

    Ref. Meeus Chap 41 p. 283

    Parameters
    ----------
    r : 'astropy.units.quantity.Quantity' au
        the planet-Sun distance in Astronomical Units

    d: 'astropy.units.quantity.Quantity' au
        the planet-Earth distance in Astronomical Units

    dSunEarth : 'astropy.units.quantity.Quantity' au
        the Sun-Earth distance in Astronomical Units

    Returns
    -------
     : 'astropy.units.quantity.Quantity' degree
        the phase-angle

    Example
    -------
    Meeus Example 41.a
    Find the illumination fraction k of the disk of Venus on 1992
    December 20, at 0h TD
    >>> import numpy as np
    >>> import astropy.units as u
    >>> import planets
    >>> r = 0.724604 * u.au
    >>> d = 0.910947 * u.au
    >>> dSunEarth = 0.983824 * u.au
    >>> k = 0.5 * (1 + np.cos(planets.phase_angle(r, d, dSunEarth)))
    >>> k
    <Quantity 0.64656109>
    >>> # <Quantity 0.64656109>
    """
    r = r.value
    d = d.value
    dSunEarth = dSunEarth.value
    x = (r**2 + d**2 - dSunEarth**2) / (2 * r * d)
    return (np.arccos(x) * u.rad).to(u.deg)


def planet_earth(planet_name, telescope, time):
    """
    Compute the planet-Sun distance, the planet-Earth distance,
    the Sun-Earth distance, the Planet-Sun-Earth phase angle.

    Wrapper to sereval astropy routines

    Parameters
    ----------
    planet_name : str
        official name of the planet (Jupiter, Mars, Saturn, Venus)

    telescope: `astropy.coordinates.earth.EarthLocation`
        The location on the Earth.

    time : 'astropy.time.core.Time'
        the time of the observation

    Returns
    -------
    rr : 'astropy.units.quantity.Quantity' au
        the distance between the planet and the Sun

    delta : 'astropy.units.quantity.Quantity' au
        the distance between the planet and the Earth

    dSunEarth : 'astropy.units.quantity.Quantity' au
        the distance of the Sun to the Earth

    ph_ang : 'astropy.units.quantity.Quantity' angle, deg
        the phase angle

    planet_geo : astropy geocentric coordinates SkyCoord object
        SkyCoord object in the GCRS frame

    planet_bary : astropy barycentric coordinates object
        CartesianRepresentation of the barycentric position of the
        body (i.e., in the ICRS frame)

    Example
    -------
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> from astropy.coordinates import EarthLocation
    >>> import planets
    >>> t = Time("2014-09-22 23:22")
    >>> t = Time("2022-01-01 0:0")
    >>> vista_location = EarthLocation(lat=-24.615833 * u.deg,
    ...                                lon=-70.3975 * u.deg,
    ...                                height=2518 * u.m)
    >>> rr, delta, dSunEarth, ph_ang, planet_geo, planet_bary =\
        planets.planet_earth('jupiter', vista_location, t)
    >>> rr, delta, dSunEarth, ph_ang, planet_geo, planet_bary =\
        planets.planet_earth('mars', vista_location, t)
    >>> rr, delta, dSunEarth, ph_ang, planet_geo, planet_bary =\
        planets.planet_earth('saturn', vista_location, t)
    >>> # 314.29291198 deg => 314° 17' 34.48313"
    >>> # -18.09602172 deg => -18° 5' 45.67819"
    >>> # Longitude= 311° 35' 40" 20h 46m 23s
    >>> # Latitude = -0° 49' 19"
    """
    with solar_system_ephemeris.set('builtin'):
        planet_geo = get_body(planet_name, time, telescope)
        planet_bary = get_body_barycentric(planet_name, time)
    dplanet = np.sqrt((planet_bary.x)**2 +
                      (planet_bary.y)**2 +
                      (planet_bary.z)**2)
    dplanet = dplanet.to(u.au)  # Sun-planet distance at time
    delta = planet_geo.distance  # Earth-planet distance
    rr = dplanet
    sun = get_sun(time)
    dSunEarth = sun.distance
    ph_ang = phase_angle(rr, delta, dSunEarth)
    return rr, delta, dSunEarth, ph_ang, planet_geo, planet_bary


def jupiter_magnitude(time, telescope):
    """
    Ref. Mallama & Hilton

    Parameters
    ----------
    time : `~astropy.time.Time` one of multiple entries
            Time of observation

    telescope: `astropy.coordinates.earth.EarthLocation`
        The location on the Earth.

    Return
    ------
    ap_mag : 'astropy.units.quantity.Quantity' mag
        apparent magnitude

    N.B. use skyfield https://rhodesmill.org/skyfield/
    https://pypi.org/project/skyfield/

    Example
    -------
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import planets
    >>> from astropy.coordinates import EarthLocation
    >>> t = Time("2014-09-22 23:22")
    >>> vista_location = EarthLocation(lat=-24.615833 * u.deg,
    ...                                lon=-70.3975 * u.deg,
    ...                                height=2518 * u.m)
    >>> planet_name = 'jupiter'
    >>> ap_mag, ap_mag_simple =\
        planets.jupiter_magnitude(t, vista_location)
    >>> ap_mag
    <Quantity -1.87540708 mag>
    >>> ap_mag_simple
    <Quantity -1.65606785 mag>
    >>> # comparison with skyfield
    >>> from skyfield.api import load
    >>> from skyfield.magnitudelib import planetary_magnitude
    >>> ts = load.timescale()
    >>> eph = load('de421.bsp')
    >>> t1 = ts.utc(2014, 9, 22, 23, 22, 0.)
    >>> astrometric = eph['earth'].at(t1).observe(eph['jupiter barycenter'])
    >>> print('%.2f' % planetary_magnitude(astrometric))
    -1.88
    >>> # -1.88
    """
    # Calculate the apparent magnitude
    rr, delta, dSunEarth, ph_ang, _, _ =\
        planet_earth('jupiter', telescope, time)

    rr = rr.value
    delta = delta.value
    dSunEarth = dSunEarth.value
    ph_ang = ph_ang.value

    # Compute the 'r' distance factor in magnitudes
    r_mag_factor = 2.5 * np.log10(rr * rr)

    # Compute the 'delta' distance factor in magnitudes
    distance_mag_factor = 2.5 * np.log10(delta * delta)

    # Compute the distance factor
    distance_mag_factor = r_mag_factor + distance_mag_factor

    # Compute the phase angle factor
    geocentric_phase_angle_limit = 12.0  # deg
    if (ph_ang <= geocentric_phase_angle_limit):
        # Use equation #8 for phase angles below the geocentric limit
        ph_ang_factor = -3.7E-04 * ph_ang + 6.16E-04 * ph_ang**2
    else:
        # Use equation #9 for phase angles above the geocentric limit
        # ph_ang_factor: phase angle factor in magnitudes
        ph_ang_factor = -2.5 * np.log10(1.0 - 1.507 * (ph_ang / 180.) -
                                        0.363 *
                                        (ph_ang / 180.)**2 - 0.062 *
                                        (ph_ang / 180.)**3 +
                                        2.809 *
                                        (ph_ang / 180.)**4 - 1.876 *
                                        (ph_ang / 180.)**5)

    # Add factors to determine the apparent magnitude
    if (ph_ang <= geocentric_phase_angle_limit):
        # Use equation #6 for phase angle <= 50 degrees
        ap_mag = -9.395 + distance_mag_factor + ph_ang_factor
    else:
        # Use equation #7 for phase angle > 50 degrees
        ap_mag = -9.428 + distance_mag_factor + ph_ang_factor

    # simple formula
    # alternative simple formula, see Meeus p. 285
    ap_mag_simple = -9.25 + 5. * np.log10(rr * delta) +\
        0.014 * ph_ang

    return ap_mag * u.mag, ap_mag_simple * u.mag


def saturn_magnitude(time, telescope):
    """
    Compute an approximate magnitude for Saturn.

    Ref. Mallama & Hilton
         Meeus Chap 45. ring of Saturn p. 317

    The apparent magnitude of saturn is between

    Vmin = -0.55 and Vmax = 1.17

    https://www.wikiwand.com/en/Apparent_magnitude#/Standard_reference_values

    The second output seems to agree better with skyfield and other
    results

    Parameters
    ----------
    time : `~astropy.time.Time` one of multiple entries
            Time of observation

    telescope: `astropy.coordinates.earth.EarthLocation`
        The location on the Earth.

    Example
    -------
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import planets
    >>> from astropy.coordinates import EarthLocation
    >>> t = Time("2014-09-22 23:22")
    >>> t = Time("1996-12-16 0:0")
    >>> t = Time("2022-01-01 0:0")
    >>> vista_location = EarthLocation(lat=-24.615833 * u.deg,
    ...                                lon=-70.3975 * u.deg,
    ...                                height=2518 * u.m)
    >>> planet_name = 'saturn'
    >>> ap_mag, ap_mag_simple =\
        planets.saturn_magnitude(t, vista_location)
    >>> ap_mag, ap_mag_simple
    (<Quantity 0.51671416 mag>, <Quantity 0.70964584 mag>)
    >>> #
    >>> # comparison with skyfield
    >>> from skyfield.api import load
    >>> from skyfield.magnitudelib import planetary_magnitude
    >>> ts = load.timescale()
    >>> eph = load('de421.bsp')
    >>> t1 = ts.utc(2014, 9, 22, 23, 22, 0.)
    >>> t1 = ts.utc(1996, 12, 16, 0, 0, 0.)
    >>> t1 = ts.utc(2022, 1, 1, 0, 0, 0.)
    >>> astrometric = eph['earth'].at(t1).observe(eph['saturn barycenter'])
    >>> print('%.2f' % planetary_magnitude(astrometric))
    0.77
    >>> a = [-1.505e-6, 2.672e-4, 2.446e-4]
    >>> ph_ang = 6.
    >>> Globe = 4.767e-9 * ph_ang
    >>> for a0 in a:
    ...     Globe = (Globe + a0) * ph_ang
    """

    rr, delta, dSunEarth, ph_ang, saturn_geo, _ =\
        planet_earth('saturn', telescope, time)

    rr = rr.value
    delta = delta.value
    dSunEarth = dSunEarth.value
    ph_ang = ph_ang.value

    # http://stjarnhimlen.se/comp/ppcomp.html#15
    # Saturn's geocentric ecliptic longitude and latitude
    lat = saturn_geo.geocentrictrueecliptic.lat
    lon = saturn_geo.geocentrictrueecliptic.lon

    # d is the "day number"
    ir = 28.06 * u.deg  # deg tilt of the rings to the ecliptic
    # NR "ascending node" of the plane of the rings
    d = time.ut1.jd - 2451543.5
    Nr = (169.51 + 3.82E-5 * d) * u.deg

    # B is the tilt of Saturn's rings
    B = np.arcsin(np.sin(lat) * np.cos(ir) -
                  np.cos(lat) * np.sin(ir) * np.sin(lon - Nr))

    # Meeus p. 285
    ring_magn = -2.6 * np.sin(abs(B)) + 1.25 * (np.sin(B))**2

    # Magnitude of the planet without the ring
    if 6.5 <= ph_ang < 150:  # eq. 12
        a = [-1.505e-6, 2.672e-4, 2.446e-4]
        Globe = 4.767e-9 * ph_ang
        for a0 in a:
            Globe = (Globe + a0) * ph_ang
        Globe += 5. * np.log10(rr * delta) - 8.94
    elif ph_ang < 6.5:  # eq. 11
        Globe = 5. * np.log10(rr * delta) - 8.95 +\
            (6.16e-4 * ph_ang - 3.7e-4) * ph_ang
    ap_mag = Globe + ring_magn

    # the phase angle is used as an approximation
    # Meeus p. 286
    Globe_simple = -8.88 + 5. * np.log10(rr * delta) + 0.044 * ph_ang
    ap_mag_simple = Globe_simple + ring_magn

    return ap_mag * u.mag, ap_mag_simple * u.mag


def mars_magnitude(time, telescope):
    """
    Compute the apparent magnitude of mars

    see skyfield-1.43.1 magnitudelib.py

    Parameters
    ----------
    time : `~astropy.time.Time` one of multiple entries
            Time of observation

    telescope: `astropy.coordinates.earth.EarthLocation`
        The location on the Earth.

    Return
    ------
    ap_mag : 'astropy.units.quantity.Quantity' mag
        apparent magnitude

    Example
    -------
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import planets
    >>> from astropy.coordinates import EarthLocation
    >>> t = Time("2014-09-22 23:22")
    >>> vista_location = EarthLocation(lat=-24.615833 * u.deg,
    ...                                lon=-70.3975 * u.deg,
    ...                                height=2518 * u.m)
    >>> ap_mag, ap_mag_simple =\
        planets.mars_magnitude(t, vista_location)
    >>> ap_mag
    <Quantity 0.74088777 mag>
    >>> ap_mag_simple
    <Quantity 0.77389899 mag>
    >>> # comparison with skyfield
    >>> from skyfield.api import load
    >>> from skyfield.magnitudelib import planetary_magnitude
    >>> ts = load.timescale()
    >>> eph = load('de421.bsp')
    >>> t1 = ts.utc(2014, 9, 22, 23, 22, 0.)
    >>> astrometric = eph['earth'].at(t1).observe(eph['mars barycenter'])
    >>> print('%.2f' % planetary_magnitude(astrometric))
    0.74
    >>> # 0.74

    """
    # Calculate the apparent magnitude
    rr, delta, dSunEarth, ph_ang, _, _ =\
        planet_earth('mars', telescope, time)

    rr = rr.value
    delta = delta.value
    dSunEarth = dSunEarth.value
    ph_ang = ph_ang.value

    r_mag_factor = 2.5 * np.log10(rr * rr)
    delta_mag_factor = 2.5 * np.log10(delta * delta)
    distance_mag_factor = r_mag_factor + delta_mag_factor

    geocentric_phase_angle_limit = 50.0

    condition = ph_ang <= geocentric_phase_angle_limit
    a = np.where(condition, 2.267E-02, - 0.02573)
    b = np.where(condition, - 1.302E-04, 0.0003445)
    ph_ang_factor = a * ph_ang + b * ph_ang**2

    # Until effects from Mars rotation are written up:
    mag_corr_rot = 0.0
    mag_corr_orb = 0.0

    # Add factors to determine the apparent magnitude
    ap_mag = np.where(ph_ang <= geocentric_phase_angle_limit, -1.601, -0.367)
    ap_mag += distance_mag_factor + ph_ang_factor + mag_corr_rot + mag_corr_orb

    # See alternative formula in Meeus p. 285
    ap_mag_simple = -1.51 + 5. * np.log10(rr * delta) + 0.016 * ph_ang
    return ap_mag * u.mag, ap_mag_simple * u.mag


def venus_magnitude(time, telescope, screen=False):
    """
    Ref. Mallama & Hilton and skyfield-1.43.1

    Parameters
    ----------
    time : `~astropy.time.Time` one of multiple entries
            Time of observation

    telescope: `astropy.coordinates.earth.EarthLocation`
        The location on the Earth.

    screen : bool, optional, default=False
        set to True to have a screen output

    Return
    ------
    ap_mag : 'astropy.units.quantity.Quantity' mag
        apparent magnitude

    N.B. use skyfield https://rhodesmill.org/skyfield/
    https://pypi.org/project/skyfield/

    Example
    -------
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import planets
    >>> from astropy.coordinates import EarthLocation
    >>> t = Time("2014-09-22 23:22")
    >>> vista_location = EarthLocation(lat=-24.615833 * u.deg,
    ...                                lon=-70.3975 * u.deg,
    ...                                height=2518 * u.m)
    >>> ap_mag, ap_mag_simple = planets.venus_magnitude(t, vista_location)
    >>> ap_mag
    <Quantity -3.92977363 mag>
    >>> ap_mag_simple
    <Quantity -3.76879454 mag>
    >>> # comparison with skyfield
    >>> from skyfield.api import load
    >>> from skyfield.magnitudelib import planetary_magnitude
    >>> ts = load.timescale()
    >>> eph = load('de421.bsp')
    >>> t1 = ts.utc(2014, 9, 22, 23, 22, 0.)
    >>> astrometric = eph['earth'].at(t1).observe(eph['venus barycenter'])
    >>> print('%.2f' % planetary_magnitude(astrometric))
    -3.93
    >>> # -3.93
    """
    # Calculate the apparent magnitude
    rr, delta, dSunEarth, ph_ang, _, _ =\
        planet_earth('venus', telescope, time)

    if screen:
        print(rr, delta, dSunEarth, ph_ang)
    rr = rr.value
    delta = delta.value
    dSunEarth = dSunEarth.value
    ph_ang = ph_ang.value

    distance_mag_factor = 5. * np.log10(rr * delta)
    if ph_ang <= 163.7:
        a0 = -4.384
        a1 = -1.044E-03
        a2 = 3.687E-04
        a3 = -2.814E-06
        a4 = 8.938E-09
    else:
        a0 = 236.05828
        a1 = - 2.81914
        a2 = 8.39034E-03
        a3 = 0.0
        a4 = 0.0
    ph_ang_factor = a4
    for a in [a3, a2, a1, a0]:
        ph_ang_factor *= ph_ang
        ph_ang_factor += a
    ap_mag = distance_mag_factor + ph_ang_factor
    # See alternative formula in Meeus p. 285
    ap_mag_simple = -4.34 + 5. * np.log10(rr * delta) + 0.013 *\
        ph_ang + 4.2E-7 * ph_ang**3

    return ap_mag * u.mag, ap_mag_simple * u.mag


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
