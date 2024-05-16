#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
Module containing routines related to the Moon and the Moon-target

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

The module requires the numpy, scipy, matplotlib, and astropy packages

The output of the module assume a clear sky.
"""
# standard library
import logging
# third-party
from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u
from astropy.constants import au
from astropy.coordinates import get_body
from astropy.coordinates import get_sun
from astropy.coordinates import SkyCoord
from astropy.coordinates import AltAz
import matplotlib as mpl
import matplotlib.pyplot as plt


__all__ = ["sunmoon_separation", "moon_pointing_separation",
           "moon_pointing_separation_time", "moon_phase_angle",
           "moon_magnitude", "moon_illumination", "moon_illuminance",
           "darktime_sky_brightness", "atmospheric_scattering_function",
           "airmassRozenberg", "airmassKS", "moon_brightness",
           "surface_brightness_to_nanoLamberts",
           "nanoLamberts_to_V_surface_brightness", "magnitude_correction",
           "krisciunas_schaefer_tests", "krisciunas_schaefer", "moon_phase",
           "plot_sky_brightness", "Paranal_sky_brightness",
           "relative_sky_brightness", "Kastner_logL", "Kastner_logL2",
           "twilight_lsst", "moon_diameter"]


def add_magnitudes(magnitude1, magnitude2):
    """
    Sum two magnitudes

    mtot = -2.5 log10(10^(-0.4 m1) + 10^(-0.4 m2))
         = m2 - 2.5 log10(10**(0.4 * (m2 - m1)) + 1)

    Notes
    -----
    When one has 2 magnitude objects from astropy.units, their summation is
    wrong.  Meeus p. 393

    Meeus j. 1988 Astronomical Algorithms, Atlantic Books, 2nd edition


    Parameters
    ----------
    magnitude1 :  float, numpy.ndarray or `astropy.units.quantity.Quantity` mag
        first magnitude

    magnitude2 :  float, numpy.ndarray or `astropy.units.quantity.Quantity` mag
        second magnitude

    Returns
    -------
    : `astropy.units.quantity.Quantity` mag
        total magnitude

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from opsys_utils.utils import *
    >>> add_magnitudes(1, 1)
    0.24742501084004698
    >>> add_magnitudes(20, 22)
    19.84026991464155
    >>> add_magnitudes([20, 21], [22., 22])
    array([19.84026991, 20.63614884])
    >>> add_magnitudes(20 * u.mag, 22 * u.mag)
    <Quantity 19.84026991 mag>
    >>> add_magnitudes([20, 20] * u.mag, [22., 22] * u.mag)
    <Quantity [19.84026991, 19.84026991] mag>
    >>> add_magnitudes(20, [21, 22])
    array([19.63614884, 19.84026991])
    >>> add_magnitudes([20, 20], [21, 22])
    array([19.63614884, 19.84026991])
    >>> add_magnitudes([21, 22], 20)
    array([19.63614884, 19.84026991])
    >>> add_magnitudes([21, 22], np.inf)
    array([21., 22.])
    >>> add_magnitudes([21, 22], [19., np.inf])
    array([18.84026991, 22.        ])
    """
    mag_big = 200.
    mag1 = np.array(magnitude1)
    mag2 = np.array(magnitude2)
    winf1 = mag1 == np.inf
    winf2 = mag2 == np.inf
    mag1[winf1] = mag_big
    mag2[winf2] = mag_big
    size1 = mag1.size
    size2 = mag2.size
    if size1 == 1 and size2 > 1:
        mag1 = np.repeat(mag1, size2)
    if size2 == 1 and size1 > 1:
        mag2 = np.repeat(mag2, size1)
    magsum = mag2 - 2.5 * np.log10(10**(0.4 * (mag2 - mag1)) + 1.)
    magt1 = isinstance(magnitude1, u.quantity.Quantity)
    magt2 = isinstance(magnitude2, u.quantity.Quantity)
    if magt1 or magt2:
        magsum *= u.mag
    return magsum


def sunmoon_separation(FLI):
    """
    Compute the Sun-Moon separation given the FLI
    The fraction of lunar illumination

    Parameter
    ---------
    FLI : float or array of floats betwween 0 and 1
        the moon illumination fraction

    Return
    ------
    : 'astropy.units.quantity.Quantity' deg
        the Sun-Moon separation

    Examples
    --------
    >>> from moon import sunmoon_separation
    >>> sunmoon_separation(1.)
    <Quantity 180. deg>
    >>> sunmoon_separation(0.5)
    <Quantity 90. deg>
    """
    if isinstance(FLI, float):
        assert FLI >= 0.
        assert FLI <= 1.
    else:
        assert all(FLI >= 0.)
        assert all(FLI <= 1.)
    return (np.arccos(1. - 2 * FLI) * u.rad).to(u.deg)


def moon_pointing_separation(time, pointing, ephemeris=None):
    """
    Broadcast the moon_pointing_separation routine
    for a range of times

    Parameters
    ----------
    pointing : `object`, any object with attribute ra and dec
                or objects whose class inheritates the skyPosition class
        Coordinates on the sky (It can one or multiple positions)

    ephemeris : str, optional
            Ephemeris to use.  If not given, use the one set with
            `~astropy.coordinates.solar_system_ephemeris` (which is
            set to 'builtin' by default).

    Returns
    -------
    Moon-pointing angular separation [degree] :
        'astropy.units.quantity.Quantity' or a arrays of arrays
        (ie a Time x position matrix)
        if the input is more than one pointing

    Notes
    -----
    The routine moon_pointing_separation_time handles the computation
    for one time value and possibly multiple sky positins.
    This routine broadcasts that routine to multiple time values.

    Examples
    --------
    >>> from moon import moon_pointing_separation
    >>> from moon import moon_pointing_separation_time
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> c = SkyCoord(ra=133 * u.degree, dec=-45 * u.degree)
    >>> time_utc = '2022-01-01T00:00:00'
    >>> ObservingTime = Time(time_utc, format='isot', scale='utc')
    >>> moon_pointing_separation(ObservingTime, c)
    <Angle 92.53663009 deg>
    >>> c = SkyCoord(ra=[133, 15] * u.degree, dec=[-45, 5] * u.degree)
    >>> time_utc = ['2022-01-01T00:00:00', '2022-01-02T00:00:00Z']
    >>> ObservingTime = Time(time_utc, format='isot', scale='utc')
    >>> moon_pointing_separation(ObservingTime, c)
    <Quantity [[ 92.53663009,  99.06745525],
               [120.51663277, 105.35794381]] deg>
    """
    if isinstance(time.value, np.ndarray):
        sep = []
        for t in time:
            sep.append(moon_pointing_separation_time(t,
                                                     pointing,
                                                     ephemeris=ephemeris)
                       .value)
        return np.transpose(sep) * u.deg
    else:
        sep = moon_pointing_separation_time(time, pointing,
                                            ephemeris=ephemeris)
        return sep


def moon_pointing_separation_time(time, pointing, ephemeris=None):
    """
    Calculate the angular separation between a pointing with SkyCoord
    and the Moon at time time [0,180] degrees

    Parameters
    ----------
    time : `~astropy.time.Time` with one entry
            Time of observation

    pointing : `object` object, any object with attributes ra and dec
        Coordinates on the sky or list of

    ephemeris : str, optional
        Ephemeris to use.  If not given, use the one set with
        `~astropy.coordinates.solar_system_ephemeris` (which is
        set to 'builtin' by default).

    Returns
    -------
    Moon-pointing angular separation [degree] :
        'astropy.units.quantity.Quantity' or a list of values
        if the input is more than one pointing

    Example
    -------
    >>> from moon import moon_pointing_separation
    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> c = SkyCoord(133, -45., unit="deg")
    >>> time_utc = '2022-01-01T00:00:00Z'
    >>> ObservingTime = Time(time_utc, format='isot', scale='utc')
    >>> Equinox = 'J2000'
    >>> moon_pointing_separation(ObservingTime, c)
    <Angle 92.53663009 deg>
    """
    moon = get_body("moon", time, ephemeris=ephemeris)
    coord = SkyCoord(ra=pointing.ra, dec=pointing.dec, obstime=time)
    return moon.separation(coord)


def moon_phase_angle(sun_distance, moon_distance, sun_moon_elongation):
    """
    Compute the moon phase angle :
    the selenocentric elongation of the Earth from the Sun

    This is an alternative to astroplan moon_phase_angle, which requires
    only time as input

    For Earth-based observations of the Moon this is typically taken as
    the angle between the center of the Sun and the center of the Earth as
    seen from the center of the Moon.

    By convention, the lunar phase angle is given a sign. It starts at
    roughly -180° near New Moon and proceeds to near 0° at Full Moon
    (see phases).
    It then goes from near 0° to near +180° during the remainder of the cycle.
    The First Quarter (Moon half illuminated on the east) occurs near a phase
    angle of -90°, and the Last Quarter (Moon half illuminated on the west)
    near +90°.

    alpha = np.arctan2(sun_distance * np.sin(sun_moon_elongation),
                       moon_distance - sun_distance *
                       np.cos(sun_moon_elongation))

    See Meeus Chap. 48 eq. 48.3
    Meeus j. 1988 Astronomical Algorithms, Atlantic Books, 2nd edition

    Low accuracy cos(alpha) = -cos(sun_moon_elongation)

    Parameters
    ----------
    sun_distance : `astropy.coordinates.distances.Distance` AU
        distance to the sun obtained using astropy
        sun = get_sun(time)
        sun_distance = sun.distance

    moon_distance : `astropy.coordinates.distances.Distance` AU
        distance to the moon obtained using astropy
        moon = get_body("moon", time, ephemeris=ephemeris)
        moon_distance = moon.distance

    sun_moon_elongation :`astropy.coordinates.angles.Angle` deg
        the sun-moon elongation. Warning: this is not equal to
        moon-sun elongation. The elongation can be computed using
        astropy
        elongation = sun.separation(moon)

    Returns
    -------
    alpha : `astropy.units.quantity.Quantity` deg
        the moon phase-anle

    Examples
    --------
    >>> from astropy.time import Time
    >>> from astropy.coordinates import get_body, get_sun
    >>> from moon import moon_phase_angle
    >>> time = Time(2021.5, format='decimalyear')
    >>> sun = get_sun(time)
    >>> moon = get_body("moon", time)
    >>> sun.separation(moon)
    <Angle 83.08036354 deg>
    >>> alpha = moon_phase_angle(sun.distance, moon.distance,
    ...                          sun.separation(moon))
    >>> alpha
    <Quantity 96.77006516 deg>
    """
    alpha = np.arctan2(sun_distance * np.sin(sun_moon_elongation),
                       moon_distance - sun_distance *
                       np.cos(sun_moon_elongation))
    return alpha.to(u.deg)


def moon_magnitude(phase_angle):
    """
    Compute the moon magnitude from the phase angle

    Parameter
    ---------
    phase_angle : 'astropy.units.quantity.Quantity' deg
        the moon phase angle

    Allen 1976 p. 144 and KS91 (eq. 9, 0. 1035)

    Used indirectly by the krisciunas_schaefer routine
    moon_magnitude -> moon_illuminance -> moon_brightness
    -> krisciunas_schaefer

    Allen
    https://astronomy.stackexchange.com/questions/10246/
    is-there-a-simple-analytical-formula-for-the-lunar-phase-brightness-curve
    m = -12.73 + 1.49 x |phi| + 0.043 x phi^4
    phi is the moon phase

    Notes
    -----
    The sign of the phase_angle does not matter to compute
    the moon magnitude

    From KS91

    "The brightness of scattered moonlight will be proportional to I.
    Equation (9) gives m = —12.73 for full Moon, in agreement with the
    results of Lane & Irvine 1973 extrapolated to alpha= 0.
    However, this ignores the "opposition effect". For
    lal < 7° the bright- ness of the Moon deviates from this relation
    (see, e.g., Whitaker 1969).
    When the Moon is exactly full it is about 35% brighter than
    the extrapolation would predict, assuming, of course, that it is
    not undergoing a penumbral or umbral eclipse."

    See Kiefer and Stone for a more accurate model
    The Astronomical Journal, 129:2887-2901, 2005 June

    Noll et al. 2012
    Astronomy & Astrophysics, Volume 543, id.A92, 23 pp.

    Winkler, Hartmut 2022
    MNRAS, Volume 514, Issue 1, pp.208-226

    https://link.springer.com/chapter/10.1007/978-1-4419-8816-4_4

    Returns
    -------
    : 'astropy.units.quantity.Quantity' mag
        the moon magnitude

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import moon_magnitude
    >>> phase_angle = 30 * u.deg
    >>> moon_magnitude(phase_angle)
    <Quantity -11.94676 mag>
    >>> phase_angle = 0 * u.deg
    >>> moon_magnitude(phase_angle)
    <Quantity -12.73 mag>
    """
    alpha = phase_angle.to(u.deg).value
    mag_fullmoon = -12.73  # Lane & Irvine 1973
    moon_mag = (mag_fullmoon + 0.026 * np.abs(alpha) +
                4e-9 * alpha**4) * u.mag
    return moon_mag


def moon_phase_angle_illumination(moon_illumination):
    """
    Compute the moon phase angle from its illumination

    Meeus Chap. 48
    Meeus j. 1988 Astronomical Algorithms, Atlantic Books, 2nd edition

    Example
    -------
    >>> from moon import moon_phase_angle_illumination
    >>> moon_phase_angle_illumination(0.9)
    <Quantity 36.86989765 deg>
    """
    phase_angle = np.arccos(2 * moon_illumination - 1.)
    return (phase_angle * u.rad).to(u.deg)


def moon_illumination(time):
    """
    Compute the Moon illumation (aka FLI Fractional Lunar Illumination)
    given the time of the observation

    The routine requires only the astropy package

    Parameters
    ----------
    time : `astropy.time.core.Time' object
        the time

    Returns
    -------
    FLI : float
        the moon fractional illumination between 0 to 1
        The output = 0 indicates a new moon, while = 1 indicates a full moon.

    Example
    -------
    >>> import numpy as np
    >>> from astropy.time import Time
    >>> from moon import moon_illumination
    >>> time = Time(2024.495, format='decimalyear')
    >>> moon_illumination(time)
    0.35912003805929105
    """
    sun = get_sun(time)
    moon = get_body("moon", time)
    alpha = moon_phase_angle(sun.distance, moon.distance,
                             sun.separation(moon))
    return moon_illumination_angle(alpha)


def moon_illumination_angle(phase_angle):
    """
    Compute the Moon fraction illumination

    Meeus Chap 41 & 48, eq. 41.1, 48.1 p.345
    Meeus j. 1988 Astronomical Algorithms, Atlantic Books, 2nd edition

    The Moon is roughly 0% illuminated when New,
    50% illuminated at the Quarters, and
    100% illuminated when Full;
    but even for a geocentric observer these values
    are not exact, and they vary from month to month.

    In Meeus, the geocentric elongation = Moon-Sun separation

    An approximation is cos(phase_angle) = -cos(Moon-Sun separation)
    See Meeus Chap. 48
    Meeus j. 1988 Astronomical Algorithms, Atlantic Books, 2nd edition

    Notes
    -----
    The sign of the phase_angle does not matter.
    |phase_angle| = pi x moon_phase (rad) or 180 x moon_phase (degrees)

    The definition at ESO is:
    Moon illumination (fraction of lunar illumination, FLI) is defined as
    the fraction of the lunar disk that is illuminated at local (Chile)
    civil midnight, where 1.0 is fully illuminated. By definition, moon
    illumination equals 0 when the moon is below the local horizon.
    Because of scattered moonlight by the Earth's atmosphere, the practical
    Moon altitude is -5 deg to ensure no contribution.

    The code does not account for the Moon altitude

        moon_illumination = 0.5 * (1 + np.cos(phase_angle))
        moon_illumination ~ 0.5 * (1 - np.cos(moon_sun_sep))

    The reverse function
        phase_angle = np.arccos(2 * moon_illumination - 1.)
        moon_sun_sep ~ np.arccos(1. - 2 * moon_illumination)

    gives the phase_angle given the moon illumination

    Parameters
    ----------
    phase_angle : 'astropy.units.quantity.Quantity' deg
      the moon phase angle (one or an array of values)

    Returns
    -------
    FLI : float
        the moon fractional illumination between 0 to 1
        The output = 0 indicates a new moon, while = 1 indicates a full moon.

    Example
    -------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from moon import moon_illumination, moon_phase_angle, moon_magnitude
    >>> from moon import moon_illumination_angle
    >>> from astropy.time import Time
    >>> from astropy.coordinates import get_body, get_sun
    >>> time = Time(2021.495, format='decimalyear')
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.coordinates import AltAz
    >>> telescope_location = EarthLocation(lat=-24.615833 * u.deg,
                                       lon=-70.3975 * u.deg,
                                       height=2518 * u.m)
    >>> sun = get_sun(time)
    >>> moon = get_body("moon", time)
    >>> moon_altaz = moon.transform_to(AltAz(obstime=time,
                                         location=telescope_location))
    >>> alpha = moon_phase_angle(sun.distance, moon.distance,
    ...                          sun.separation(moon))
    >>> alpha
    <Quantity 76.02656341 deg>
    >>> moon_illumination_angle(alpha)
    0.6207360110686341
    """
    if phase_angle.unit != "deg":
        logging.error("The input to moon_illumination is an angle in deg")
    moon_illumination = 0.5 * (1 + np.cos(phase_angle))
    if isinstance(moon_illumination, u.Quantity):
        moon_illumination = moon_illumination.value
    return moon_illumination  # float


def moon_illuminance(phase_angle):
    """
    Compute the moon illuminance in footcandles I*(phase_angle)

    eq. 8 in KRISCIUNAS & SCHAEFER (KS91)
    Publications of the Astronomical Societyof the Pacific
    103:1033-1039, September 1991
    http://dx.doi.org/10.1086/132921)

    Notes
    -----
    One foot-candle equals 10.76 lumens

    Parameter
    ---------
    phase_angle : 'astropy.units.quantity.Quantity' deg
        the moon phase angle

    Returns
    -------
    : float or numpy array
        the moon illuminance in footcandles

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import moon_illuminance
    >>> phase_angle = 30 * u.deg
    >>> moon_illuminance(phase_angle)
    0.014148291529693756
    >>> moon_illuminance(0 * u.deg)
    0.029107171180666053
    """
    moon_mag = moon_magnitude(phase_angle).value
    return 10 ** (-0.4 * (moon_mag + 16.57))  # KS91 eq. 8


def darktime_sky_brightness(zenith_angle, Bzen, kext):
    """
    Compute the dark nightime sky brightness at Paranal
    as function of the zenith distance given the zenith value

    eq.2 and 3 in KS91

    Parameters
    ----------
    Bzen : float or numpy array
        zenith sky brightness
        nanoLamberts = 1e-9 / np.pi * u.lumen / u.cm**2 / u.sr

    kext : float or numpy array
        extinction coefficient at the wavelength(s) of the zenith sky
        brightness in units of magnitude / airmass

    Returns
    -------
    : float or numpy
        the dark sky brightness corrected for the line of sight extinction
        in the same units as the input Bzen

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import darktime_sky_brightness
    >>> Bzen = 79.0  # nanoLamberts mag V = 21.587 mag /sec**2
    >>> zenith_angle = 60 * u.deg
    >>> kext = 0.172  # mag / airmass
    >>> darktime_sky_brightness(zenith_angle, Bzen, kext)
    <Quantity 129.66665183>
    """
    x = airmassKS(zenith_angle)
    return Bzen * 10**(-0.4 * kext * (x - 1.)) * x


def atmospheric_scattering_function(scattering_angle):
    """
    Compute the atmospheric scattering function f(rho) in the V-band

    eq. 21 KS91

    The scattering function, f(rho), is proportional to the fraction
    of incident light scattered into a unit solid angle with a
    scattering angle rho. The scattering angle, rho, will be equal to the
    angular separation between the Moon and the sky position for
    single scattering.

    The first type is Rayleigh scattering from atmospheric gases, which will
    contribute fR(p).
    The second type is Mie scattering by atmospheric aerosols, which will
    contribute fM(p). The two terms will add

    The two terms will be added Rozenberg 1966

    Remember that the scattering functions have absorbed constant factors
    relating to unit conversions and normalizations.

    The scattering function is wavelength-dependent so that this function is
    only valid for the V-band

    At the dry, low water vapour site of Paranal during clear or photometric
    nights we expect that the night sky brightness does not show a strong
    dependence of the lunar distance (Rayleigh scattering I ~ I0 x (1 + cosΘ^2)

    Observations at shorter wavelengths (B- to R-band) in grey and dark time
    are also not affected very much by the presence of the moon if the distance
    is >~50-60 deg, and often the sky is darker at 60-70 deg away from the moon
    than at 120 deg away, when the moon is low above the horizon (see Fig 5 of
    Patat 2004, Messenger Issue 118).

    Rayleigh scattering Qscat has a 1 / lambda^4 dependency
    Mie scattering has 1 / lambda^4

    Parameters
    ----------
    scattering_angle : 'astropy.units.quantity.Quantity' deg
        the scattering angle. It is the angular separation between
        the light source and the sky position

    Returns
    -------
    f : float or numpy array
        the atmospheric scattering function

    fR : float or numpy array
        Rayleigh scattering term

    fM : float or numpy array
        Mie scattering term

    Example
    -------
    see KS91 Fig. 1 and Fig. 2
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from moon import atmospheric_scattering_function
    >>> import matplotlib.pyplot as plt
    >>> moonsep = 5 * u.deg
    >>> np.log10(atmospheric_scattering_function(moonsep))
    array([6.4698483 , 5.67226283, 6.39445168])
    >>> moonsep = 60 * u.deg
    >>> np.log10(atmospheric_scattering_function(moonsep))
    array([5.53753202, 5.4772713 , 4.65      ])
    >>> np.log10(atmospheric_scattering_function(70 * u.deg))
    array([5.46945219, 5.43076826, 4.4       ])
    >>> angle = np.arange(1, 180, 1) * u.deg
    >>> f, fR, fMm = np.log10(atmospheric_scattering_function(angle))
    >>> angle = np.arange(1, 180, 1) * u.deg
    >>> f, fR, fMm = np.log10(atmospheric_scattering_function(angle))
    >>> plt.plot(angle, f)
    >>> plt.xlim(1., 150)
    >>> plt.ylim(5.2, 6.5)
    >>> plt.xlabel('Scattering angle (deg)')
    >>> plt.ylabel('Log(f)')
    >>> plt.show()
    """
    rho = scattering_angle.to(u.deg)
    # fM is the Mie scattering term
    fM1 = 10**(6.15 - rho.value / 40.)
    # < 10 deg Shaeffer 1991
    fM2 = 6.2e7 / rho.value**2
    fM = (rho >= 10 * u.deg) * fM1  # * (rho <= 80 * u.deg)
    fM += (rho < 10 * u.deg) * fM2
    # The first term is the Rayleigh scattering
    # eq. 17 KS & Rozemnberg 1966
    # CR = 10**5.36
    fR = 10**5.36 * (1.06 + np.cos(rho)**2).value
    f = fR + fM
    return f, fR, fM


def airmassRozenberg(zenith_angle):
    """

    Parameters
    ----------
    zenith_angle : `astropy.units.quantity.Quantity` deg
        the zenith angle of the pointing

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import airmassRozenberg
    >>> zenith_angle = 60 * u.deg
    >>> airmassRozenberg(zenith_angle)
    <Quantity 1.99959141>
    """
    cosZ = np.cos(zenith_angle)
    X = (cosZ + 0.025 * np.exp(-11. * cosZ)).value
    if zenith_angle.isscalar:
        if zenith_angle == 0 * u.deg:
            return 1.
        else:
            return 1. / X
    else:
        w = np.where(zenith_angle == 0 * u.deg)[0]
        if len(w) > 0:
            X[w] = 1.
        return 1. / X


def airmassKS(zenith_angle):
    """
    Airmass appropriate for night glow

    KRISCIUNAS & SCHAEFER (KS91)
    Publications of the Astronomical Societyof the Pacific
    103:1033-1039, September 1991

    KS91 eq. 3

    see also

    Garstang (1989)
    Garstang, R. H. 1989, PASP, 101, 306

    Used by Kittler, Kocifaj & Darula 2012 Propagation of Light in the Atmo-
    spheric Environment. Springer, New York,
    Kasten, F. and Young, A.T.: Revised optical air mass
    tables and approximation formula. Applied Optics, 28, 22. 4735-4738 (1989)

    https://en.wikipedia.org/wiki/Air_mass_(astronomy)
    za is the zenith angle in degrees
    am = 1 / (cos(za) + 0.50572(96.07995-za)^(-1.6364))

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import airmassKS
    >>> zenith_angle = 60 * u.deg
    >>> airmassKS(zenith_angle)
    <Quantity 1.88982237>
    """
    am = (1. - 0.96 * np.sin(zenith_angle)**2)**(-0.5)
    return am


def plot_moon_brightness_diagram(moon_zenith_angle=70 * u.deg,
                                 sky_zenith_angle=60 * u.deg,
                                 kext=0.172, sunmoon_grid=1,
                                 moontarget_grid=1,
                                 min_moonsep=30,
                                 plot_min_moonsep=20,
                                 darksky_magV=21.587 * u.mag / u.arcsec**2):
    """
    Make de grid of Moon sky brightness

    See the routine moon_brightness for more details

    Moon avoidance implies that the minimum Moon - Target separation is 30deg.

    Parameters
    ----------
    moon_zenith_angle : `astropy.units.quantity.Quantity` deg
        optional, default=70 * u.deg
        the Moon zenith angle
        Warning: the code does not produce meaningful resultes for
        moon_zenith_angle <= 90 deg (ie above the)

    sky_zenith_angle : `astropy.units.quantity.Quantity` deg
        optional, default=60 * u.deg
        the sky position zenith angle

    kext : float, optional, default=0.172
        the sky extinction coefficient in the V-band with
        in 1 / airmass units

    darksky_magV : `astropy.units.quantity.Quantity`u.mag / u.arcsec**2
        optional, default=21.587 * u.mag / u.arcsec**2
        the dark sky V-band surface brightness in magnitude / arcsec^2

    Returns
    -------
    fig : matlplotlib object
        the figure itself

    Bmoon : float or numpy array
        the Moon brightness in units of nanoLamberts

    dmagV : `astropy.units.quantity.Quantity` u.mag / u.arcsec**2
        the Moon contribution of the V-band sky surface brightness
        in magnitude / arcsec^2

    moon_mag : `astropy.units.quantity.Quantity` u.mag
        the Moon magntiude as function of the sun-moon separation

    phase_angle : `astropy.units.quantity.Quantity` u.deg
        an array of phase angles

    sunmoon : `astropy.units.quantity.Quantity` u.deg
        an array of sun-moon angular separation

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> from moon import plot_moon_brightness_diagram
    >>> fig, Bmoon, dmagV, moon_mag, phase_angle, sunmoon =\
    ...  plot_moon_brightness_diagram()
    >>> plt.plot(sunmoon, moon_mag)
    >>> plt.xlim(0., 180.)
    >>> plt.xlabel('Sun-Moon separation angle (degree)')
    >>> plt.ylabel('Moon magnitude in the V-band')
    >>> plt.show()
    """
    # moonsep and sky_zenith_angle are of the same size
    moonsep = np.arange(min_moonsep,
                        120 + moontarget_grid, moontarget_grid) * u.deg
    lm = len(moonsep)
    sky_zenith_angle = np.repeat(sky_zenith_angle.value, lm) * u.deg
    # FLI = np.arange(0, 1.05, 0.05)
    # sunmoon = np.arccos(1. - 2 * FLI)
    sun_dist = 149.6e9 * u.km  # approximate value
    moon_dist = 384400 * u.km  # approximate value
    sunmoon = np.arange(0, 180 + sunmoon_grid, sunmoon_grid) * u.deg
    phase_angle = moon_phase_angle(sun_dist, moon_dist,
                                   sunmoon)
    # (180 * u.deg - sunmoon) ~ phase_angle
    moon_mag = moon_magnitude(phase_angle)

    lp = len(phase_angle)
    Bmoon = np.empty((lm, lp))
    dmagV = np.empty((lm, lp))
    Bdarkzen = surface_brightness_to_nanoLamberts(darksky_magV)
    for i, (sep, ssa) in enumerate(zip(moonsep, sky_zenith_angle)):
        Bdarksky = darktime_sky_brightness(ssa, Bdarkzen, kext)
        for j, pa in enumerate(phase_angle):
            Bm = np.round(moon_brightness(sep, kext, pa,
                                          ssa, moon_zenith_angle))
            Bmoon[i, j] = Bm
            dmagV[i, j] = magnitude_correction(Bdarksky, Bm).value
    print("max Moon V-mag:", -dmagV.max())
    print("min Moon V-mag:", -dmagV.min())

    X, Y = np.meshgrid(sunmoon, moonsep)
    # X, Y = np.meshgrid(FLI, moonsep)
    cs2_levels = np.array([0.18, 0.5, 0.7, 1., 1.5, 1.75, 2., 2.5, 3., 3.5])
    cs3_levels = np.arange(0, 4.1, 0.1)
    cs2_levels = np.flip(-1 * cs2_levels)
    cs3_levels = np.flip(-1 * cs3_levels)
    cmap=mpl.colormaps['binary']
    Z = dmagV
    
    # Z = np.flip(np.log10(Bmoon), 0)
    #
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    CS2 = ax.contour(X, Y, Z, colors='red',
                     levels=cs2_levels)
    ax.clabel(CS2, cs2_levels)
    CS3 = ax.contourf(X, Y, Z, levels=cs3_levels,  cmap=cmap)
    plt.ylim([plot_min_moonsep, 120.])
    plt.colorbar(CS3, label='Moon V-band contribution (mag)')
    plt.xlabel('Sun-Moon separation angle (degree)')
    plt.ylabel('Moon-object separation angle (degree)')
    if min_moonsep >= 30.:
        plt.text(62, 23, "Moon avoidance region", color='black')
    plt.text(60, 130, "Paranal dark sky mag(V)=21.6", color='black')
    plt.show()
    return fig, Bmoon, dmagV * u.mag / u.arcsec**2,\
        moon_mag, phase_angle, sunmoon


def moon_brightness(moonsep, kext, phase_angle,
                    sky_zenith_angle,
                    moon_zenith_angle,
                    moon_minalt=-5 * u.deg):
    """
    Compute an approximation of the Moon brightness in V band
    using the model of KRISCIUNAS & SCHAEFER (KS91)
    Publications of the Astronomical Society of the Pacific
    103:1033-1039, September 1991

    This method has several caveats but the authors find agreement
    with data at the 8% - 23% level.  See the paper for details.

    Notes
    -----
    Kitt Peak average UBVRI extinction (Landolt & Uomoto, 2007)
    kext=[0.621,0.281,0.162,0.119,0.075]
    -----
    The routine is only valid for moon zenith-angle <= 90 deg.

    see also the C code in OpSim_timeline_lib.c and
    John Thorstensen, Dartmouth College skycalc.c

    Table 2 of KS
    The value of rho (angular distance between the moon and sky position) and
    the sky zenith angle correspondance is
    rho sky  zenith angle in deg
    ----------------------------
    5               55
    30              30
    60               0
    90              30
    120             60

    Parameters
    ----------
    moonsep : `astropy.units.quantity.Quantity` deg
        the Moon-source separation

    kext : float or numpy array
        extinction coefficient at the wavelength(s) of the zenith sky
        brightness in units of magnitude / airmass
        kext is mostly at V-band and one uses a differential extinction
        function to compute in other bands

    phase_angle : 'astropy.units.quantity.Quantity' deg
        the moon phase angle. The moon phase angle can be computed from
        the moon phase by multiplyinf by pi for ab output in radains or 180 for
        an output in degrees

    sky_zenith_angle : 'astropy.units.quantity.Quantity' deg
        the source zenith angle (from RA, dec of the source and
        the observation time)

    moon_zenith_angle : 'astropy.units.quantity.Quantity` deg
        the moon zenith angle at the time of the observation

    Returns
    -------
    : float or numpy array
        the Moon brightness in units of nanoLamberts

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from moon import *
    >>> kext = 0.172  # mag / airmass
    >>> # ----------------------------
    >>> sky_zenith_angle = 30 * u.deg
    >>> moon_zenith_angle = 60 * u.deg
    >>> moonsep =  30 * u.deg  # rho
    >>> phase_angle = 30 * u.deg  # alpha
    >>> Bmoon= moon_brightness(moonsep, kext, phase_angle,
    ...                        sky_zenith_angle,
    ...                        moon_zenith_angle)
    >>> Bmoon
    <Quantity 1160.34328112>
    >>> # KS Table 2 : 1160
    >>> Bzen = 79.0  # nanoLamberts mag V = 21.587 mag /sec**2
    >>> darktime_sky_brightness(zenith_angle, Bzen, kext)
    >>> # ------------------------------
    >>> sky_zenith_angle = 0 * u.deg
    >>> moon_zenith_angle = 60 * u.deg
    >>> moonsep =  60 * u.deg  # rho
    >>> phase_angle = 30 * u.deg  # alpha
    >>> Bmoon= moon_brightness(moonsep, kext, phase_angle,
    ...                        sky_zenith_angle,
    ...                        moon_zenith_angle)
    >>> Bmoon
    <Quantity 529.75418957>
    >>> # KS Table 2 : 530
    >>> # --------------------------------
    >>> sky_zenith_angle = 55 * u.deg
    >>> moon_zenith_angle = 60 * u.deg
    >>> moonsep =  5 * u.deg  # rho
    >>> phase_angle = 30 * u.deg  # alpha
    >>> Bmoon= moon_brightness(moonsep, kext, phase_angle,
    ...                        sky_zenith_angle,
    ...                        moon_zenith_angle)
    >>> Bmoon
    >>> <Quantity 7216.42998734>
    >>> # KS Table 2: 7216
    >>> # -------------------------------------------------
    >>> # KS Table 2
    >>> darksky_magV = 21.587 * u.mag / u.arcsec**2
    >>> kext = 0.172
    >>> moon_zenith_angle = 60 * u.deg
    >>> # moonsep and sky_zenith_angle are of the same size
    >>> moonsep = [5, 30, 60, 90, 120] * u.deg
    >>> sky_zenith_angle = [55., 30., 0., 30., 60.] * u.deg
    >>> phase_angle = [30, 60, 90, 120] * u.deg
    >>> Bmoon = np.empty((5, 4))
    >>> dmagV = np.empty((5, 4))
    >>> Bdarkzen = surface_brightness_to_nanoLamberts(darksky_magV)
    >>> for i, (sep, ssa) in enumerate(zip(moonsep, sky_zenith_angle)):
    ...    Bdarksky = darktime_sky_brightness(ssa, Bdarkzen, kext)
    ...    for j, pa in enumerate(phase_angle):
    ...       Bm = np.round(moon_brightness(sep, kext, pa,
    ...                                     ssa, moon_zenith_angle))
    ...       Bmoon[i, j] = Bm
    ...       dmagV[i, j] = magnitude_correction(Bdarksky, Bm).value
    >>> # -------------------------------------------------

    Notes
    -----
    For a given observation time, the moon_zenith_angle and the moon
    phase angle will be determined.
    For a list of sources with RA and Dec, the source zenith angle and the
    Moon-source separation can be computed for all the sources
    The extinction is for one band. In this case the Moon brightness for
    the different RA and Dec can be computed.

    See also SpecSim
    https://specsim.readthedocs.io/en/stable/index.html
    https://specsim.readthedocs.io/en/stable/api/
    specsim.atmosphere.krisciunas_schaefer.html
    #specsim.atmosphere.krisciunas_schaefer

    """
    # moon_angular_radius = 0.26 * u.deg
    # Check if the moon is below the astronomical/civil
    # twilight alt + moon radius
    # We consider no moon contribution the the sky brightness
    if moon_zenith_angle.isscalar:
        if moon_zenith_angle >= (90 * u.deg - moon_minalt):
            return 0.0
        elif moon_zenith_angle > 90 * u.deg:
            moon_zenith_angle2 = 90 * u.deg
        else:
            moon_zenith_angle2 = np.copy(moon_zenith_angle)
    else:  # create an independent copy
        moon_zenith_angle2 = np.copy(moon_zenith_angle)
        wneg = moon_zenith_angle > 90 * u.deg
        moon_zenith_angle2[wneg] = 90 * u.deg

    # print("Moon zenith angle as used:", moon_zenith_angle2)
    # Calculate the illuminance of the moon outside the atmosphere in
    # foot-candles (eqn. 8).
    Is = moon_illuminance(phase_angle)  # candlefoots = 10.76 lumens

    # Calculate the scattering function (eqn.21).
    f, _, _ = atmospheric_scattering_function(moonsep)

    # Calculate the scattering airmass along the lines of sight to the
    # observation and moon (eqn. 3).
    # the KS airmass function is not valid for angles > 90 degrees
    XZm = airmassKS(moon_zenith_angle2)
    XZ = airmassKS(sky_zenith_angle)

    # extinction
    extinction = 10**(-0.4 * kext * XZm) * (1. - 10**(-0.4 * kext * XZ))

    Bmoon = f * Is * extinction

    # Here it makes the code to return zero Moon contribution to the
    # sky brightness below the horizon
    # The Moon contribution is null for
    # a Moon altitude below moon_minalt = -5 degrees
    if not moon_zenith_angle.isscalar:
        print(moon_zenith_angle)
        wzero = moon_zenith_angle >= (90 * u.deg - moon_minalt)
        Bmoon[wzero] = 0.

    return Bmoon  # moonsep, sky_zenith_angle


def surface_brightness_to_nanoLamberts(SB):
    """
    Parameters
    ----------
    SB : `astropy.units.quantity.Quantity' in mag / arcsec^2
        the surface brightness in the V-band

    Examples
    --------
    >>> from moon import surface_brightness_to_nanoLamberts
    >>> from moon import nanoLamberts_to_V_surface_brightness
    >>> SBdark = nanoLamberts_to_V_surface_brightness(79.0)
    >>> surface_brightness_to_nanoLamberts(SBdark)
    78.9999999999999
    """
    SB = SB.value
    BnL = 34.08 * np.exp(20.7233 - SB * 0.92104)
    return BnL


def nanoLamberts_to_V_surface_brightness(BnL):
    """
    Convert from nanoLamberts to to mag / arcsec**2 using eqn.19 of
    Garstang, "Model for Artificial Night-Sky Illumination",
    PASP, vol. 98, Mar. 1986, p. 364 (http://dx.doi.org/10.1086/131768)

    Notes
    -----
    1 lambert (L) = (1/π) candela per square centimetre
                  = 1e4 / np.pi candela per square centimetre
                  = 3183.098861837907 candela per square centimetre

    The candela is the base unit of luminous intensity in the International
    System of Units.

    1 candela (= lumen per steradian)
    1 nanolambert (5550 A radiation) = 1.31e6 photons s^-1 cm^-2 sr^-1
    1 nanolambert = (1e-9 / pi) lumen per square centimetre per steradian
    V-band zeroth magnitude 3640 Jy :

    In astropy candela is cd
    Wikipedia
    A truly dark sky has a surface brightness of 2x10^-4  cd m^-2
    or 21.8 mag arcsec

    https://stjarnhimlen.se/comp/radfaq.html
    1 lambert = 1/pi cd/cm2 = 1 lumen/cm2 for perfect duffusor
    Apparent magnitude per square degree is a radiance, luminance, intensity,
    or "specific intensity". This is sometimes also called
    "surface brightness".

    Parameters
    ----------
    BnL : float or `astropy.units.quantity.Quantity' with no units specified
        a luminance in unit of nanoLamberts
        1e-9 / np.pi * u.lumen / u.cm**2 / u.sr

    Returns
    -------
    : `astropy.units.quantity.Quantity' mag / arcsec^2
        the surface brightness in mag / arcsec^2 in the V-band

    in KS91: Bzen = 79.0  # nanoLamberts mag V = 21.587 mag /sec**2

    Examples
    --------
    >>> from moon import nanoLamberts_to_V_surface_brightness
    >>> SBdark =  nanoLamberts_to_V_surface_brightness(79.0)
    >>> SBdark
    <Quantity 21.58707857 mag / arcsec2>
    >>> SBmoon = nanoLamberts_to_V_surface_brightness(Bmoon)
    """
    return ((20.7233 - np.log(BnL / 34.08)) / 0.92104 *
            u.mag / (u.arcsec ** 2))


def magnitude_correction(Bdarksky, Bmoon):
    """
    Compute the contribution of the Moon to the sky brightness
    in magnitude

    See KS91 eq. 22

    Bdarksky and Bmoon are in the same linear units of photon flux
    In KS91, they are computed in nanoLamberts

    Parameters
    ----------
    Bdarksky : float or numpy array of floats
        the dark sky luminance

    Bmoon : float or numpy array of floats
        the moon luminance in the same units as for Bdarksky

    Returns
    -------
    dmag : `astropy.units.quantity.Quantity' in mag / arcsec^2
        the contribution of the Moon in addition to the dark sky

    Ref
    ---
    KRISCIUNAS & SCHAEFER (KS91)
    Publications of the Astronomical Societyof the Pacific
    103:1033-1039, September 1991

    Examples
    --------
    >>> import astropy.units as u
    >>> from moon import magnitude_correction
    >>> from moon import darktime_sky_brightness
    >>> from moon import moon_brightness
    >>> from moon import surface_brightness_to_nanoLamberts
    >>> from moon import nanoLamberts_to_V_surface_brightness
    >>> darksky_magV = 21.587 * u.mag / u.arcsec**2
    >>> # Bdarkzen = 79.0  # nanoLamberts mag V = 21.587 mag /sec**2
    >>> Bdarkzen = surface_brightness_to_nanoLamberts(darksky_magV)
    >>> sky_zenith_angle = 9. * u.deg
    >>> kext = 0.172  # mag / airmass
    >>> Bdarksky = darktime_sky_brightness(sky_zenith_angle, Bdarkzen, kext)
    >>> moon_zenith_angle = 79 * u.deg
    >>> moonsep = moon_zenith_angle + sky_zenith_angle  # rho
    >>> phase_angle = -30 * u.deg  # alpha
    >>> Bmoon = moon_brightness(moonsep, kext, phase_angle,
    ...                         sky_zenith_angle,
    ...                         moon_zenith_angle)
    >>> dmagV = magnitude_correction(Bdarksky, Bmoon)
    >>> SBdark = nanoLamberts_to_V_surface_brightness(Bdarksky)
    >>> SBmoon = nanoLamberts_to_V_surface_brightness(Bmoon)

    """
    dmag = -2.5 * np.log10((Bmoon + Bdarksky)/Bdarksky)
    return dmag * u.mag / (u.arcsec ** 2)


def krisciunas_schaefer_tests(sun_moon_separation,
                              darksky_magV=21.587 * u.mag / u.arcsec**2,
                              kextV=0.162):
    """
    Output the sky brightness for specific test cases
    sun_moon_separation = 37, 90, 143 for dark, grey, bright time

    From Patat Table A.1
    extinction  above Paranal U, B, V, R are 0.43, 0.22, 0.11, 0.07, 0.05

    >>> import astropy.units as u
    >>> from moon import krisciunas_schaefer_tests
    >>> krisciunas_schaefer_tests(37 * u.deg)
    (<Quantity 21.28981758 mag / arcsec2>,
     <Quantity 21.42923384 mag / arcsec2>,
     <Quantity -0.13941625 mag / arcsec2>,
     <Quantity 142.91121585 deg>,
     0.101149001951925)
    >>> krisciunas_schaefer_tests(90 * u.deg)
    (<Quantity 20.317094 mag / arcsec2>,
     <Quantity 21.42923384 mag / arcsec2>,
     <Quantity -1.11213983 mag / arcsec2>,
     <Quantity 89.85277565 deg>,
     0.5012847700612147)
    >>> krisciunas_schaefer_tests(143. * u.deg)
    (<Quantity 19.04250455 mag / arcsec2>,
     <Quantity 21.42923384 mag / arcsec2>,
     <Quantity -2.38672928 mag / arcsec2>,
     <Quantity 36.9115795 deg>,
     0.899781647284696)
    """
    # absolute value of the phase angle
    moon_zenith_angle = 45 * u.deg
    sky_zenith_angle = (90. - 56.443) * u.deg
    moonsep = 45 * u.deg
    earth_sun_distance = au
    earth_moon_distance = 384399. * u.km  # mean distance to the Moon
    phase_angle = moon_phase_angle(earth_sun_distance,
                                   earth_moon_distance,
                                   sun_moon_separation)

    Bdarkzen = surface_brightness_to_nanoLamberts(darksky_magV)

    Bdarksky = darktime_sky_brightness(sky_zenith_angle, Bdarkzen, kextV)

    Bmoon = moon_brightness(moonsep, kextV, phase_angle,
                            sky_zenith_angle,
                            moon_zenith_angle)

    dmagV = magnitude_correction(Bdarksky, Bmoon)
    SBdark = nanoLamberts_to_V_surface_brightness(Bdarksky)
    SBsky = SBdark + dmagV

    FLI = moon_illumination_angle(phase_angle)

    return SBsky, SBdark, dmagV, phase_angle, FLI


def krisciunas_schaefer(obsTime, telescope,
                        darksky_magV=21.587 * u.mag / u.arcsec**2,
                        kextV=0.162,
                        sky_altaz=None,
                        skyPosition=None,
                        Rozenberg=False,
                        Paranal=True,
                        verbose=0,
                        grid=False):
    """
    Calculate the scattered moonlight surface brightness in V band.
    Based on Krisciunas and Schaefer (1991)

    zodiacal light was ignored

    Ref
    ----
    R. H. Garstang. Night-sky brightness at observatories and sites.
    Astronomical Society of the Pacific, 101:306-329, March 1989.

    K. Krisciunas. Optical night-sky brightness at mauna kea over the
    course of a complete sunspot cycle. Astronomical Society of the Pacific,
    109:1181-1188, October 1997.

    Krisciunas and Schaefer
    "A model of the brightness of moonlight",
    PASP, vol. 103, Sept. 1991, p. 1033-1039
    (http://dx.doi.org/10.1086/132921).
    Equation numbers in the code comments refer to this paper.

    Kieffer & Stone. THE SPECTRAL IRRADIANCE OF THE MOON
    The Astronomical Journal, 129:2887-2901, 2005 June

    Roman et al. Correction of a lunar-irradiance model for aerosol
    optical depth retrieval and comparison with a star photometer
    Atmos. Meas. Tech., 13, 6293–6310, 2020
    https://doi.org/10.5194/amt-13-6293-2020

    Zhang Hui-Hua et al. Atmospheric extinction coefficients and
    night sky brightness at the Xuyi Observation Station
    Research in Astron. Astrophys. 2013 Vol. 13 No. 4, 490-500
    http://www.raa-journal.org http://www.iop.org/journals/raa

    Neilsen et al.
    Dark Energy Survey's Observation Strategy, Tactics, and Exposure Scheduler
    https://arxiv.org/pdf/1912.06254.pdf
    The DES project has a specific twilight model.
    2019, skybright. https://github.com/ehneilsen/skybright

    F. Patat
    A&A 400, 1183-1198 (2003)
    DOI: 10.1051/0004-6361:20030030

    UBVRI night sky brightness during sunspot maximum at ESO-Paranal
    22.3, 22.6, 21.6 20.9 and 19.7 mag arcsec-2 in U, B, V, R and I
    see Paranal_sky_brightness

    From Patat Table A.1
    extinction  above Paranal U, B, V, R are 0.43, 0.22, 0.11, 0.07, 0.05

    see also
    https://github.com/mcoughlin/skybrightness
    https://arxiv.org/abs/1510.07574
    https://www.michaelwcoughlin.com/publication/coughlin-daytime-2015/

    Notes
    -----
    In the future, this approximation will be compared to the ESO sky model
    https://escience.aip.de/readthedocs/OpSys/etc/obs-setsky/esoskymodel/
    also
    https://github.com/AstarVienna/skycalc_ipy/blob/master/docs/source/
    index.rst

    Parameters
    ----------
    obsTime : astropy.time.Time` one of multiple entries
        Time of observation

    telescope: `astropy.coordinates.earth.EarthLocation`
        The location on the Earth. This can be specified either as an
        EarthLocation object or as anything that can be transformed to an
        ITRS frame.

    darksky_magV`: astropy.units.quantity.Quantity' u.mag / u.arcsec**2
        Sky measured/estimated V-band dark time magnitude
        optional, default=21.587 * u.mag / u.arcsec**2

    kextV : float, optional, default=0.162
        the atmospheric extinction coefficient at the wavelength(s) of the
        zenith sky brightness in units of magnitude / airmass

    Rozenberg : bool, optional, default=False
        to use the Rozenberg's airmass definition

    grid : bool, optional, default=False
        whether to create a grid of sky pointings with sky_altaz values

    verbose : int, optional, default = 0
        verbose level

    Returns
    -------
    SBsky : `astropy.units.quantity.Quantity' mag / arcsec^2
        the atmospheric sky surface brightness in mag / arcsec^2 in the V-band

    SBdark : `astropy.units.quantity.Quantity' mag / arcsec^2
        the dark sky surface brightness in mag / arcsec^2 in the V-band
        ie WITHOUT the Moon contribution, only the scattered sunlight is taken
        into account

    moon_altaz: astropy altaz object
        the moon altitude and azimuth (the azimuth is
        in the astropy convention)

    moonsep : `astropy.units.quantity.Quantity' in degrees
        the moon-pointing angular separation in degrees

    FLI : float
        the moon illumination

    sun_altaz : astropy altaz object
        the Sun altitude and azimuth (the azimuth is
        in the astropy convention)

    sky_azimuth : `astropy.units.quantity.Quantity' in degrees
        the sky pointing azimuth using the astropy convention

    sky_zenith_angle : `astropy.units.quantity.Quantity' in degrees
        the sky zenith angle

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.time import Time
    >>> from astropy.coordinates import SkyCoord
    >>> from moon import krisciunas_schaefer
    >>> from moon import plot_sky_brightness
    >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
                                  lon=-70.3975 * u.deg,
                                  height=2518 * u.m)
    >>> obsTime = Time('2021-09-15T6:00:00')
    >>> c = SkyCoord(ra=80.103 * u.degree, dec=-45 * u.degree)
    >>> SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz,\
    ...    sky_azimuth, sky_zenith_angle =\
    ...    krisciunas_schaefer(obsTime, telescope,
    ...                        skyPosition=c)
    >>> moon = get_body("moon", obsTime)
    >>> c = SkyCoord(ra=moon.ra, dec=moon.dec)
    >>> SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz,\
    ...    sky_azimuth, sky_zenith_angle =\
    ...    krisciunas_schaefer(obsTime, telescope,
    ...                        skyPosition=c, Paranal=True)
    >>> #
    >>> c = SkyCoord(ra=[80.103, 45.] * u.degree,
    ...              dec=[-45., -23.] * u.degree)
    >>> SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz,\
    ...    sky_azimuth, sky_zenith_angle =\
    ...    krisciunas_schaefer(obsTime, telescope,
    ...                        skyPosition=c, Paranal=True)
    >>> c = SkyCoord(ra=[80.103] * u.degree,
    ...              dec=[-45.] * u.degree)
    >>> SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz,\
    ...    sky_azimuth, sky_zenith_angle =\
    ...    krisciunas_schaefer(obsTime, telescope,
    ...                        skyPosition=c, Paranal=True)
    >>> c = SkyCoord(ra= 45. * u.degree,
    ...              dec=-23. * u.degree)
    >>> SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz,\
    ...    sky_azimuth, sky_zenith_angle =\
    ...    krisciunas_schaefer(obsTime, telescope,
    ...                        skyPosition=c, Paranal=True)
    >>> obsTime = Time('2021-10-5T6:00:00')
    >>> c = SkyCoord(ra=80.103 * u.degree, dec=-45 * u.degree)
    >>> SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz,\
    ...    sky_azimuth, sky_zenith_angle =\
    ...    krisciunas_schaefer(obsTime, telescope,
    ...                        skyPosition=c)
    >>> # if alt=90 for the sky position, the result does not depend on az
    >>> sky_altaz = {'az': 5 * u.deg, 'alt': 90 * u.deg}
    >>> SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz,\
    ...     sky_azimuth, sky_zenith_angle =\
    ...     krisciunas_schaefer(obsTime, telescope,
    ...                         sky_altaz=sky_altaz)
    >>> #
    >>> obsTime = Time(2023.8417410182594, format='decimalyear')

    """
    SBnull = 0. * u.mag / u.arcsec**2
    if verbose > 0:
        if Paranal:
            logging.info("Paranal sky model")
        else:
            logging.info("Standard sky model")

    if sky_altaz is None:
        sky_altaz = skyPosition.transform_to(AltAz(obstime=obsTime,
                                             location=telescope))
        sky_zenith_angle = 90. * u.deg - sky_altaz.alt
        sky_azimuth = sky_altaz.az
    else:
        sky_azimuth = sky_altaz['az']
        sky_zenith_angle = 90. * u.deg - sky_altaz['alt']

    # airmass
    if Rozenberg:
        airmass = airmassRozenberg(sky_zenith_angle)
        if verbose > 0:
            logging.info("Rozenberg airmass formula")
    else:
        airmass = airmassKS(sky_zenith_angle)
        if verbose > 0:
            logging.info("K&S airmass formula")

    if isinstance(airmass, np.ndarray):
        w = np.where((airmass > 5.) | (airmass < 0.))[0]
        if verbose > 0:
            logging.info('max airmass %s', np.max(airmass))
    else:
        if (airmass > 5.) | (airmass < 0.):
            if verbose > 0:
                logging.info('airmass %s', airmass)
            w = [1]

    # get the Sun and Moon characteristics at obsTime
    sun = get_sun(obsTime)
    sun_altaz = sun.transform_to(AltAz(obstime=obsTime,
                                 location=telescope))
    moon = get_body("moon", obsTime)
    moon_altaz = moon.transform_to(AltAz(obstime=obsTime,
                                         location=telescope))

    # the moon zenigh angle and azimuth at the time of the observation
    moon_zenith_angle = 90. * u.deg - moon_altaz.alt
    moon_azimuth = moon_altaz.az

    sun_azimuth = sun_altaz.az
    if verbose > 0:
        logging.info("Sun altitude %s", sun_altaz.alt)

    if skyPosition is not None:
        moonsep = moon.separation(SkyCoord(ra=skyPosition.ra,
                                           dec=skyPosition.dec))
    else:
        # if sky_zenith_angle = 0 (zenith), then moonsep = moon_zenith_sngle
        # Need to be checked
        cos_sep = (np.cos(moon_zenith_angle) * np.cos(sky_zenith_angle) +
                   np.cos(moon_azimuth - sky_azimuth) *
                   np.sin(moon_zenith_angle) *
                   np.sin(sky_zenith_angle))
        moonsep = np.arccos(cos_sep).to(u.deg)

    sun_az_sep = sun_azimuth - sky_azimuth
    moon_zenith_angle = 90. * u.deg - moon_altaz.alt

    # The moon phase angle in degrees is the moon phase * 180
    # without the sign attached to it
    phase_angle = moon_phase_angle(sun.distance, moon.distance,
                                   sun.separation(moon))
    FLI = moon_illumination_angle(phase_angle)

    Bmoon = moon_brightness(moonsep, kextV, phase_angle,
                            sky_zenith_angle,
                            moon_zenith_angle)

    # moonsep.shape : ngrid x ngrid
    # sky_zenith_angle : ngrid
    # the sun
    if sun_altaz.alt > -11 * u.deg:
        if verbose > 0:
            logging.info("Routine is not valid for Sun alt > -11 degrees")
        return SBnull, SBnull, moon_altaz, moonsep, FLI, sun_altaz, \
            sky_azimuth, sky_zenith_angle

    if len(w) > 0:
        if verbose > 0:
            logging.info("Some sky positions are too low")
            logging.info("Find another observing time")
        return SBnull, SBnull, moon_altaz, moonsep, FLI, sun_altaz, \
            sky_azimuth, sky_zenith_angle

    if not sun_az_sep.isscalar:
        lp = len(sun_az_sep)        # related to coord azimuth
    else:
        lp = 0

    if Paranal:
        mdark = Paranal_sky_brightness(180 * u.deg, 'V')
        if verbose > 0:
            logging.info('Dark sky Paranal zenith sky brightness %s', mdark)
        darksky_magV = twilight_lsst('G', mdark,
                                     sky_zenith_angle,
                                     sun_altaz.alt,
                                     sun_az_sep,
                                     Paranal=Paranal,
                                     grid=grid,
                                     verbose=verbose)
        if darksky_magV is np.nan:
            SBsky = np.nan
            SBdark = np.nan
            return SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz, \
                sky_azimuth, sky_zenith_angle

        Bdarksky = surface_brightness_to_nanoLamberts(darksky_magV)
    else:
        Bdarkzen = surface_brightness_to_nanoLamberts(darksky_magV)
        # correction for the sky position zenith angle
        Bdarksky = darktime_sky_brightness(sky_zenith_angle,
                                           Bdarkzen, kextV)
        if not sun_az_sep.isscalar:
            Bdarksky = np.tile(Bdarksky, (lp, 1))

    dmagV = magnitude_correction(Bdarksky, Bmoon)
    if sun_az_sep.isscalar and verbose > 0:
        logging.info("Sky pointing zenith angle %s", sky_zenith_angle)
        logging.info("Sun azimuth %s", sun_azimuth)
        logging.info("Sky azimuth %s", sky_azimuth)
        logging.info("Sky-Sun azimuth separation %s", sun_az_sep)
        logging.info("Sky surface brightness (No Moon, Sun) %s",
                     darksky_magV)
        logging.info("kextV %s", kextV)
        logging.info("Moon FLI %s", FLI)
        logging.info("Moon-object separation %s,", moonsep)
        logging.info("Moon zenith angle %s", moon_zenith_angle)
        logging.info("Moon phase angle %s", phase_angle)
        logging.info("Moon phase %s", moon_phase(FLI))
        logging.info("BMoon %s", nanoLamberts_to_V_surface_brightness(Bmoon))
        logging.info("Moon V-band magnitude correction %s", dmagV)
    SBdark = nanoLamberts_to_V_surface_brightness(Bdarksky)
    SBsky = SBdark + dmagV

    return SBsky, SBdark, moon_altaz, moonsep, FLI, sun_altaz, \
        sky_azimuth, sky_zenith_angle


def moon_phase(moon_illumination):
    """
    Compute the moon phase from the moon fraction illumination

    Phase of the moon from 0.0 (full) to 1.0 (new), which can be calculated
        as abs((d / D) - 1) where d is the time since the last new moon
        and D = 29.5 days is the period between new moons.  The corresponding
        illumination fraction is ``0.5*(1 + cos(pi * moon_phase))``.

    Day of lunar phase. Lunar cycle 29.5 days.
      age,days   Phase_angle  Name
      0          -180         new
      7.375       -90         first quarter
      14.75         0         full
      22.125       90         last quarter
      29.50       180         new

    Ref.
    https://specsim.readthedocs.io/en/stable/api/
    specsim.atmosphere.krisciunas_schaefer.html
    #specsim.atmosphere.krisciunas_schaefer

    Parameters
    ----------
    moon_illumination : float or array of floats.
          Moon illumination (between 0.0 and 1.0) aka FLI

    Returns
    -------
    : float between 0 and 1
    The moon phase: 0.0 (full) to 1.0 (new)

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import moon_phase
    >>> moon_illumination = 0.2
    >>> moon_phase(moon_illumination)
    0.7048327646991335
    """
    routine = 'moon_phase'
    if moon_illumination < 0.:
        logging.info("Routine %s", routine)
        logging.info("Error: Negative input moon illumination")
        return np.nan
    if moon_illumination > 1.:
        logging.info("Routine %s", routine)
        logging.info("Error: Moon illumination greater than 1")
        return np.nan
    mp = (np.arccos(2 * moon_illumination - 1.) / np.pi)
    if isinstance(mp, u.Quantity):
        mp = mp.value
    return mp


def plot_sky_brightness(obsTime, telescope, moon=False, darksky=False,
                        darksky_magV=21.587 * u.mag / u.arcsec**2,
                        kextV=0.162, ngrid=360, max_zenith=70 * u.deg,
                        cmap='YlGnBu', figure_size=(8, 6), Paranal=True,
                        test=False):
    """
    Create a polar plot of the scattered moon brightness in V band.
    Evaluates the model of :func:`krisciunas_schaefer` on a polar grid of
    observation pointings, for a fixed moon position and phase.
    This method requires that matplotlib is installed.

    Parameters
    ----------
    obsTime : astropy.time.Time` one of multiple entries
        Time of observation

    telescope: `astropy.coordinates.earth.EarthLocation`
        for example
        telescope = EarthLocation(lat=-24.615833 * u.deg,
                                  lon=-70.3975 * u.deg,
                                  height=2518 * u.m)

    darksky_magV :
        See :func:`krisciunas_schaefer`.

    kextV :
        See :func:`krisciunas_schaefer`.

    ngrid : int
        Size of observing location zenith and azimuth grids to use.

    cmap : str
        Name of the matplotlib color map to use.

    figure_size : tuple or None
        Tuple (width, height) giving the figure dimensions in inches.

    max_zenith : `astropy.units.quantity.Quantity` deg, optional
        the maximum zenith angle, default = 30 deg

    Returns
    -------
    tuple
        Tuple (fig, ax, cax) of matplotlib objects created for this plot. You
        can ignore these unless you want to make further changes to the plot.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.time import Time
    >>> from astropy.coordinates import SkyCoord
    >>> from moon import krisciunas_schaefer
    >>> from moon import plot_sky_brightness
    >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
                                  lon=-70.3975 * u.deg,
                                  height=2518 * u.m)
    >>> obsTime = Time(2023.8416744276963, format='decimalyear')
    >>> obsTime.format='iso'
    >>> fig, ax, cax, SB = plot_sky_brightness(obsTime,
    ...                                        telescope,
    ...                                        Paranal=True, ngrid=250,
    ...                                        moon=False)
    >>> obsTime = Time('2021-09-14T23:30:00')
    >>> fig, ax, cax, SB = plot_sky_brightness(obsTime,
    ...                                        telescope, ngrid=250, moon=True)
    >>> fig, ax, cax, SB = plot_sky_brightness(obsTime,
    ...                                        telescope, ngrid=250,
    ...                                        Paranal=True,
    ...                                        moon=False, darksky=True)
    >>> fig, ax, cax, SB = plot_sky_brightness(obsTime,
    ...                                        telescope, ngrid=250,
    ...                                        Paranal=True,
    ...                                        moon=False, darksky=False)
    """

    # Build a grid in observation (zenith, azimuth).
    # Build a grid in observation (zenith, azimuth).
    sky_zenith = np.linspace(0., max_zenith.value, ngrid,
                             endpoint=True) * u.deg
    sky_az = (np.linspace(0., 360., ngrid) * u.deg)[:, np.newaxis]
    sky_altaz = dict()
    sky_altaz['az'] = sky_az
    sky_altaz['alt'] = 90. * u.deg - sky_zenith

    # Calculate the V-band sky surface brightness using
    # the model of krisciunas & Schaefer
    SBsky, SBdark, moon_altaz, _, FLI, sun_altaz, _, _ =\
        krisciunas_schaefer(obsTime, telescope, grid=True,
                            darksky_magV=darksky_magV, kextV=kextV,
                            sky_altaz=sky_altaz, Paranal=Paranal)

    moon_azimuth = moon_altaz.az
    moon_zenith = 90. * u.deg - moon_altaz.alt
    mphase = moon_phase(FLI)
    sun_azimuth = sun_altaz.az
    sun_alt = sun_altaz.alt
    moon_alt = moon_altaz.alt

    # Initialize the plot. We are borrowing from:
    # http://blog.rtwilson.com/producing-polar-contour-plots-with-matplotlib/
    fig, ax = plt.subplots(
        figsize=figure_size, subplot_kw=dict(projection='polar'))
    r, theta = np.meshgrid(sky_zenith.to(u.deg).value,
                           sky_az.to(u.rad).value[:, 0], copy=False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0., 90.)

    # Draw a polar contour plot.
    if moon:
        SB = SBsky - SBdark
        name = 'Moon delta V-band [mag/arcsec2]'
    elif darksky:
        SB = SBdark
        name = 'Sky (Sun + astronomical backgorund) V-band [mag/arcsec$^{2}$]'
    else:
        SB = SBsky
        name = 'Sky V-band [mag/arcsec$^{2}$]'

    cax = ax.contourf(theta, r, SB.value, 50, cmap=cmap)
    fig.colorbar(cax).set_label(name)
    # cax = ax.contour(theta, r, SB.value, 50)

    # Draw a point indicating the moon position.
    if not darksky:
        plt.scatter(moon_azimuth.to(u.rad).value,
                    moon_zenith.to(u.deg).value,
                    s=20., marker='o', color='w', lw=0.5, edgecolor='k')

    # Draw a point indicating the azimuth position of the sun
    plt.scatter(sun_azimuth.to(u.rad).value,
                90.,
                s=150., marker='*', color='red', lw=0.5, edgecolor='k')

    # Add labels.
    xy, coords = (1., 0.), 'axes fraction'
    plt.annotate('$k_V$ = {0:.3f}'.format(kextV),
                 xy, xy, coords, coords,
                 horizontalalignment='right', verticalalignment='top',
                 size='large', color='k')
    xy, coords = (0., 0.), 'axes fraction'

    plt.annotate('$\\phi$ = {0:.1f}%'.format(100. * mphase),
                 xy, xy, coords, coords,
                 horizontalalignment='left', verticalalignment='top',
                 size='large', color='k')

    xy, coords = (1., 1.1), 'axes fraction'
    plt.annotate('FLI = {0:.1f}'.format(FLI),
                 xy, xy, coords, coords,
                 horizontalalignment='right', verticalalignment='top',
                 size='large', color='k')

    xy, coords = (0.8, 1.1), 'axes fraction'
    plt.annotate('Moon alt = {0:.1f}'.format(moon_alt),
                 xy, xy, coords, coords,
                 horizontalalignment='right', verticalalignment='top',
                 size='large', color='k')

    xy, coords = (1., 1.05), 'axes fraction'
    plt.annotate('Sun alt = {0:.1f}'.format(sun_alt),
                 xy, xy, coords, coords,
                 horizontalalignment='right', verticalalignment='top',
                 size='large', color='k')

    xy, coords = (0., 1.1), 'axes fraction'
    plt.annotate(obsTime,
                 xy, xy, coords, coords,
                 horizontalalignment='left', verticalalignment='top',
                 size='large', color='k')

    plt.tight_layout()
    if not test:
        plt.show()
    return fig, ax, cax, SB


def Paranal_sky_brightness(zenith_sun, filter):
    """
    Return the sky brightness at zenith for different filters

    UBVRI twilight sky brightness at ESO-Paranal

    Table 4 from Patat et al. 2006
    A&A 455, 385-393 (2006)
    DOI: 10.1051/0004-6361:20064992

    Notes
    -----
    The measurements were made for pointings with zenith angles < 40 deg

    Civilian twilight: zenith_sun >= 90 + 6 = 96 deg
    Astronomical twilight: >= 90 + 18 = 108 deg

    Parameters
    ----------
    zenith_sun : `astropy.units.quantity.Quantity` deg
        the Sun zenith angle (must be between 95 and 110)
        For values < 95 deg, np.nan is return
        For values >= 106 deg, the value at 106 deg is provided

    filter : str
        one of the available filters
        'U', 'B', 'V', 'R', 'I'

    Returns
    -------
    : `astropy.units.quantity.Quantity` mag / arcsec**2
        the sky brightness in the filter

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import Paranal_sky_brightness
    >>> Paranal_sky_brightness(110 * u.deg, 'V')
    <Quantity 21.785 mag / arcsec2>
    >>> Paranal_sky_brightness(110 * u.deg, 'V')
    """
    twilight_param = {'U': [11.78, 1.376, -0.039],
                      'B': [11.84, 1.411, -0.041],
                      'V': [11.84, 1.518, -0.057],
                      'R': [11.40, 1.567, -0.064],
                      'I': [10.93, 1.470, -0.062]}
    if zenith_sun < 95 * u.deg:
        logging.error("Sun zenith angle must be > 95 deg")
        return np.nan
    try:
        a = twilight_param[filter]
    except KeyError:
        logging.error("Filter non valid")
        logging.info("Valid filters are U, B, V, R, I")
        return np.nan
    if zenith_sun >= 106. * u.deg:
        zenith_sun = 106. * u.deg
    dz = (zenith_sun - 95 * u.deg).value
    return (a[0] + (a[2] * dz + a[1]) * dz) * u.mag / u.arcsec**2


def relative_sky_brightness(zenith_angle):
    """
    Relative sky brightness increase as function of the zenith distance
    Implement an interpolation of the ratios in Table 6 in Roach 1964
    Space Science Reviews 3, 512-540 (1964)

    ToDo: implement the correcton factor in Patat Table C.1 and eq. C.3

    Parameter
    ---------
    zenith_angle : `astropy.units.quantity.Quantity` deg
        the pointing zenith angle

    Returns
    -------
    : `astropy.units.quantity.Quantity` mag
        the extra flux (in magntiudes <=0) due the pointing zenith angle

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import relative_sky_brightness
    >>> relative_sky_brightness(45 * u.deg)
    <Quantity -0.16697914 mag>
    """
    z = [0., 40., 60., 70., 75., 80.] * u.deg_C
    flux_ratio = np.array([1.0, 1.145, 1.23, 1.341, 1.407, 1.439])
    f = interp1d(z, flux_ratio)
    r = f(zenith_angle.to(u.deg))
    return -2.5 * np.log10(r) * u.mag


def Kastner_logL(zenith_angle, sun_alt, phi):
    """
    Kastner S. 1976
    The Journal of the Royal Astronomical Society of Canada
    Vol 70, No 4, p153

    Parameters
    ----------
    zenith_angle : `astropy.units.quantity.Quantity` deg
        the pointing zenith angle

    sun_alt :  `astropy.units.quantity.Quantity` deg
        the sun altitude (<0 if below the horizon)

    phi : `astropy.units.quantity.Quantity` deg
        the azimuth difference between the pointing and th sun

    Returns: float
        the logL

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import Kastner_logL
    >>> Kastner_logL(0 * u.deg, - 10 * u.deg, 45 * u.deg)
    -1.5822499999999997
    """
    h = -sun_alt.value  # convert to solar depression > 0
    z = zenith_angle.value
    theta = phi.value
    theta0 = -(4.12e-2 * z + 0.582) * h +\
        0.417 * h + 97.5  # eq 1c
    if theta <= theta0:
        logL = -(7.5e-5 * z + 5.05e-3) * theta +\
            (3.67e-4 * z - 0.458) * h + 9.17e-3 * z + 3.225  # eq 1a
    else:
        logL = -0.0010 * theta + (1.12e-3 * z - 0.47) * h -\
            4.17e-3 * z + 3.225  # eq 1b
    return logL


def Kastner_logL2(zenith_angle, sun_alt, phi):
    """
    Kastner note added in proof
    The Journal of the Royal Astronomical Society of Canada
    Vol 70, No 4, p153

    >>> import astropy.units as u
    >>> from moon import Kastner_logL2
    >>> Kastner_logL2(0 * u.deg, - 10 * u.deg, 45 * u.deg)
    -2.4341000000000004
    """
    z = zenith_angle.value
    logL90 = Kastner_logL(90 * u.deg, sun_alt, phi)
    logL30 = Kastner_logL(30 * u.deg, sun_alt, phi)
    return (z - 60.) / 30. * logL90 + (90. - z) / 30. * logL30


def twilight_lsst(filter, mdark, zenith_angle, sun_alt, phi, Rozenberg=False,
                  Paranal=True, grid=False, verbose=0):
    """
    Use the LSST twilight sky brightness empirical model by
    Yoachim P. et al.

    An Optical to IR Sky Brightness Model for the LSST

    Proc. of SPIE Vol. 9910 9910A-5

    Parameters
    ----------
    filter : str
        LSST filer B, G, R

    mdark : `astropy.units.quantity.Quantity` mag / arcsec**2
        the dark sky brightness

    zenith_angle : `astropy.units.quantity.Quantity` deg array of ndim x 1
        the zenith angle of the pointing(s)

    sun_lat : `astropy.units.quantity.Quantity` deg
        the Sun altitutde 0 means the horizon, negative means below the horizon

    phi : `astropy.units.quantity.Quantity` deg array of ndim
        the azimuth difference between the pointing(s) and the Sun

    Rozenberg : bool, default=False
        use the Rozenberg formula to compute the airmass from the zenith angle

    Paranal : bool, optional, default=True
        apply the LSST model to measure Paranal values
        Since the default is True, the result is geared toward the Paranal site

    verbose : int, optional, default = 0
        verbose level

    Returns
    -------
    : `astropy.units.quantity.Quantity` mag / arcsec**2
        the sky surface brightness due to the Sun + residual
        the Moon contribution is not taken into account

    Notes
    -----
    There are differences between Joachim and Patat

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import twilight_lsst
    >>> from moon import Paranal_sky_brightness
    >>> zenith_angle = 0.0 * u.deg
    >>> sun_alt = -25 * u.deg
    >>> phi = 120 * u.deg
    >>> mdark = Paranal_sky_brightness(110 * u.deg, 'B')
    >>> mtwi1 = twilight_lsst('B', mdark, zenith_angle, sun_alt, phi)
    >>> mtwi2 = Paranal_sky_brightness(90 * u.deg - sun_alt, 'B')
    >>> print(mtwi1, mtwi2)
    22.4 mag / arcsec2 22.4 mag / arcsec2
    >>> mdark = Paranal_sky_brightness(110 * u.deg, 'V')
    >>> mtwi1 = twilight_lsst('G', mdark, zenith_angle, sun_alt, phi)
    >>> mtwi2 = Paranal_sky_brightness(90 * u.deg - sun_alt, 'V')
    >>> print(mtwi1, mtwi2)
    21.641 mag / arcsec2 21.641 mag / arcsec2
    >>> az_sun_sep = [5., 10., 15., 35, 45, 65] * u.deg
    >>> out = twilight_lsst('B', mdark, [0., 5., 15., 20., 25., 45] * u.deg,
    ...                     -12 * u.deg, az_sun_sep)

    import matplotlib.pyplot as plt
    import numpy as np
    zenith_angle = 0.0 * u.deg
    sun_alt = np.arange(-11, -25, -0.1) * u.deg
    mtwi1, mtwi2, mtwi3  = [], [], []
    mdark = Paranal_sky_brightness(110 * u.deg, 'V')
    for alt in sun_alt:
        mtwi1.append(twilight_lsst('G', mdark, zenith_angle, alt, phi,
        Paranal=False).value)
        mtwi2.append(twilight_lsst('G', mdark, zenith_angle, alt, phi,
        Paranal=True).value)
    plt.plot(sun_alt, mtwi1, label='Yoachim')
    plt.plot(sun_alt, mtwi2, label='Paranal (Patat et al.)')
    plt.plot(sun_alt, mtwi2, label='Yoachim-Paranal')
    plt.xlabel('Sun alt (deg)')
    plt.ylabel('V band mag/arcsec$^{2}$')
    plt.title('Zenith')
    plt.legend()
    plt.show()

    """

    if sun_alt > -11 * u.deg:
        if verbose > 0:
            logging.info("LSST twilight works for sun alt below -11 deg")
        return 0.

    # Convert LSST filters to ESO filters
    filter_ESO = {'B': 'B', 'G': 'V', 'R': 'R'}

    param = dict()
    # Table 1 in Yoachim et al.
    # r12, a, b, c, mdark
    # a is the coeff with sun alt
    param['B'] = [8.42, 22.96, 0.29, 0.30, 22.35]
    # param['G'] = [4.14, 22.94, 0.30, 0.32, 21.71] . # original
    param['G'] = [4.14, 22.94, 0.30, 0.32, 21.78]
    param['R'] = [2.73, 22.20, 0.30, 0.33, 21.30]

    r12 = param[filter][0]
    a = param[filter][1]
    b = param[filter][2]
    c = param[filter][3]
    # mLSSTdark = param[filter][4]

    alpha = (sun_alt.to(u.deg) + 12 * u.deg).to(u.rad).value

    if Rozenberg:
        airmass = airmassRozenberg(zenith_angle).value
    else:
        airmass = airmassKS(zenith_angle).value
        # . airmass = (1 / np.cos(zenith_angle)).value

    airmass_limit = 5.
    if isinstance(airmass, np.ndarray):
        w = np.where(airmass > airmass_limit)[0]
    else:
        if airmass > airmass_limit:
            w = [0]
        else:
            w = []
    if len(w) > 0:
        if verbose > 0:
            logging.info("Airmass > %s Cannot proceed", airmass_limit)
            logging.info("Airmass %s", airmass)
            logging.info("Return NaN for the sky brightness")
        return np.nan

    if Paranal:
        # Replace the LSST values with the Paranal values
        mtwiz = Paranal_sky_brightness(90 * u.deg - sun_alt,
                                       filter_ESO[filter]).value
    else:
        # eq. 4
        mdark = mdark.value
        mtwiz = mdark - 2.5 * np.log(r12) - 2.5 * a * alpha

    if phi.isscalar:
        phi = [phi.value] * u.deg
    if not isinstance(airmass, np.ndarray):
        airmass = np.array([airmass])
    lp = len(phi)
    la = len(airmass)
    if la != lp:
        mssg = "Error: zenith angle and the phi have to be of the same size"
        logging.info(mssg)
        return 0 * u.mag / u.arcsec**2
    if grid:
        mtwi = np.empty((lp, la))
        for ia, am in enumerate(airmass):
            # Part 2 of eq. 4 : airmass (zenith angle) dependency
            maway = mtwiz - 2.5 * b * (am - 1.)
            for ip, p in enumerate(phi):
                # eq. 5 : azimuth pointing-Sun dependency
                if (-90 * u.deg < p < 90 * u.deg) and (am >= 1.1):
                    mtwi[ip, ia] = maway - 2.5 * (c * (am-1) * np.cos(p).value)
                else:
                    mtwi[ip, ia] = maway
        if (lp == 1) & (la == 1):
            mtwi = mtwi[0][0]

    else:
        if len(phi) != len(airmass):
            logging.info("Not equal length for the zenith angle and azimuth")
        maway = mtwiz - 2.5 * b * (airmass - 1.)
        wp = (-90 * u.deg < phi) & (phi < 90 * u.deg) & (airmass >= 1.1)
        mtwi = maway
        mtwi[wp] = maway[wp] - 2.5 * (c * (airmass[wp]-1) *
                                      np.cos(phi[wp]).value)

        if (lp == 1) & (la == 1):
            mtwi = mtwi[0]

    if Paranal:
        return mtwi * u.mag / u.arcsec**2
    else:
        return add_magnitudes(mdark, mtwi) / u.arcsec**2


def moon_diameter(r, Lamy=False):
    """
    Compute the Moon apparent diameter in arsec

    The Moon's apparent diameter in arcsec is:
    d = 1873.7" * 60 / r
    where r is the Moon's distance in Earth radii.
    about 31 arcmin or 1/2 deg

    Ref. http://stjarnhimlen.se/comp/ppcomp.html#15

    Parameter
    ---------
    r : float
        the Moon's distance in Earth radii

    Lamy : Boolean, optional, default = False
        if True, use the value of 1919.98 instead of 1873,7
        Ref. Lamy et al. 2015 Solar Physics, 290, 2617

    Return
    ------
     : `astropy.units.quantity.Quantity` arcsec
     The Moon's apparent diameter in arcsec

    Example
    -------
    >>> import astropy.units as u
    >>> from moon import moon_diameter
    >>> d = 384400.  # km
    >>> Eradius = 6371.   # km
    >>> out = moon_diameter(d / Eradius).to(u.deg)
    >>> # <Quantity 1863.26889178 arcsec>
    >>> # <Quantity 0.51757469 deg>
    >>> moon_diameter(d / Eradius, Lamy=True)
    <Quantity 1909.29124558 arcsec>
    """
    if Lamy:
        return (1919.98 * 60 / r) * u.arcsec
    else:
        return (1873.7 * 60 / r) * u.arcsec


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
