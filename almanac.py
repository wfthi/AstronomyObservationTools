#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
Use astropy routines to compute an almanac for a night observations

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
from datetime import datetime
from datetime import timedelta
import logging
# third-party
import numpy as np
import astropy
import astropy.units as u
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun
from moon import moon_illumination
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body
from timezonefinder import TimezoneFinder  # type: ignore
import pytz
# local
from classUtils import classUtils


class almanac(classUtils):

    """
    See Astropy & SkyCalc

    This class deals with time-dependent and location-dependent parameters

    The main way to create an almanac is to use make_night_almanac_date
    given the telescope Earthlocation and the date when the observation
    starts.

    Disclaimer
    ----------
    For offical use, one should use the data provided by the Astronomical
    Applications Department of the U.S. Naval Observatory
        https://aa.usno.navy.mil

    Notes
    -----
    The code requires the classUtils class

    Sunset/sunrise USNO returns the time when the solar disk center is
    at -0.8333 degrees altitude to account for the solar radius and atmospheric
    refraction.

    Civil twilight: time at evening civil (-6 degree) twilight
    Nautical twilight: time at evening nautical (-12 degree) twilight
    Astronomical twilight: time at evening astronomical (-18 degree) twilight

    As an example, we will use the ESO-VISTA telescope.
    VISTA is located at lat=-24.615833 deg, lon=-70.3975 deg, height=2518 m
    24° 36' 57" S, 70° 23' 51" W
    Sutherland et al. A&A 575, A25 (2015) DOI: 10.1051/0004-6361/201424973
    The code will find determine the time zone.

    The convention is  Latitude (+N) and Longitude (+E)

    Tests would be performed against the JPL's HORIZONS system
    """
    iso = '%Y-%m-%dT%H:%M:%S'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # delegate the init to classUtils

    @classmethod
    def make_night_almanac_date(cls, telescope, date, min_sunalt,
                                step=1 * u.min, extra=1 * u.hr):
        """
        Make an almanac for the day date ('2024-06-01') starting
        from when the sun is below min_sunalt - extra till the next
        sun set above min_sunalt + extra. The code will compute the
        time for the sun setting and rising below/above a minimum
        altitude min_sunalt (with a negative value bellow the horizon).

        Use for min_sunalt;
        * -18 * u.deg for Astronomical twilight
        * -6 * u.deg for Civilian twilight
        * -0.8333 * u.deg for the standard sunset/sunrise

        Notes
        -----
        The Sun rise is set at the horizon (Sun altitude= 0 degree)
        the solar disk center is at -0.8333 degrees altitude to account
        for the solar radius and atmospheric refraction (USNO)
        horizon=-0.8333 * u.deg is set by default

        Parameter
        ---------
        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        step : `astropy.units.quantity.Quantity` in time units (s, min, hr)
            the time step

        extra : `astropy.units.quantity.Quantity` in time units (s, min, hr)
                optional. default = 1 hr
            the extra time around the twilight where the almanac has
            to be computed

        Returns
        -------
            an almanac class object

        Notes
        -----
        The almanac object can be saved in a file of different formats.
        1) Fits: the alamanac object can be converted into an Astropy
           Table, which in turn can be saved as a fits file
        2) ecsv file: the alamanac object can be saved directly as an
           extended csv file
        3) pickle file: the almanac object can be converted into a python
           dictionnary, which can be saved as a pickle file
        4) as a standard csv: >>> al.to_csv('al.csv'). The csv file can be
           read by spreadsheet readers

        The times are saved in an ascii form and not in a high precision
        astropy Time format. The Julian Date is in the modified form MJD.

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.coordinates import EarthLocation
        >>> from almanac import *
        >>> from astropy.io import ascii
        >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
                                      lon=-70.3975 * u.deg,
                                      height=2518 * u.m)
        >>> date = '2024-06-01'
        >>> min_sunalt = -18 * u.deg
        >>> al = almanac.make_night_almanac_date(telescope, date,
                                                 min_sunalt,
                                                 step=1 * u.min,
                                                 extra=1 * u.hr)
        >>> tal = al.to_table()  # convert the object into an astropy table
        >>> tal.write('tal.fits',format='fits', overwrite=True)
        >>> # or
        >>> ascii.write(tal, 'test_almanac.ecsv',
                        format='ecsv', overwrite=True)
        >>> # Read the saved almanac from a fits file
        >>> from astropy.table import QTable
        >>> tab = QTable.read('tal.fits')
        >>> al_tab = almanac.from_table(tab)
        >>> al == al_tab  # compare the two objects
        True
        """
        object = 'sun'
        tevening = \
            find_time_at_elevation(object, telescope, date,
                                   min_sunalt,
                                   'set', which='next')
        tmorning = \
            find_time_at_elevation(object, telescope, date,
                                   min_sunalt,
                                   'rise', which='next')
        tstart = Time(tevening) - extra
        tend = Time(tmorning) + extra
        al = almanac.make_night_almanac(telescope,
                                        tstart, tend, step=step)
        return al

    @classmethod
    def make_night_almanac(cls, telescope, tstart, tend,
                           step=1 * u.min):
        """
        Create the night ephemris/alamanac given the start and
        end time in utc. It creates the time grid.

        Parameter
        ---------
        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        tstart: `astropy.time.Time` object
            the start of the almanac

        tend: `astropy.time.Time`
            the end of the almanac

        step : `astropy.units.quantity.Quantity` in time units (s, min, hr)
            the time step

        Returns
        -------
            an almanac class object

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.coordinates import EarthLocation
        >>> from almanac import *
        >>> from astropy.io import ascii
        >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
                                      lon=-70.3975 * u.deg,
                                      height=2518 * u.m)
        >>> start_utc = '2021-01-01T23:00:00'
        >>> end_utc = '2021-01-02T14:00:00'
        >>> al = almanac.make_night_almanac(telescope,
                                            date_utc=day_utc,
                                            step=1 * u.hr,
                                            verbose=True)
        >>> tal = al.to_table()  # convert the object into an astropy table
        >>> tal.write('tal.fits',format='fits', overwrite=True)
        >>> # or
        >>> al.save('test_almanac.fits',format='fits', overwrite=True)
        >>> # or
        >>> ascii.write(tal, 'test_almanac.ecsv',
                        format='ecsv', overwrite=True)
        >>> # Read the saved almanac from a fits file
        >>> from astropy.table import QTable
        >>> tab = QTable.read('tal.fits')
        >>> al_tab = almanac.from_table(tab)
        >>> al == al_tab  # compare the two objects
        True
        """
        if isinstance(tstart, str):
            tstart = Time(tstart)
        if isinstance(tend, str):
            tend = Time(tend)
        if not tstart < tend:
            logging.error("The end time must be aftet the start time")
            return None
        obstime = create_obsTime(tstart, tend, step)
        return cls.make_almanac(telescope, utc_iso=obstime.isot)

    @classmethod
    def make_almanac(cls, telescope, utc_iso='2021-01-01T14:00:00'):
        """
        Make an almanac based on the astropy built-in ephemeris.

        Parameter
        ---------
        telescope: `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        utc_iso : str or list of str
            the time or list of times where the almanac has to be computed

        Returns
        -------
            an almanac class object

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.coordinates import EarthLocation
        >>> from almanac import almanac
        >>> from astropy.time import Time
        >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
                                      lon=-70.3975 * u.deg,
                                      height=2518 * u.m)
        >>> time_utc = Time('2459216.6', format='jd')
        >>> utc_iso = Time(time_utc).isot
        >>> al = almanac.make_almanac(telescope,
                                      utc_iso=utc_iso)
        TODO
        ----
        Provide an option to use the more accurate JPL ephemeris
        """
        al = cls()
        al.utc_iso = utc_iso
        al.utc_MJD = Time(utc_iso).jd - 2450000.0  # JD - 2450000
        al.moon_position(telescope)
        al.moon_FLI()
        al.sun_position(telescope)
        al.planet_position(telescope, 'venus')
        al.planet_position(telescope, 'mars')
        al.planet_position(telescope, 'jupiter')
        al.planet_position(telescope, 'saturn')
        al.lmst = almanac.utc_to_lst(utc_iso, telescope, 'mean')
        al.local_time = almanac.utc_to_local(utc_iso, telescope)
        return al

    def moon_position(self, telescope):
        """
        Compute the Moon cooridnates (ra, dec, alt, az) at obstime

        Parameter
        ---------
        telescope : `astropy.coordinates.earth.EarthLocation`
            The location of the telescope on the Earth.

        Example
        -------
        >>> from almanac import *
        >>> from astropy.coordinates import EarthLocation
        >>> telescope_location = EarthLocation(lat=-24.615833 * u.deg,
                                       lon=-70.3975 * u.deg,
                                       height=2518 * u.m)
        >>> al = almanac()
        """
        obstime = Time(self.utc_iso)
        moon_coord = get_body("moon", obstime)
        self.moon_ra = moon_coord.icrs.ra  # ICRS frame
        self.moon_dec = moon_coord.icrs.dec  # ICRS frame

        altaz_frame = AltAz(obstime=obstime, location=telescope)
        altaz = moon_coord.transform_to(altaz_frame)
        self.moon_alt = altaz.alt
        self.moon_az = altaz.az

    def moon_FLI(self):
        """
        Compute the Moon illumnination
        """
        self.FLI = moon_illumination(Time(self.utc_iso))

    def sun_position(self, telescope):
        """
        Compute the Sun cooridnates (ra, dec, alt, az)

        Parameter
        ---------
        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.
        """
        obstime = Time(self.utc_iso)
        sun_coord = get_sun(obstime)
        self.sun_ra = sun_coord.ra
        self.sun_dec = sun_coord.dec

        altaz_frame = AltAz(obstime=obstime, location=telescope)
        altaz = sun_coord.transform_to(altaz_frame)
        self.sun_alt = altaz.alt
        self.sun_az = altaz.az

    def planet_position(self, telescope, body):
        """
        Compute the planet cooridnates (ra, dec, alt, az)

        Parameter
        ---------
        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        body : string
            the name of the planet
            body = 'jupiter', 'mars', 'venus', 'saturn

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.coordinates import EarthLocation
        >>> from almanac import almanac
        >>> from astropy.time import Time
        >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
                                      lon=-70.3975 * u.deg,
                                      height=2518 * u.m)
        >>> time_utc = Time('2459216.6', format='jd')
        >>> utc_iso = Time(time_utc).isot
        >>> al = almanac(utc_iso=utc_iso)
        >>> al.planet_position(telescope, 'venus')
        >>> al.planet_position(telescope, 'mars')
        >>> al.planet_position(telescope, 'jupiter')
        >>> al.planet_position(telescope, 'saturn')
        """
        accepted_body = ['jupiter', 'mars', 'venus', 'saturn']
        if body not in accepted_body:
            logging.info(body + " is not a valid planet")
            return
        obstime = Time(self.utc_iso)
        # use built-in astropy ephemeris
        with solar_system_ephemeris.set('builtin'):
            body_pos = get_body(body, obstime, telescope)
        altaz_frame = AltAz(obstime=obstime, location=telescope)
        altaz = body_pos.transform_to(altaz_frame)
        if body == 'jupiter':
            self.jupiter_ra = body_pos.ra
            self.jupiter_dec = body_pos.dec
            self.jupiter_alt = altaz.alt
            self.jupiter_az = altaz.az
            return
        elif body == 'saturn':
            self.saturn_ra = body_pos.ra
            self.saturn_dec = body_pos.dec
            self.saturn_alt = altaz.alt
            self.saturn_az = altaz.az
            return
        elif body == 'mars':
            self.mars_ra = body_pos.ra
            self.mars_dec = body_pos.dec
            self.mars_alt = altaz.alt
            self.mars_az = altaz.az
            return
        elif body == 'venus':
            self.venus_ra = body_pos.ra
            self.venus_dec = body_pos.dec
            self.venus_alt = altaz.alt
            self.venus_az = altaz.az
            return

    @staticmethod
    def local_to_utc(localtime, telescope):
        """
        Find the UTC time from the local time at the telescope

        Parameters
        ----------
        localtime : str
            local time in theISO 8601 format
            example '2020-02-08T09:30:26.123'

        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        Returns
        -------
        : str
            UTC time in isot format

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.time import Time
        >>> from astropy.coordinates import EarthLocation
        >>> from almanac import almanac
        >>> localtime = '2020-02-08T09:30:26.123'
        >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
        ...                           lon=-70.3975 * u.deg,
        ...                           height=2518. * u.m)
        >>> almanac.local_to_utc(localtime, telescope)
        '2020-02-08T12:30:26.123'
        """
        # local time in the datetime format
        localdt = datetime.fromisoformat(localtime)
        # find the timezone at the telescope at that time
        location = TimezoneFinder()
        telescope_timezone = location.timezone_at(lng=telescope.lon.value,
                                                  lat=telescope.lat.value)
        # use pytz to get a DstTzInfo object
        timezone = pytz.timezone(telescope_timezone)
        localdt = timezone.localize(localdt)
        time_utc = Time(localdt.astimezone(pytz.UTC))
        time_utc.format = 'isot'
        return time_utc.value

    @staticmethod
    def utc_to_lst(time_utc, telescope, option, model=None):
        """
        Get the local sideral time given the UTC and the location
        (staticmethod)

        Notes
        -----
        The C code from pyerfa is  gmst82.c

        Parameter
        ---------
        time_utc : str
            the Julian Date of the UTC in iso format
            of the day
            %Y-%m-%dT%H:%M:%S (2021-01-01T12:00:00)

        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        option : str
            'mean' to compute the local mean sideral time
            'apparent' to compute the local apparent sideral time

        model : str, optional, default=None
            the precssion model to be used for mean sideral time
            the precession and nutation model to be used for the apparent
            sideral time

        Returns
        -------
        lst : `astropy.units.quantity.Quantity` degree
            the local sideral time at UTC either the
            mean of the apparent (lmst or last)

        Notes
        -----
        It is a wrapper of an astropy Time mehod, which in turn is a wrapper
        of an ERFA/SOFA routine.

        https://gssc.esa.int/navipedia/index.php/Coordinate_Systems

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.coordinates import EarthLocation
        >>> vista = EarthLocation(lat=-24.615833 * u.deg,
        ...                       lon=-70.3975 * u.deg,
        ...                       height=2518 * u.m)
        >>> from almanac import almanac
        >>> time_utc = '2020-01-01T12:00:00'
        >>> almanac.utc_to_lst(time_utc, vista, 'mean')
        <Longitude 210.2163921 deg>
        """
        obstime = Time(time_utc)
        lst = obstime.sidereal_time(option, model=model,
                                    longitude=telescope.lon)
        return lst.to(u.degree)

    @staticmethod
    def utc_to_local(time_utc, telescope):
        """
        Given the location (telescope) and the utc time, compute the local time

        Notes
        -----
        The code uses TimezoneFinder to find the time zone at the location
        of the telescope.

        Parameter
        ---------
        time_utc : str or Time Quantity
            str should be recognizable by the astropy Time routine

        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        Returns
        -------
            local_time_iso : datetime object
            The local time in the '%Y-%m-%dT%H:%M:%S' format

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.coordinates import EarthLocation
        >>> from almanac import almanac
        >>> vista = EarthLocation(lat=-24.615833 * u.deg,
        ...                       lon=-70.3975 * u.deg,
        ...                       height=2518 * u.m)
        >>> time_utc = '2020-01-01T12:00:00'
        >>> almanac.utc_to_local(time_utc, vista)
        '2020-01-01T09:00:00'
        """
        if not isinstance(time_utc, astropy.time.core.Time):
            time_utc = Time(time_utc)

        iso = almanac.iso
        # Use TimezoneFinder to get the time zone at the telescope site
        location = TimezoneFinder()
        telescope_timezone = location.timezone_at(lng=telescope.lon.value,
                                                  lat=telescope.lat.value)
        # use pytz to get a DstTzInfo object
        timezone = pytz.timezone(telescope_timezone)
        local_time = time_utc.to_datetime(timezone=timezone)
        # output in the iso format
        if not time_utc.isscalar:
            local_time_iso = [t.strftime(iso) for t in local_time]
        else:
            local_time_iso = local_time.strftime(iso)
        return local_time_iso

    @staticmethod
    def lst_to_utc(time_utc, lst, telescope, option):
        """
        Compute the UTC at LST for day

        http://aa.usno.navy.mil/faq/docs/GAST.php
        Astronomical Phenomena for the year 2020
        The Nautical Almanac Office and USNO

        http://astro.dur.ac.uk/~ams/users/lst.html
        https://rdrr.io/cran/astroFns/src/R/ut2lst.R

        Parameter
        ---------
        time_utc : str
            the Julian Date of the UTC in isot format
            of the day
            %Y-%m-%dT%H:%M:%S (2021-01-01T12:00:00)

        lst : `astropy.units.quantity.Quantity` degree
            lmst or last

        telescope : `astropy.coordinates.earth.EarthLocation`
            The location on the Earth.

        option : str
        'mean' for local mean sideral time lmst
        'apparent' for local aparent sideral time last
        The option has to be coherent with the definition of
        lst (lmst or last)

        Notes
        -----
        GMST_1h is the Greenwich mean sieral time at UTC ='2020-01-01T01:00:00'
        GMST_0h is the Greenwich mean sieral time at UTC ='2020-01-01T00:00:00'
        The constant in the code is GMST_1h - GMST_0h = 15.04106873377593
        degrees

        A good approximation of the ratio of universal to sidereal times is
        1.002737909350795 + 5.900610-11 Tu - 5.9e-15 Tu^2 .

        Example
        -------
        >>> import astropy.units as u
        >>> from astropy.time import Time
        >>> from almanac import almanac
        >>> from astropy.coordinates import EarthLocation
        >>> from numpy import isclose
        >>> greenwich = EarthLocation(lat=51.4778 * u.deg,
        ...                           lon=0 * u.deg,
        ...                           height=0 * u.m)
        >>> time_utc0 = '2020-01-01T12:00:00'
        >>> lst0 = (6.6091 + 0.06571) * 15 * u.deg
        >>> time_utc = almanac.lst_to_utc(time_utc0, lst0, greenwich, 'mean')
        >>> # the time should be close to 00:00:00 within 1 second
        >>> lst = almanac.utc_to_lst(time_utc, greenwich, 'mean')
        >>> isclose(lst, lst0)
        >>> t1 = Time(time_utc).jd
        >>> t2 = Time('2020-01-01T00:00:00').jd
        >>> isclose(t1, t2)
        """
        utc = Time(time_utc, format='isot')  # UT of the day at any time
        utc0 = Time(np.ceil(utc.jd) - 0.5, format='jd')  # UT at 0 hr
        # lst at UT 0 hr
        lst0 = almanac.utc_to_lst(utc0, telescope, option)
        # frac of a day in UT since lst0
        day_frac = TimeDelta((lst - lst0).value / (15.04106873377593 * 24),
                             format='jd')
        return Time(utc0 + day_frac, format='jd').iso

    @staticmethod
    def nutate_iau1980(time_utc):
        """
        Computes the Earth's nutation in longitude and obliquity for a
        given (array) of Julian date consistent with the IAU 1980 nutation
        theory. Correction factors have been added later on.

        Nutation describes the motion of the true pole relative to the mean
        pole and may be resolved into the components delta_phi in longitude
        and delta_epsilon in obliquity.

        The second timescale for motion of the Earth’s axis is due to nutation.
        This correction to precession compensates for second-order torques by
        the Sun, the Moon, and other planets. The principle term is due to the
        Moon, whose orbital plane precesses on a timescale of 18.6 years.
        The amplitude of nutation (i.e., the constant of nutation) is
        N = 9.210􏰀.

        Parameters
        ----------
        time_utc : str
            the Julian Date of the UTC in iso format
            of the day
            %Y-%m-%dT%H:%M:%S (2021-01-01T12:00:00)

        Returns
        -------
        nut_lon : `astropy.units.quantity.Quantity` degree
            The nutation in longitude (in deg).
        nut_obliq : `astropy.units.quantity.Quantity` degree
            The nutation in latitude (in deg).

        Notes
        -----
        Adapted from the IDL code by W.Landsman (Goddard/HSTX)
        Uses the formula in Chapter 22 of "Astronomical Algorithms" by Jean
        Meeus (1998, 2nd ed.) which is based on the 1980 IAU Theory of
        Nutation and includes all terms larger than 0003".

        It is mainly used as a comparison code with astropy nuation models.

        Examples
        --------
        Find the nutation in longitude and obliquity 1987 on Apr 10 at Oh.
        This is example 22.a from Meeus
        >>> import astropy.units as u
        >>> from astropy.time import Time
        >>> from PyAstronomy import pyasl
        >>> from almanac import almanac
        >>> time_utc = '1987-04-10T00:00:00'
        >>> nut_long, nut_obliq = almanac.nutate_iau1980(time_utc)
        >>> print(nut_long.to(u.arcsec))
        -3.787931077110904 arcsec
        >>> print(nut_obliq.to(u.arcsec))
        9.442522632175837 arcsec
        >>> # ==> nut_long = -3.788    nut_obliq = 9.443
        >>> jd = Time(time_utc).jd
        >>> pnutate = pyasl.nutate(jd)
        >>> print(nut_long.value, pnutate[0][0])
        -0.001052203076975251 -0.0010522030769751856
        >>> print(nut_obliq.value, pnutate[1][0])
        0.002622922953382177 0.0026229224162901237
        >>> time_utc = ['1999-12-10T00:00:00', '2021-04-10T00:00:00']
        >>> nut_long, nut_obliq = almanac.nutate_iau1980(time_utc)
        >>> nut_long
        <Quantity [-0.00407668, -0.00471902] deg>
        >>> nut_obliq
        <Quantity [-0.00167534,  0.00088904] deg>
        """
        mod = 2 * np.pi * u.rad
        # Form time in Julian centuries from 1900.0
        jdcen = almanac.julianCenturies(time_utc)
        if isinstance(jdcen, float):
            jdcen = np.array([jdcen])

        # Mean elongation of the Moon from the Sun
        coef_moon = [1.0 / 189474.0, -0.0019142, 445267.111480, 297.85036]
        d = (np.polyval(coef_moon, jdcen) * u.deg).to(u.rad) % mod

        # Sun's mean anomaly
        coef_sun = [-1.0 / 3e5, -0.0001603, 35999.050340, 357.52772]
        sun = (np.polyval(coef_sun, jdcen) * u.deg).to(u.rad) % mod

        # Moon's mean anomaly
        coef_mano = [1.0 / 5.625e4, 0.0086972, 477198.867398, 134.96298]
        mano = (np.polyval(coef_mano, jdcen) * u.deg).to(u.rad) % mod

        # Moon's argument of latitude
        coef_mlat = [-1.0 / 3.27270e5, -0.0036825, 483202.017538, 93.27191]
        mlat = (np.polyval(coef_mlat, jdcen) * u.deg).to(u.rad) % mod

        # Longitude of the ascending node of the Moon's mean orbit
        #  on the ecliptic, measured from the mean equinox of the date
        coef_moe = [1.0 / 4.5e5, 0.0020708, -1934.136261, 125.04452]
        omega = (np.polyval(coef_moe, jdcen) * u.deg).to(u.rad) % mod

        d_lng = np.array([0, -2, 0, 0, 0, 0, -2, 0, 0, -2, -2, -2, 0, 2, 0, 2,
                          0, 0, -2, 0, 2, 0, 0, -2, 0, -2, 0, 0, 2, -2, 0, -2,
                          0, 0, 2, 2, 0, -2, 0, 2, 2, -2, -2, 2, 2, 0, -2, -2,
                          0, -2, -2, 0, -1, -2, 1, 0, 0, -1, 0, 0, 2, 0, 2],
                         float)

        m_lng = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, -1, 0,
                          0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0,
                          1, 0, 0, 1, 0, 0, 0, -1, 1, -1, -1, 0, -1], float)

        mp_lng = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, -1, 0, 1, -1,
                           -1, 1, 2, -2, 0, 2, 2, 1, 0, 0, -1, 0, -1, 0, 0,
                           1, 0, 2, -1, 1, 0, 1, 0, 0, 1, 2, 1, -2, 0, 1, 0,
                           0, 2, 2, 0, 1, 1, 0, 0, 1, -2, 1, 1, 1, -1, 3, 0],
                          float)

        f_lng = np.array([0, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2,
                          0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 0, 0, 0, -2, 2,
                          2, 2, 0, 2, 2, 0, 2, 2, 0, 0, 0, 2, 0, 2, 0, 2, -2,
                          0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2], float)

        om_lng = np.array([1, 2, 2, 2, 0, 0, 2, 1, 2, 2, 0, 1, 2, 0, 1, 2, 1,
                           1, 0, 1, 2, 2, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 0,
                           1, 2, 2, 0, 2, 1, 0, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 2], float)

        sin_lng = np.array([-171996, -13187, -2274, 2062, 1426, 712, -517,
                            -386, -301, 217, -158, 129, 123, 63, 63, -59,
                            -58, -51, 48, 46, -38, -31, 29, 29, 26, -22,
                            21, 17, 16, -16, -15, -13, -12, 11, -10, -8, 7,
                            -7, -7, -7, 6, 6, 6, -6, -6, 5, -5, -5, -5, 4,
                            4, 4, -4, -4, -4, 3, -3, -3, -3, -3, -3, -3, -3],
                           float)

        sdelt = np.array([-174.2, -1.6, -0.2, 0.2, -3.4, 0.1, 1.2, -0.4, 0.,
                          -0.5, 0, 0.1, 0, 0, 0.1, 0, -0.1, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, -0.1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0], float)

        cos_lng = np.array([92025, 5736, 977, -895, 54, -7, 224, 200, 129,
                            -95, 0, -70, -53, 0, -33, 26, 32, 27, 0, -24, 16,
                            13, 0, -12, 0, 0, -10, 0, -8, 7, 9, 7, 6, 0, 5, 3,
                            -3, 0, 3, 3, 0, -3, -3, 3, 3, 0, 3, 3, 3, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)

        cdelt = np.array([8.9, -3.1, -0.5, 0.5, -0.1, 0.0, -0.6, 0.0, -0.1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0.], float)

        # Sum the periodic terms
        n = jdcen.size
        nut_lon = np.zeros(n)
        nut_obliq = np.zeros(n)
        arg = np.outer(d_lng, d) + np.outer(m_lng, sun) +\
            np.outer(mp_lng, mano) +\
            np.outer(f_lng, mlat) + np.outer(om_lng, omega)
        arg = np.transpose(arg)
        sarg = np.sin(arg)
        carg = np.cos(arg)
        for i in range(n):
            nut_lon[i] = 0.0001 * np.sum((sdelt * jdcen[i] +
                                          sin_lng) * sarg[i])
            nut_obliq[i] = 0.0001 * np.sum((cdelt * jdcen[i] +
                                            cos_lng) * carg[i])

        #  Until here result are in arcseconds!
        #  Convert to degrees
        nut_lon *= u.arcsec
        nut_obliq *= u.arcsec
        if nut_lon.size == 1:
            return nut_lon.to(u.deg)[0], nut_obliq.to(u.deg)[0]
        else:
            return nut_lon.to(u.deg), nut_obliq.to(u.deg)

    @staticmethod
    def julianCenturies(date, format=iso):
        """
        Compute the Julian centuries from J2000 of the date in different
        input format and units

        Reference
        ---------
        [1] Meeus, Jean (1998) Astronomical Algorithms (2nd Edition).
            Willmann-Bell, Virginia

        Parameter
        ---------
        date : multiple entry types are valid
            1. str, numpy str_ : time in iso format "%Y-%m-%dT%H:%M:%S"
            2. a list or numpy array of strings
            3. astropy Time object
            4. float of an array (Numpy array or list) of float
               the Julian dates
            5. a Python datetime

        format : str, optional, default='Y-%m-%dT%H:%M:%S'
            if optional 1 is chosen, one can enter the format of the
            date-time string

        Returns
        -------
        jdcen : Numpy array of floats
            the Julian century from J2000
            With a single entry the output will be a single entry array

        Example
        -------
        >>> from almanac import almanac
        >>> from astropy.time import Time
        >>> from datetime import datetime
        >>> jd2000 = 2451545.0
        >>> almanac.julianCenturies(jd2000)
        0.0
        >>> jd = 2451545.0 + 365
        >>> almanac.julianCenturies(jd)
        0.00999315537303217
        >>> jd = [2451545.0 - 365, 2451545.0 + 365]
        >>> almanac.julianCenturies(jd)
        array([-0.00999316,  0.00999316])
        >>> utc = ['2021-01-01T00:00:00', '1991-01-01T00:00:00']
        >>> almanac.julianCenturies(utc)
        array([ 0.21000684, -0.09000684])
        >>> utc = Time(['2021-01-01T00:00:00', '1991-01-01T00:00:00'])
        >>> almanac.julianCenturies(utc)
        array([ 0.21000684, -0.09000684])
        >>> utc = datetime.strptime("1/1/1991 00:00:00", "%d/%m/%Y %H:%M:%S")
        >>> almanac.julianCenturies(utc, format="%d/%m/%Y %H:%M:%S")
        -0.09000684462696783
        >>> utc = datetime.strptime("1991-01-01T00:00:00",
        ...                         "%Y-%m-%dT%H:%M:%S")
        >>> almanac.julianCenturies(utc)
        -0.09000684462696783
        """
        jd2000 = 2451545.0
        hundredyears = 36525.0

        if isinstance(date, (float, str, np.str_, datetime)):
            date = np.array([date])
        elif isinstance(date, (list, tuple)):
            date = np.array(date)
        elif isinstance(date, Time):
            djd = (date.jd1 - jd2000) + date.jd2
            jdcen = djd / hundredyears  # Meeus eq 12.1
            return jdcen

        jd1 = np.empty(date.size)
        jd2 = np.empty(date.size)

        for i, t in enumerate(date):
            if isinstance(t, float):
                jdcen = (date - jd2000) / hundredyears
                if jdcen.size == 1:
                    return jdcen[0]
                else:
                    return jdcen
            if isinstance(t, (str, np.str_)):
                t1 = datetime.strptime(t, format)
            else:
                t1 = t
            # UTC at 0 hr in Julian day
            jd1[i] = Time(t1.date().isoformat()).jd
            # jd2 is the fraction of a day such that jd = jd1 + jd2
            jd2[i] = (t1 - datetime(t1.year, t1.month, t1.day)) / timedelta(1)
        djd = (jd1 - jd2000) + jd2
        jdcen = djd / hundredyears
        if jdcen.size == 1:
            return jdcen[0]
        else:
            return jdcen

    @staticmethod
    def obliqEcliptic(time_utc, d_eps, full_terms=False):
        """
        (True) obliquity of ecliptic

        The obliquity of the ecliptic, or inclination of the Earth's
        axis of rotation, is the angle between the equator and the ecliptic

        The mean obliquity has no nutation correction applied

        Parameter
        ---------
        time_utc: `astropy.time.core.Time`
            the observing time in isot format.

        d_eps : `astropy.units.quantity.Quantity` Angle (degrees)
            the nutation in latitude
            if d_eps is non 0, the true obliquity of ecliptic is return

        full_terms : bool, optional, default = False
            if False use the 3 degree polynomial from Meeus [1]
            if True use the 10 degree polynomual from SPA [2] eq. 24

        Returns
        -------
        eps : `astropy.units.quantity.Quantity` in degrees
            the (true) obliquity of the ecliptic

        References
        ----------
        [1] Meeus Astronomical Algorithms Hardcover – 31 Dec. 1998
            Willmann-Bell, Inc.

        [2] Ibrahim Reda and Afshin Andreas
            Solar Position Algorithm for Solar Radiation Applications
            National Renewable Energy Laboratory, Technicl Report
            560-34302
            https://www.nrel.gov/docs/fy08osti/34302.pdf

        Example
        -------
        >>> from almanac import almanac
        >>> time_utc = '1987-04-10T00:00:00'
        >>> nut_long, nut_obliq = almanac.nutate_iau1980(time_utc)
        >>> almanac.obliqEcliptic(time_utc, nut_obliq)
        <Quantity 23.4435694 deg>
        >>> almanac.obliqEcliptic(time_utc, nut_obliq, full_terms=True)
        <Quantity 23.44356921 deg>
        >>> time_utc = ['1999-12-10T00:00:00', '2021-04-10T00:00:00']
        >>> nut_long, nut_obliq = almanac.nutate_iau1980(time_utc)
        >>> almanac.obliqEcliptic(time_utc, nut_obliq, full_terms=True)
        <Quantity [23.43762378, 23.43741428] deg>
        """
        jdcen = almanac.julianCenturies(time_utc)
        d_eps = d_eps.to(u.rad)

        # the obliquity of the ecliptic
        # Eq 22.2 from Meeus consisten with IAU
        # 23.4392911 is in degrees
        # * 3600 is the term to convert to arcsec
        coeff = np.array([0.001813, - 0.00059, - 46.8150, 23.4392911 * 3600])
        eps0 = np.polyval(coeff, jdcen) * u.arcsec

        # SPA eq. 24, 11 terms expansion
        # use Horner to limit summing terms with different ranges
        if full_terms:
            coeff = np.array([2.45, 5.79, 27.8, 7.12, -39.05, -249.67,
                              -51.38 + 1999.25, -1.55, -4680.93, 84381.448])
            x = 1e-2 * jdcen
            eps0 = np.polyval(coeff, x) * u.arcsec

        # True obliquity of the ecliptic
        # Apply the nutation correction
        # Old way to apply the equation of the equinox
        eps = (eps0 + d_eps.to(u.arcsec)).to(u.deg)

        return eps  # in degrees

    @staticmethod
    def eccen_earth(time_utc):
        """
        Earth's orbital eccentricity

        Parameter
        ---------
        time_utc: str or `astropy.time.core.Time`
            the observing time in isot format.

        Returns
        -------
        e : float
            The Earth's orbital eccentricity

        Notes
        -----
        Only good precsion for Epoch difference up to 100 years

        Example
        -------
        >>> from almanac import almanac
        >>> time_utc = '1987-04-10T00:00:00'
        >>> almanac.eccen_earth(time_utc)
        0.01671398
        >>> time_utc = ['1999-12-10T00:00:00', '2021-04-10T00:00:00']
        >>> almanac.eccen_earth(time_utc)
        array([0.01670866, 0.01669969])
        """
        jdcen = almanac.julianCenturies(time_utc)
        coeff = np.array([-0.0000001267, -0.000042037, 0.016708634])
        e = np.polyval(coeff, jdcen)
        return e

    @staticmethod
    def lon_perihelion(time_utc):
        """
        Longitude of the perihelion in degrees

        References
        ----------
        [1] Meeus, Jean (1998) Astronomical Algorithms (2nd Edition).
        Willmann-Bell, Virginia.

        Parameter
        ---------
        time_utc: str or `astropy.time.core.Time`
            the observing time in isot format.

        Returns
        -------
        pi : `astropy.units.quantity.Quantity` Angle
            The longitude of the perihelion in degrees

        Example
        -------
        >>> from almanac import almanac
        >>> time_utc = '1987-04-10T00:00:00'
        >>> almanac.lon_perihelion(time_utc)
        <Quantity 102.71847643 deg>
        >>> time_utc = ['1999-12-10T00:00:00', '2021-04-10T00:00:00']
        >>> almanac.lon_perihelion(time_utc)
        <Quantity [102.93629078, 103.30312973] deg>
        """
        jdcen = almanac.julianCenturies(time_utc)
        coeff = np.array([0.00046, 1.71946, 102.93735])
        pi = np.polyval(coeff, jdcen) * u.deg
        return pi

    @staticmethod
    def precession_coeff(time_utc0, time_utc):
        """
        Compute the precession coefficients for FK5
        from epoch0 (time_utc0) to epoch
        (time_utc)

        Meeus p. 134

        Parameter
        ---------
        time_utc0: str or `astropy.time.core.Time`
            the starting epoch

        time_utc: str or `astropy.time.core.Time`
            the final epoch

        Returns
        -------
        precoeff :  `astropy.units.quantity.Quantity` arcsec of 3 elements
            the three precessions correction coefficients

        Notes
        -----
        More recent constants can be found in Capitaine et al. 2003/IAU2006.

        Example
        -------
        >>> from almanac import almanac
        >>> time_utc = ['1999-12-10T00:00:00', '2021-04-10T00:00:00']
        >>> almanac.precession_coeff(time_utc[0], time_utc[1])
        <Quantity [492.00690345, 492.04298664, 427.56661399] arcsec>

        >>> almanac.precession_coeff(time_utc[0], time_utc[0])
        <Quantity [0., 0., 0.] arcsec>

        >>> almanac.precession_coeff(time_utc[1], time_utc[1])
        <Quantity [0., 0., 0.] arcsec>

        >>> from astropy.time import Time
        >>> t0 = Time(2000.0, format='jyear')
        >>> almanac.precession_coeff(t0, t0)
        <Quantity [0., 0., 0.] arcsec>
        """
        t0 = almanac.julianCenturies(time_utc0)
        t = almanac.julianCenturies(time_utc)

        m = np.array([[2306.218, 0.30188, 0.017998],
                      [2306.218, 1.09468, 0.018203],
                      [2004.3109, -0.42665, -0.041833]])

        if t0 != 0.0:  # if the starting date is not exactly J2000.0
            m0 = np.array([[1.39656, -0.000139],
                           [1.39656, -0.000139],
                           [-0.85330, -0.000217]])
            T0 = np.array([t0, t0**2])
            m1 = np.array([-0.000344, 0.000066, 0.000217])

            m2 = m.T
            m2[0] += np.dot(m0, T0)
            m2[1] += m1 * t0
            m = m2.T

        dt = t - t0
        T = np.array([dt, dt**2, dt**3])

        # the zi, zeta, theta values in arcsec
        return np.dot(m, T) * u.arcsec

    @staticmethod
    def approx_sunpos(time_utc):
        """
        Apparent coordinate of the Sun

        Formulas from page C24 of the Astronomical Almanac 1996 [1]

        The position of the Sun in the sky has an accuracy of 0.01 degree
        between the years 1950 and 2050.

        The formulas are based on an elliptical orbit for the Earth, using
        mean orbital elements and a two term approximation for the 'equation of
        centre'. There is also an approximate allowance made for the change i
        obliquity of the ecliptic with time, needed when converting to right
        ascension and declination. The positions are thus apparent positions,
        they are referred to the mean ecliptic and equinox of date.

        1. Find the days before J2000.0
        2. Find the Mean Longitude (L) of the Sun
        3. Find the Mean anomaly (g) of the Sun
        4. Find the ecliptic longitude (lambda) of the sun
        5. Find the obliquity of the ecliptic plane (epsilon)
        6. Find the Right Ascension (alpha) and Declination (delta) of
            the Sun

        References
        ----------
        [1] Astronomical Almanac 1996

        [2] Meeus, Jean (1998) Astronomical Algorithms (2nd Edition).
        Willmann-Bell, Virginia

        Parameter
        ---------
        time_utc: str or `astropy.time.core.Time`
            the observing time in isot format.

        Returns
        -------
        alpha, dec : astropy.units.quantity.Quantity` Angle
            the RA and Dec of the Sun at time_utc

        Example
        -------
        >>> # Sun at 11:00 UT on 1997 August 7th
        >>> from almanac import almanac
        >>> from astropy.coordinates import get_sun
        >>> from PyAstronomy import pyasl
        >>> from astropy.time import Time
        >>> time_utc = "1997-08-07T11:00:00"
        >>> sunpos1 = almanac.approx_sunpos(time_utc)
        >>> sunpos1
        (<Quantity 137.44421045 deg>, <Quantity 16.34198898 deg>)
        >>> t = Time(time_utc)
        >>> sunpos2 = get_sun(t)
        >>> sunpos2.ra
        <Longitude 137.47399496 deg>
        >>> sunpos2.dec
        <Latitude 16.33345951 deg>
        >>> sunpos3 = pyasl.sunpos(t.jd)
         # (array([2450667.95833333]), array([137.43824799]),
         # array([16.34205224]))
        """

        d = Time(time_utc) - Time(2000, format='jyear')  # 1
        d = d.value

        L = (280.461 + 0.9856474 * d) % 360  # 2

        g = (357.528 + 0.9856003 * d) % 360  # 3
        g *= u.deg

        # note that the sin(g) and sin(2*g) terms constitute an
        # approximation to the 'equation of centre' for the orbit
        # of the Sun
        lamb = L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)  # 4
        lamb *= u.deg

        epsilon = 23.439 - 0.0000004 * d  # 5
        epsilon *= u.deg

        y = np.cos(epsilon) * np.sin(lamb)  # 6
        x = np.cos(lamb)

        a = np.arctan2(y, x).to(u.deg)  # Meuus eq. 25.6

        alpha = a % (360 * u.deg)

        delta = np.arcsin(np.sin(epsilon) * np.sin(lamb))  # Meeus eq. 25.7
        delta = (delta % ((2 * np.pi) * u.rad)).to(u.deg)

        return alpha, delta


# Helper routines
def create_obsTime(start, end, step):
    """
    Create a time range in steps

    Parameters
    ----------
    start: `astropy.time.core.Time`
        start of the night

    end : `astropy.time.core.Time`
        end of the night

    step : `astropy.time.core.TimeDelta`
        time step in seconds, example TimeDelta(1.*u.min,format='sec')

    Returns
    -------
    A range of time step in steps of 1 minutes from the start
    to the end of the night :
        'astropy.time.core.Time' the units is the fraction of a day

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.time import Time, TimeDelta
    >>> from astropy.coordinates import EarthLocation
    >>> from almanac import create_obsTime, find_time_at_elevation
    >>> date  = '2025-01-01'
    >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
    ...                           lon=-70.3975 * u.deg,
    ...                           height=2518. * u.m)
    >>> alt_astronomical = -18 * u.deg
    >>> object = 'sun'
    >>> twilight_evening_astronomical =\
        find_time_at_elevation(object, telescope, date,
                               alt_astronomical,
                               'set', which='next')
    >>> twilight_morning_astronomical =\
        find_time_at_elevation(object, telescope, date,
                               alt_astronomical,
                               'rise', which='next')
    >>> obsTime = create_obsTime(twilight_evening_astronomical,
    ...                          twilight_morning_astronomical,
    ...                          TimeDelta(1 * u.min, format='sec'))
    >>> obsTime[0].jd
    2460677.54375
    """
    step_min = step.to(u.min).value
    start = Time(start)
    end = Time(end)
    night_duration = (end - start)
    night_time = start + TimeDelta(np.arange(0.,
                                             night_duration.to(u.min).value,
                                             step_min) * u.min)
    return night_time


def object_altaz(object, time_utc, telescope):
    """
    Parameters
    ----------
    object : str
        the object to be considered 'sun' or 'moon'

    telescope : `astropy.coordinates.earth.EarthLocation`
        The location on the Earth.

    time : str or array of str or `astropy.time.core.Time`
        the utc time in ISO format

    Returns
    -------
    altaz : astropy AltAz object
    """
    time_utc = Time(time_utc)
    altaz_frame = AltAz(obstime=time_utc, location=telescope)
    object_coord = get_body(object, Time(time_utc))
    altaz = object_coord.transform_to(altaz_frame)
    return altaz


def search_crossing(object, time_utc, telescope, crossing_alt):
    """
    Helper function for the bissection search
    """
    altaz = object_altaz(object, time_utc, telescope)
    return (altaz.alt - crossing_alt)


def bissection_crossing(object, t1, t2, telescope,
                        crossing_alt, tol=0.1 * u.deg):
    """
    Use bissection to refine the search for the crossing time
    """
    obj = object
    tel = telescope
    alt = crossing_alt
    t1 = Time(t1)
    t2 = Time(t2)
    t1.format = 'jd'
    t2.format = 'jd'
    tmid = Time((t1.value + t2.value) / 2., format='jd')
    val = search_crossing(obj, tmid, tel, alt)
    if np.abs(val) < tol:
        # stopping condition, report tmid as root
        tmid.format = 'isot'
        return tmid.value
    elif np.sign(search_crossing(obj, t1, tel, alt)) == np.sign(val):
        # case where tmid is an improvement on t1.
        # Make recursive call with t1 = tmid, keep t2
        return bissection_crossing(obj, tmid, t2,  tel, alt, tol=tol)
    elif np.sign(search_crossing(obj, t2, tel, alt)) == np.sign(val):
        # case where tmid is an improvement on t2.
        # Make recursive call with t2 = tmid, keep t1
        return bissection_crossing(obj, t1, tmid, tel, alt, tol=tol)


def find_time_at_elevation(object, telescope, date,
                           crossing_alt, object_path,
                           step=TimeDelta(1 * u.min),
                           tol=0.1 * u.deg,
                           which='next', output_all=False):
    """
    Find the UTC time when an object cross the given altitude
    at the telescope

    Parameters
    ----------
    object : str
        the object to be considered 'sun' or 'moon'

    telescope : `astropy.coordinates.earth.EarthLocation`
        The location on the Earth.

    date : str
        the local date at the start of the observation in the ISO format
        example '2024-06-01'

    crossing_alt : `astropy.units.quantity.Quantity` degree
        the crossing altitude (see notes for the different definitions of
        the twilight)

    object_path : str
        the path of the object: 'set' or 'rise'

    step ; astropy.time.core.TimeDelta, optional, default= 1 min
        the time grid delta time

    which : str, optional, default='next'
        the 'next' time or the 'nearest' time when the object crosses
        the crossing altitude. The zero point is noon of the day at the
        start of the observation.

    output_all : boolean, default=False
        True to also output the altitude of the object.
        if False, only a time of the event is provided

    Returns
    -------
    : str in ISO format
        the UTC time of the object (Sun or Moon) crossing the
        minimum altitude crossing_alt in deg either rising or setting

    Optional output
    ---------------
    tt, tt2, altaz, ind

    tt : str in iso format
        an approximate time

    tt2: str in iso format
        a more accurate time

    altaz: AltAz astropy object
        the altaz grid used for the search for tt

    ind: int
        index in altaz for tt

    Example
    -------
    >>> import astropy.units as u
    >>> from astropy.coordinates import EarthLocation
    >>> from almanac import *
    >>> from astropy.time import Time
    >>> telescope = EarthLocation(lat=-24.615833 * u.deg,
                                  lon=-70.3975 * u.deg,
                                  height=2518 * u.m)
    >>> date = '2024-05-01'
    >>> object = 'sun'
    >>> object_path = 'set'
    >>> which = 'next'
    >>> crossing_alt = -6 * u.deg  # civilian sunset
    >>> step = TimeDelta(5 * u.min)
    >>> tol = 0.1 * u.deg
    >>> tt, tt2, altaz, ind = \
        find_time_at_elevation(object, telescope, date,
                           crossing_alt, object_path, step=step,
                           tol=tol,
                           which=which, output_all=True)
    >>> search_crossing(object, tt, telescope, crossing_alt)
    <Angle -0.02511631 deg>
    >>> search_crossing(object, tt2, telescope, crossing_alt)
    <Angle -0.02511631 deg>
    >>> object_altaz(object, tt2, telescope).alt
    >>> which = 'nearest'
    >>> tt, tt2, altaz, ind = \
        find_time_at_elevation(object, telescope, date,
                           crossing_alt, object_path,
                           which=which, output_all=True)
    >>> search_crossing(object, tt, telescope, crossing_alt)
    <Angle 0.19521984 deg>
    >>> search_crossing(object, tt2, telescope, crossing_alt)
    <Angle 0.08506376 deg>
    >>> object_altaz(object, tt, telescope).alt
    <Latitude -5.80478016 deg>
    >>> object_altaz(object, tt2, telescope).alt
    <Latitude -5.91493624 deg>
    """
    # Find the local noon time in UTC
    localtime = date + 'T12:00:00'
    noon_utc = Time(almanac.local_to_utc(localtime, telescope))
    noon_utc.format = 'isot'
    if which == 'next':  # search the next 24 hr from noon that day
        start = noon_utc
        end = noon_utc + 24 * u.hr
    elif which == 'nearest':  # 12 hr search arounf noon
        start = noon_utc - 12 * u.hr
        end = noon_utc + 12 * u.hr
    else:
        logging.error('Unknown value for which: next or nearest')
        return None
    # get the grid of time
    step = TimeDelta(1 * u.min)
    time_range = create_obsTime(start, end, step)
    altaz = object_altaz(object, time_range, telescope)
    if which == 'next':
        if object_path == 'rise':
            ind = np.where(altaz.alt <= crossing_alt)[0][-1]
        elif object_path == 'set':
            ind = np.where(altaz.alt <= crossing_alt)[0][0]
        else:
            logging.error('Object_path can be "set" or "rise" only')
            return None
    elif which == 'nearest':
        if object_path == 'rise':
            ind = np.where(altaz.alt >= crossing_alt)[0][0]
        elif object_path == 'set':
            ind = np.where(altaz.alt >= crossing_alt)[0][-1]
        else:
            logging.error('Object_path can be "set" or "rise" only')
            return None
    len_time_range = len(time_range)
    if ind < len_time_range:
        t2 = time_range[ind + 1]
    else:
        t2 = time_range[ind]
    if ind > 0:
        t1 = time_range[ind - 1]
    else:
        t1 = ind
    tt2 = bissection_crossing(object, t1, t2, telescope,
                              crossing_alt, tol=tol)

    tt = time_range[ind]
    tt.format = 'isot'
    if output_all:
        return tt.value, tt2, altaz, ind
    else:
        return tt2  # return the more precise value only
