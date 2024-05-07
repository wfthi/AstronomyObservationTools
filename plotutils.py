#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
plotting functions

See specific authors

"""
# standard library
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Longitude, Latitude
# third-party
import matplotlib.pyplot as plt
import numpy as np
# local


def aitoff(long, b):
    """
    Carry out Aitoff projection.

    Parameters
    ----------
    long, b : float or array
        The longitude and latitude [deg]

    long is between -180 and 180

    b is between -90 and 90

    Returns
    -------
    x, y : float or array
        Aitoff-projected coordinates (`x`, `y`)

    Notes
    -----

    .. note:: This function was ported from the IDL Astronomy User's Library.

    :IDL - Documentation:

    pro aitoff,l,b,x,y
    +
     NAME:
           AITOFF
     PURPOSE:
           Convert longitude, latitude to X,Y using an AITOFF projection.
     EXPLANATION:
           This procedure can be used to create an all-sky map in Galactic
           coordinates with an equal-area Aitoff projection.  Output map
           coordinates are zero longitude centered.

     CALLING SEQUENCE:
           AITOFF, L, B, X, Y

     INPUTS:
           L - longitude - scalar or vector, in degrees
           B - latitude - same number of elements as L, in degrees

     OUTPUTS:
           X - X coordinate, same number of elements as L.   X is normalized to
                   be between -180 and 180
           Y - Y coordinate, same number of elements as L.  Y is normalized to
                   be between -90 and 90.

     NOTES:
           See AIPS memo No. 46, page 4, for details of the algorithm.  This
           version of AITOFF assumes the projection is centered at b=0 degrees.

     REVISION HISTORY:
           Written  W.B. Landsman  STX          December 1989
           Modified for Unix:
                   J. Bloch        LANL SST-9      5/16/91 1.1
           Converted to IDL V5.0   W. Landsman   September 1997
    """
    RADEG = 180.0 / np.pi
    wasFloat = False
    if isinstance(long, float):
        long = np.array([long])
        b = np.array([b])
        wasFloat = True

    sa = long.copy()
    x180 = np.where(sa > 180.0)[0]
    if len(x180) > 0:
        sa[x180] -= 360.
    alpha2 = sa / (2 * RADEG)  # convert degree to rad
    delta = b / RADEG  # convert degree to rad
    r2 = np.sqrt(2.)
    f = 2 * r2 / np.pi
    cdec = np.cos(delta)
    denom = np.sqrt(1. + cdec * np.cos(alpha2))
    x = cdec * np.sin(alpha2) * 2. * r2 / denom
    y = np.sin(delta) * r2 / denom
    x = x * RADEG / f
    y = y * RADEG / f

    if wasFloat:
        return float(x), float(y)
    else:
        return x, y


def inverseAitoff(x, y):
    """
    Carry out an inverse Aitoff projection.

    This function reverts to aitoff projection made by the
    function *aitoff*. The result is either two floats or
    arrays (depending on whether float or array was used
    as input) representing longitude and latitude. Both are
    given in degrees with -180 < longitude < +180 and
    -90 < latitude < 90.

    Parameters
    ----------
    x : float or array
        A value between -180. and +180.
        (see convention in *aitoff* function).

    y : float or array
        A value between -90. and +90
        (see convention in *aitoff* function).

    Returns
    -------
    Deprojected coordinates : float or array
        If arrays are used for input, the function returns an
        array for the longitude, for the latitude, and
        an index array containing those array indices for which
        the reprojection could be carried out.

    Reference
    ---------
    From PyAstronomy. Adapted from a IDL routine
    """
    wasFloat = False
    if isinstance(x, float):
        x = np.array([x])
        y = np.array([y])
        wasFloat = True

    # First, rescale x and y
    x = x / 180.0 * 2.0 * np.sqrt(2.0)
    y = y / 90.0 * np.sqrt(2.0)

    zsqr = 1.0 - (x / 4.0)**2 - (y / 2.0)**2
    # Check whether x,y coordinates are within the ellipse of
    # invertible values.
    indi = np.where((x**2 / 8. + y**2 / 2. - 1.0) <= 0.0)[0]
    if len(indi) == 0:
        print("Deprojection is not possible.")
        print("in inverseAitoff")
        print("No values inside valid space.")
        exit()
    z = np.sqrt(zsqr[indi])
    long = 2.0 * np.arctan(z * x[indi] / (2.0 * (2.0 * z**2 - 1.0)))
    b = np.arcsin(z * y[indi])

    long = long * 180.0 / np.pi
    b = b*180.0 / np.pi

    if wasFloat:
        return float(long), float(b)
    else:
        return long, b, indi


def aitoff_grid(max_lat, dlong=30., dlat=30., title=None, label360=True):
    """
    Generate a grid for a Aitoff projection map

    Wing-Fai Thi, dec 2018, MPE
    adapted from the IDL routine PRO AITOFF_GRID
    use numpy, basic matplotlib (no basemap)

    Parameters
    ----------

    max_lat : float
        here the upper limit of
        the aitoff grid in degrees (for 4most 10 to 40 degrees)

    dlong,optional : real. Default = 30 degrees
        the longitude interval for the longitude grid,

    dlat,optional  : real. Default = 30 degrees
        the latitude interval for the latitude grid,

    label360 : real, optional, default=True
        if True from 0 to 360 deg
        if False from -180 to 180 deg

    title, optional : str

    Returns
    --------
    lngtot : int
        the number of longitude grid lines

    lattot : int
        the number of latitude grid lines
    """
#
#       Do lines of constant longitude
#
    transparency = 0.5
    shift = 3
    lat = np.arange(-90, 1 + max_lat, 1)
    llat = len(lat)
    lngtot = np.floor(180. / dlong).astype(np.int64)
    for i in range(lngtot):
        lng = np.full(llat, -180.0 + (i * dlong))
        x, y = aitoff(lng, lat)
        plt.plot(x, y, c='black', alpha=transparency)
        plt.plot(-x, y, c='black', alpha=transparency)
        # labels
        # i * dlong from 0 to +180
        if label360:
            str1 = str(np.floor(i * dlong + 180).astype(np.int64))
        else:
            str1 = str(360. - np.floor(i * dlong).astype(np.int64))
        plt.text(x[llat - 1] - shift, np.max(y) + shift, str1, style='italic')
        str2 = str(np.floor(180.-i * dlong).astype(np.int64))
        plt.text(-x[llat - 1] - shift, np.max(y) + shift, str2, style='italic')

    # plt the longitude 0 line
    x, y = aitoff(np.full(llat, 0.), lat)   #
    plt.plot(x, y, c='black', alpha=transparency)
    x0, y0 = aitoff(0., max_lat)
    plt.text(x0 - shift, y0 + shift, '0', style='italic')
#
#       Do lines of constant latitude
#
    lng = np.arange(-180, 180, 1)  # between -180 and 180
    llng = len(lng)
    lattot = np.floor((90. + max_lat) / dlat).astype(np.int64) + 1
    for i in range(lattot):
        lat = np.full(llng, -90.0 + (i * dlat))
        x, y = aitoff(lng, lat)
        plt.plot(x, y, c='black', alpha=transparency)
        str1 = str(np.floor(i*dlat - 90.).astype(np.int64))
        if (np.min(y) <= 0):
            if (np.min(lat) >= -80):
                plt.text(np.max(x) + shift, np.min(y) - shift, str1,
                         style='italic')
        else:
            if (np.min(lat) <= 80):
                plt.text(np.max(x) + shift, np.max(y) + shift, str1,
                         style='italic')

    x, y = aitoff(lng, max_lat)
    plt.plot(x, y, c='black', alpha=transparency)

    if (title is not None):
        font = 18
        xshift_title = len(title) * (font / 7.)
        yshift_title = 15 * (font / 10.)
        plt.text(x0 - xshift_title, y0 + yshift_title, title, size=font)

    plt.axis('off')

    return lngtot, lattot


def aitoff_density_map(ra_in, dec_in, max_lat=40,
                       bin_list=[300, 1000],
                       colbarlabel=None,
                       figsize=[12, 7],
                       title=None,
                       colormap=plt.cm.ocean_r, filename=None):
    """
    Plot density plot (2D histogram) on a Aitoff projected map

    Parameter
    ---------
    ra_in : array-like of floats
        the RA in degrees 0-360

    dec_in : array-like of floats
        the Declinations between -90 and +90 degrees

    max_lat : float, optional, default=40
        the maximum latitude shown in the plot

    bin_list : array of 2 rank (float), optional, default bin_list=[300, 1000]
        the number of bins for RA and Dec

    figize : 1D array of rank 2, optional, default figsize=[12, 7]
        the size of the figure

    tile : str, optional, default=None
        the tile of the plot

    colbarlabel : str, optional, default=None
        the label for the colorbar

    colormap : str, optional default=plt.cm.ocean_r
        the colormap

    filename: str, optional, default filename=None
        if filename is not None, a file with filename will be
        written

    Returns
    -------
        screen output and optional output file

    Example
    -------
    >>> import numpy as np
    >>> from plotutils import aitoff_density_map
    >>> npoints = 100000
    >>> ra_in = 359.99 * np.random.random(npoints)
    >>> dec_in = -60 + 90 * np.random.random(npoints)
    >>> nbins = int(npoints / 500)
    >>> tit = 'Example: Artificial satellites'
    >>> colbarlabel = 'Number density'
    >>> aitoff_density_map(ra_in, dec_in, bin_list=[nbins, nbins],
    ...                    colbarlabel=colbarlabel, title=tit)
    """
    ra180 = Longitude(ra_in * u.deg)
    dec90 = Latitude(dec_in * u.deg)

    ra180.wrap_angle = 180 * u.deg

    xra, ydec = aitoff(ra180.value, dec90.value)  # aitoff projection
    xfig_size = figsize[0]
    yfig_size = figsize[1]
    for nbins in bin_list:
        plt.figure(figsize=(xfig_size, yfig_size), frameon=0)
        plt.hist2d(xra, ydec, bins=nbins, cmap=colormap)
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.axis('off')
        _, _ = aitoff_grid(max_lat, title=title)
        plt.colorbar(label=colbarlabel)
        plt.tight_layout()
        if (filename is not None):
            stem = Path(filename).stem
            extension = Path(filename).suffix
            output_filename = stem + '_nbins_' + str(nbins) + extension
            print('Saving maps in ', output_filename)
            plt.savefig(output_filename)
        else:
            plt.show()


def aitoff_map(ra_in, dec_in, z,
               max_lat=40,
               s=10,
               colbarlabel=None,
               title=None,
               figsize=[12, 7],
               colormap='rainbow',
               ra_extra=None,
               dec_extra=None,
               s_extra=15,
               c_extra='black',
               filename=None):
    """
    Make a Aitoff plot of points at RA and Dec with values z

    Parameter
    ---------
    ra_in : array-like of floats
        the RA in degrees 0-360

    dec_in : array-like of floats
        the Declinations between -90 and +90 degrees

    z : array-like of floats
        the values at (ra_in, dec_in)

    max_lat : float, optional, default=40
        the maximum latitude shown in the plot

    s : int, optional, default=10
        the size of the plot symbols

    colbarlabel : str, optional, default=None
        the label for the colorbar

    title : str, optional, default=None
        the title of the plot

    figize : 1D array of rank 2, optional, default figsize=[12, 7]
        the size of the figure

    ra_extra : array-like, optional, default=None
        RA of the extra points to be plotted

    dec_extra : array-like, optional, default=None
        Dec of the extra points to be plotted

    s_extra: int, optional, default s_extra=15
        the size of the extra symbols

    c_extra: str, optional, default c_extra='back'
        the color of the extra symbols

    filename: str, optional, default filename=None
        if filename is not None, a file with filename will be
        written

    Returns
    -------
        screen output and optional output file

    Example
    -------
    >>> import numpy as np
    >>> from plotutils import aitoff_map
    >>> npoints = 100000
    >>> ra_in = 359.99 * np.random.random(npoints)
    >>> dec_in = -60 + 90 * np.random.random(npoints)
    >>> z = np.random.random(npoints) * 10.
    >>> extra_ra_in = [0, 90, 180, 270, 360, 45., 23.]
    >>> extra_dec_in = [-30, -20, -10, 0, 10, -45, -23.]
    >>> extra_z = [1, 2, 1, 2, 1, 3, 4]
    >>> aitoff_map(ra_in, dec_in, z, ra_extra=extra_ra_in,
    ...            dec_extra=extra_dec_in)
    """
    xfig_size = figsize[0]
    yfig_size = figsize[1]
    # Ensure that -180 deg < ra < 180 deg
    ra = Longitude(ra_in * u.deg)
    dec = Latitude(dec_in * u.deg)
    ra.wrap_angle = 180 * u.deg
    xra, ydec = aitoff(ra.value, dec.value)
    plt.figure(figsize=(xfig_size, yfig_size), frameon=0)
    plt.scatter(xra, ydec, c=z, cmap=colormap, s=s)
    plt.colorbar(label=colbarlabel)
    if (ra_extra is not None) & (dec_extra is not None):
        ra_extra180 = Longitude(ra_extra * u.deg)
        dec_extra90 = Latitude(dec_extra * u.deg)
        ra_extra180.wrap_angle = 180 * u.deg
        xra_extra, ydec_extra = aitoff(ra_extra180.value, dec_extra90.value)
        if isinstance(xra_extra, float):
            xra_extra = [xra_extra]
        if isinstance(ydec_extra, float):
            ydec_extra = [ydec_extra]
        if len(xra_extra) == len(ydec_extra):
            print('plot extra')
            plt.scatter(xra_extra, ydec_extra, c=c_extra, s=s_extra)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.axis('off')
    _, _ = aitoff_grid(max_lat, title=title)
    plt.tight_layout()
    if (filename is not None):
        print('Saving map in ', filename)
        plt.savefig(filename)
    else:
        plt.show()


def polar_sky(ra, dec, z, altaz_frame, figure_size=(8, 6),
              extra_az_rad=None, extra_za_deg=None,
              zlabel=None, title=None):
    """
    Example
    -------
    >>> from plotutils import polar_sky
    >>> import astropy.units as u
    >>> from astropy.coordinates import AltAz
    >>> from astropy.time import Time
    >>> from astropy.coordinates import get_body
    >>> from astropy.coordinates import EarthLocation
    >>> vista = EarthLocation(lat=-24.615833 * u.deg,
    ...                       lon=-70.3975 * u.deg,
    ...                       height=2518 * u.m)
    >>> obsTime = Time('2024-06-1T4:00:00', format='isot')
    >>> x = [0., 45., 123., 87., 256., 305, 167., 22., 220., 145.]
    >>> y = [-45., -35., -23., -12., 5., 7.5, -2.5, -12.7, -9.0, -21.]
    >>> z = [18., 19., 20., 21., 18.5, 19.2, 20.8, 19.2, 19.7, 20.2]
    >>> zlabel = 'Star magnitude'
    >>> moon=get_body('moon', obsTime)
    >>> altaz_frame = AltAz(obstime=obsTime,
    ...                    location=vista)
    >>> moon_altaz = moon.transform_to(altaz_frame)
    >>> extra_az_rad = moon_altaz.az.to(u.rad).value
    >>> extra_za_deg = 90. - moon_altaz.alt.value
    >>> obsTime.format='isot'
    >>> tit = 'lat ' + str(vista.lat) + '   lon ' + str(vista.lon)
    >>> tit += '   at ' + obsTime.value
    >>> polar_sky(x, y, z, altaz_frame,
    ...           extra_az_rad=extra_az_rad,
    ...           extra_za_deg=extra_za_deg,
                  zlabel=zlabel, title=tit)
    """
    _, ax = plt.subplots(figsize=figure_size,
                         subplot_kw=dict(projection='polar'))
    if isinstance(ra, (list, np.ndarray)):
        x = ra * u.deg
    if isinstance(dec, (list, np.ndarray)):
        y = dec * u.deg
    coord = SkyCoord(ra=x, dec=y)
    coord_altaz = coord.transform_to(altaz_frame)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0., 90.)
    plt.scatter(coord_altaz.az.to(u.rad).value,
                90. - coord_altaz.alt.value, c=z, s=5)
    plt.colorbar(label=zlabel)
    plt.title(title)
    plt.scatter(extra_az_rad, extra_za_deg, c='black', s=15)
    plt.show()
