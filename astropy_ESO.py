import numpy as np
import astropy
import astropy.units as u


def astropy_to_ESO_azimuth(az):
    """
    Convert the Astropy azimuth convention to ESO convention

    ESO's FITS convention where it is measured from the south and
    increasing westward
    https://archive.eso.org/cms/tools-documentation/dicb/
    ESO-044156_6DataInterfaceControlDocument.pdf

    Astropy
    A coordinate or frame in the Altitude-Azimuth system
    (Horizontal coordinates)
    with respect to the WGS84 ellipsoid.
    Azimuth is oriented East of North (i.e., N=0, E=90 degrees).

    * az =   0 <=> south
    * az =  90 <=> east
    * az = 180 <=> north
    * az = 270 <=> west

    Astropy(az)   ESO(az)   ESO(az)    This
                    fits   telescope   code
          deg        deg    deg        deg
        N=0        N=180    180        180
          1          181    179        179
          45         225    135        135
        E=90       E=270     90         90
          120        300     60         60
          135        315     45         45
          179        359      1          1
        S=180      S=0        0          0
          181        1      359        359
          225        45     315        315
        W=270      W=90     270        270
          300        120    240        240

    Parameter
    ---------
    az : `astropy.quantity` deg or float or numpy array, list
        the azimuth in the astropy convention (0 to 360)

    Returns
    -------

    :the same format than the input apart from a list when the
     output is a numpy array
        the corresponding azimuth with the ESO convetion

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy_ESO import *
    >>> ang = [0., 1., 45., 90., 120., 135., 179., 180.,
    ...        181., 225., 270., 300.] * u.deg
    >>> astropy_to_ESO_azimuth(ang)
    <Quantity [180., 179., 135.,  90.,  60.,  45.,   1.,   0., 359., 315.,
               270., 240.] deg>
    """
    if isinstance(az, list):
        az = np.array(az)

    if type(az) is astropy.units.quantity.Quantity:
        cst1 = 180. * u.deg
        cst2 = 360. * u.deg
    else:
        cst1 = 180.
        cst2 = 360.
    return (cst1 - az) % cst2


def wind_to_ESO_azimuth(az):
    """
    Convert wind azimuth information into an ESO azimuth

    Wind(az)   ESO(az)
      N=0        N=180
      E=90       E=270
      S=180      S=0
      W=-90      W=90

    Parameter
    ---------
    az : `astropy.quantity` deg or float or numpy array, list
        the azimuth in the wind convention (-180 to 180)

    Returns
    -------

    :the same format than the input apart from a list when the
     output is a numpy array
        the corresponding azimuth with the ESO convetion

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy_ESO import *
    >>> wind_to_ESO_azimuth(0.)
    180.0
    >>> wind_to_ESO_azimuth([0., 90, -90., 180, -180] * u.deg)
    <Quantity [180., 270.,  90.,   0.,   0.] deg>
    >>> wind_to_ESO_azimuth([0., 90, -90., 180, -180])
    array([180., 270.,  90.,   0.,   0.])
    >>> wind_to_ESO_azimuth(np.array([0., 90, -90., 180, -180]))
    array([180., 270.,  90.,   0.,   0.])
    """
    if isinstance(az, list):
        az = np.array(az)

    q = type(az) is astropy.units.quantity.Quantity
    if q:
        az = az.to(u.deg).value

    c = az < 0.
    eso_az = (((360 + az) * c + az * ~c) + 180.) % 360.

    if q:
        eso_az *= u.deg

    return eso_az


def aperture_mag(mag_arcsec2, aperture):
    """
    Compute the aperture flux from surface brightness and
    aperture angular area on the sky.

    The aperture can be the area of a fibre-fed system and
    thus will compute the fibre magnitude.

    Parameters
    ----------
    mag_arcsec2 : `astropy.units.quantity.Quantity` in mag / arcsec^2
        surface brightness

    aperture : `astropy.units.quantity.Quantity` in arcsec^2
        the aperture area in arsec^2

    Returns
    -------
    : `astropy.units.quantity.Quantity` mag
        the magnitude in the aperture

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy_ESO import *
    >>> aperture = 1.64078944 * u.arcsec**2
    >>> mag_arcsec2 = 15 * u.mag / u.arcsec**2
    >>> aperture_mag(mag_arcsec2, aperture)
    <Quantity 14.46236787 mag>
    """
    aper_mag = -2.5 * np.log10(10**(-0.4 * mag_arcsec2.value) * aperture.value)
    return aper_mag * u.mag


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
