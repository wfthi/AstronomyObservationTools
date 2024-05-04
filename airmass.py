import numpy as np
import astropy.units as u
from scipy.optimize import bisect


def airmass(altdegree):
    """
    Plan-parallel airmass formula

    https://en.wikipedia.org/wiki/Air_mass_(astronomy)

    Parameters
    ----------
    altdegree : 'astropy.units.quantity.Quantity' in degrees
        The altitude altdegree in degrees.
        The altitude is 90 deg - zenith angle (deg)

    Returns
    -------
    The airmass : 'astropy.units.quantity.Quantity' (no units)

    Example
    -------
    >>> from airmass import airmassTholenHardie
    >>> import astropy.units as u
    >>> airmass(30 * u.deg)
    1.9999999999999996
    """
    return (1. / np.cos(90. * u.deg - altdegree)).value


def airmassTholenHardie(altdegree, Hardie=False):
    """
    Compute the airmass of an object ginve its altitude in degree
    This routine uses the Tholen Hardie airmass formula and is
    adapted from airmass.pro by Marc Buie IDL routine

    This is more accurate than the simple 1/cos

    https://www.boulder.swri.edu/~buie/idl/pro/airmass.html

    Parameters
    ----------
    altdegree : 'astropy.units.quantity.Quantity' in degrees
        The altitude altdegree in degrees.
        The altitude is 90 deg - zenith angle (deg)

    Returns
    -------
    The airmass : 'astropy.units.quantity.Quantity' (no units)

    Examples
    --------
    >>> from airmass import airmassTholenHardie
    >>> import astropy.units as u
    >>> airmassTholenHardie(45 * u.degree)
    <Quantity 1.41235393>
    >>> a = airmassTholenHardie(45)
    Warning: input has no units. Assume to be degrees
    """
    if not isinstance(altdegree, u.quantity.Quantity):
        print("Warning: input has no units. Assume to be degrees")
        altdegree *= u.deg
    PI = np.pi * u.rad
    altrad = altdegree.to(u.rad)
    zenith = PI / 2. - altrad
    # z = np.where(zenith <= 1.521 * u.rad)  # < 87.15 deg
    n = 1.00029   # for default wave, pressure, temp, relhum
    zenith = np.arcsin(np.sin(zenith) / n)
    cz = np.cos(zenith)
    if (Hardie):
        x = 1.0 / np.cos(zenith) - 1
        am = 1.0 + ((0.9981833 - (0.002875 + 0.0008083 * x) * x) * x)
    else:
        # Tholen
        am = np.sqrt(235225.0 * cz * cz + 970.0 + 1.0) - 485. * cz
    return am


def airmassfunc(altdegree, airmass):
    """
    Helper function for altdegree_from_airmass

    Parameters
    ----------
    altdegree : 'astropy.units.quantity.Quantity' in degrees
        The altitude altdegree in degrees.
        The altitude is 90 deg - zenith angle (deg)

    airmass : float >= 1

    Returns
    -------
    The airmass : 'astropy.units.quantity.Quantity' (no units)
    """
    return airmassTholenHardie(altdegree * u.deg) - airmass


def altdegree_from_airmass(airmass):
    """
    Use the bissection method to compute the 90-zenigh angle = alt in degrees
    given the airmass using Hardie function.
    Return the altitude in degrees

    Parameters
    ----------
        airmass : float >= 1

    Returns
    -------
    altdegree : `astropy.units.quantity.Quantity` in degrees
        telescope altitude in degrees

    Required external packages:
        scipy.bissect

    Functions called:
        airmassfunc

    Examples
    --------
    >>> import astropy.units as u
    >>> from airmass import altdegree_from_airmass
    >>> from airmass import airmassTholenHardie
    >>> airmass = 1.2
    >>> altdegree = altdegree_from_airmass(airmass)
    >>> altdegree
    <Quantity 56.39251709 deg>
    >>> am = airmassTholenHardie(altdegree)
    >>> am
    <Quantity 1.19999963>
    >>> # altdegree = altdegree_from_airmass(0.9)
    """
    assert np.all(airmass >= 1.), 'airmass has to be >= 1'

    altdegree = bisect(airmassfunc, 10, 90,
                       args=(airmass),
                       rtol=1e-5, maxiter=100)
    return altdegree * u.deg


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
