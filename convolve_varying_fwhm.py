"""
Convolve an array with a varying kernel

    Copyright (C) 2020-2024 Wing-Fai Thi

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

The Resolving power is independent of wavelengthbecause of the properties of
a simple diffraction grating, where the resolving power is just the order
multiplied by the numberof rule lines on the grating. On the other hand the
resolution delta_wavelength depends on the wavelength given a resolving power.
The FWHM of the convolving kernel in a simulated spectrum should account for
this change in FWHM.

Doctest >>> python3 convolve_varying_fwhm.py
"""
import sys
import numpy as np
from scipy.integrate import newton_cotes


def gaus(x, a, x0, sigma):
    """
    Used for fitting a gaussian
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def gauss1d(pars, x):
    """
    FHWM = 2*np.sqrt(2*np.log(2)) sigma
    1/ (2 sigma**2) = (4. * 2. np.log(2)) / FWHM / 2 = 4. * np.log(2) / FWHM

    gauss1d pars = { fwhm, pos, ampl }

    normalized gauss1d, set ampl = 1./ (sigma*np.sqrt(2.*np.pi))
    ampl = 2*np.sqrt(2.np.log(2))/(FWHM*np.sqrt(2.*np.pi))
         = 2.*np.sqrt(np.log(2.)/np.pi)/FWHM
         = 0.9394372786996513/FWHM
    This value should be close to 1: ampl*sigma*np.sqrt(2.*np.pi))
    gfactor = 4.0 * ln(2.0)
    """
    gfactor = 2.7725887222397811
    diff = (x - pars[1]) / pars[0]
    return pars[2] * np.exp(-gfactor * diff * diff)


def int_trapezoidal(x, f):
    """
    Use the Numpy trapezoidal numerical integration method
    """
    return np.trapz(f, x)


def int_rectangular(xwidth, ymid):
    """
    Rectangular numerical integration
    """
    if (len(xwidth) != len(ymid)):
        print('input x and y must be of the same length!')
        sys.exit()
    return np.sum(xwidth*ymid)


def int_tabulated(x, f, p=5):
    """
    Numerical integration for tabulated function using
    the Newton-Cotes interpolation
    """
    def int_newton_cotes(x, f):
        if x.shape[0] < 2:
            return 0
        rn = (x.shape[0] - 1) * (x - x[0]) / (x[-1] - x[0])
        weights = newton_cotes(rn)[0]
        return (x[-1] - x[0]) / (x.shape[0] - 1) * np.dot(weights, f)
    ret = 0
    for idx in range(0, x.shape[0], p - 1):
        ret += int_newton_cotes(x[idx:idx + p], f[idx:idx + p])
    return ret


def make_lsf_mat(wave, fwhm, normalize=True):
    """
    1. interpolate the input source and sky data onto a uniform wavelength
        grid, save the uniform grid
    2. interpolate the FWHM as function of wavelength onto this uniform grid
    3. create a lsf matrix, ie a lsf with varying FWHM
    4. convolve the source and the sky with the wavelength varying profile
    5. re-interpolate to the input wavelength grid (the input wavelength
        grid does not need to be uniform, equal distance between grid points)
    6. check the flux conservation and re-scale

    wave = np.arange(3000., 4000., 0.3)
    fwhm = np.full(len(wave), 1.0)
    lsf = make_lsf(wave, fwhm)
    flux = gauss1d([1.,3500.,1.], wave)
    """
    wave = np.array(wave)
    mssg = 'Wavelength sampling must be uniform (uniform lambda grid).'
    if not (abs(np.max(np.diff(wave)) - np.min(np.diff(wave))) <
            0.000001 * np.diff(wave)[0]):
        raise RuntimeError(mssg)
    fwhm = np.array(fwhm)
    afac = 2. * np.sqrt(np.log(2.) / np.pi)
    lsf_y = []
    # no need to np.flip because the lsf is symetric!
    for lamb, lsf_fwhm in zip(wave, fwhm):
        g = gauss1d([lsf_fwhm, lamb,
                     afac / lsf_fwhm], wave)
        if normalize:
            g = g / int_tabulated(wave, g)
        lsf_y.append(g)
    return np.array(lsf_y)


def convol_varying_kernel(a, bmat):
    """
    Convolve an array to a matrix of kernels

    Paramater
    ---------
    a : array-like
        the array of values to be convolved

    bmat : 2D array
        the convolution matrix

    Return
    ------
    mat : 2D array
        transpose of the matrix so that convolution of any other array c is
        np.dot(c, mat.T)

    conv : 1D numpy array
        the convolution output array

    Examples
    --------
    >>> from convolve_varying_fwhm import *
    >>> a =  np.array([3., 4., 5., 6., 8., 10., 8.])
    >>> bmat =  np.tile([2., 1., 3.], 9).reshape(9, 3)
    >>> conv0 = np.convolve(a, bmat[0])
    >>> mat, conv1 = convol_varying_kernel(a, bmat)
    >>> all(conv0 == conv1)
    True
    >>> # Kernel varies for each element
    >>> bmat =  np.array([[2., 1., 3.],[3., 1., 3.],[2., 5., 3.],[2., 0., 3.],
    ...                   [6., 2., 3.],[2., 1., 5.],
    ...                   [8., 1., 2.],[2., 3., 3.],[2., 2., 3.]])
    >>> mat, conv1 = convol_varying_kernel(a, bmat)
    >>> conv1
    array([ 6., 15., 39., 24., 75., 58., 90., 54., 24.])
    """
    lb = len(bmat[0])
    la = len(a)
    if (len(bmat) != la + lb - 1):
        print('the convolution matrix has dimension 0 should be'
              'len(input) - 1')
        sys.exit()
    mat = np.full((la + lb - 1, la), 0.)
    for i in range(lb):
        # print(i, i + 1, np.flip(bmat[i, :])[lb - i - 1: lb])
        mat[i, 0:i + 1] = np.flip(bmat[i, :])[lb - i - 1: lb]
    for i in range(lb, la):
        # print(i, i - lb + 1,i)
        mat[i, i - lb + 1: i + 1] = np.flip(bmat[i, :])
    for i in range(la + 1, la + lb):
        # print(i - 1, i - lb,la, np.flip(bmat[i - 1, :])[0:la + lb - i])
        mat[i - 1, i - lb: la] = np.flip(bmat[i - 1, :])[0: la + lb - i]
    conv = np.dot(a, mat.T)
    return mat, conv


def convol_var_fwhm_mat(flux, lsf_mat):
    """
    Convolve a simulated input flux by a line spectral function
    of varying shape

    Note
    ----
    The routine does not preserve the flux

    Parameter
    ---------
    flux : array-like

    lsf_mat : 2D array
        the number of rows has to correspond to the number of elements in flux

    Return
    ------
    : array-like
        the convolve flux

    Example
    -------
    >>> import numpy as np
    >>> from scipy.optimize import curve_fit
    >>> from convolve_varying_fwhm import *
    >>> wave = np.arange(3000.,4000.,0.3)
    >>> wavelength0 = np.array([3200.,3500.,3800.])
    >>> fwhm_cst = 3.0
    >>> fwhm0 = np.full(3, fwhm_cst)
    >>> lsf_fwhm0 = wavelength0**3 * 1e-10
    >>> lsf_fwhm = wave**3 * 1e-10
    >>> lsf_mat = make_lsf_mat(wave, lsf_fwhm, normalize=False)
    >>> # make a spectrum with 3 gaussians
    >>> flux = (gauss1d([fwhm0[0], wavelength0[0], 1.], wave) +
    ...         gauss1d([fwhm0[1], wavelength0[1], 1.], wave) +
    ...         gauss1d([fwhm0[2], wavelength0[2], 1.], wave))
    >>> results = convol_var_fwhm_mat(flux, lsf_mat)
    >>> # Flux normalization
    >>> int_flux = int_tabulated(wave, flux)
    >>> int_results = int_tabulated(wave, results)
    >>> results = results / int_results * int_flux
    >>> # test by fitting the convolved spectrum and compare the theretical
    >>> # and measured FWHM
    >>> for mean, line_fwhm, lsf_fwhm in zip(wavelength0, fwhm0, lsf_fwhm0):
    ...     ampl = 1.0
    ...     total_fwhm = np.sqrt(line_fwhm**2 + lsf_fwhm**2)
    ...     lsf_sigma = total_fwhm / (2 * np.sqrt(2 * np.log(2.)))
    ...     popt, _ = curve_fit(gaus, wave, results,
    ...                         p0=[ampl, mean, lsf_sigma])
    >>> np.isclose(popt[2], lsf_sigma, rtol=1e-3)
    True
    """
    return np.dot(flux, lsf_mat.T)


if __name__ == "__main__":
    import doctest
    DOCTEST = True
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
