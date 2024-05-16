"""
Convert from the Vega to the AB system an vice-versa

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
"""
import numpy as np


def Vega_AB_magnitude_conversion(inputMagSystem, inputBand,
                                 inputMag, verbose=False):
    """
    Convert from AB to Vega and vice-versa

    Ref.
    http://www.astronomy.ohio-state.edu/~martini/usefuldata.html

    Blanton et al. 2007

    Gaia
    https://gea.esac.esa.int/archive/documentation/GDR2/
    Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html#SSS6

            ZP(Vega)   ZP(AB)   ZP(AB)-ZP(Vega)
    ------------------------------------------
    G       25.6884    25.7934      0.105
    Gbp     25.3514    25.3806	    0.0292
    Grp     24.7619    25.1161	    0.3542

    Parameter
    ---------
    inputMagSystem : str
        'AB' or 'Vega'

    inputBand : str
        one of the valid AB or Vega band

    inputMag : float or array of floats
        magnitude in the input-band + input system

    Return
    ------
    : float or array
        the magnitude in the other system

    Example
    -------
    >>> import numpy as np
    >>> from photutils import Vega_AB_magnitude_conversion
    >>> f, magVega = Vega_AB_magnitude_conversion('AB', 'U', 6.35)
    >>> np.isclose(magVega, 5.56, rtol=1e-2)
    True
    >>> f, magVega = Vega_AB_magnitude_conversion('AB', 'g', 5.12)
    >>> np.isclose(magVega, 5.20, rtol=1e-2)
    True
    >>> f, magVega = Vega_AB_magnitude_conversion('AB', 'g',
    ...                                           [5.12, 6., 8.5, 16.5])
    >>> magVega
    array([ 5.2 ,  6.08,  8.58, 16.58])
    """
    flag = 0
    validMagSystem = ('AB', 'Vega')
    validBands = ('U', 'B', 'V', 'R', 'I', 'J', 'H', 'Ks',
                  'u', 'g', 'r', 'i', 'z', 'Y', 'G', 'GBP', 'GRP')
    mAB_minus_mVega = (0.79, -0.09, 0.02, 0.21, 0.45, 0.91, 1.39, 1.85,
                       0.91, -0.08, 0.16, 0.37, 0.54, 0.634, 0.105,
                       0.0292, 0.3542)

    if inputMagSystem not in validMagSystem:
        print("Invalid input Magnitude system")
        return flag, 999.

    # test if inputBand is valid
    if (inputBand not in validBands):
        print('Invalid input magnitude band. It has to be one of')
        for vband in validBands:
            print(vband, ' ', end="")
        print()
        flag = -1
        return flag, 999.

    # test the type of each input magnitudes. They should be
    # convertable to a float
    inputType = type(inputMag)
    if (inputType == str or inputType == int or inputType == float or
            inputType == np.float64):
        correctedInput = float(inputMag)
    if (inputType in [list, tuple, np.ndarray]):
        correctedInput = []
        for i in inputMag:
            try:
                if isinstance(i, float):
                    correctedInput.append(float(i))
            except TypeError:
                print("One of the input magnitudes is invalid")
                flag = -1
                return flag, 999.
        correctedInput = np.array(correctedInput)

    conversion = dict(zip(validBands, mAB_minus_mVega))
    mssg = "Invalid magnitude input system. It has to be AB " \
        "(to be converted to Vega) or Vega (to be converted to AB)"
    if (inputMagSystem == 'AB'):  # conversion from AB to Vega system
        if verbose:
            print('Conversion to the Vega system')
            print('Conversion factor:', conversion[inputBand])
        magOutput = correctedInput - conversion[inputBand]
    else:
        if (inputMagSystem == 'Vega'):  # conversion from Vega to AB
            if verbose:
                print('Conversion to the AB system')
            magOutput = correctedInput + conversion[inputBand]
        else:
            print(mssg)
            flag = -1
            return flag, 999.
    return flag, magOutput


if __name__ == "__main__":
    import doctest
    DOCTEST = True
    doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)
