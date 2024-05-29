"""
Simulate Interferometric observations with a cartesian grid model images
as input. The code performs the van Cittert-Zernike theorem integration
in polar coordinates

    python >= 3.9

    Copyright (C) 2024>  Wing-Fai Thi

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
from numpy import exp, abs, angle, cos, sin
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from multiprocessing import Pool


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def xyz_to_mesh(x, y, z):
    X, Y = np.meshgrid(x, y)
    Z = np.reshape(z, np.shape(X))
    return X, Y, Z


def zip_xy(x, y):
    return np.array([[p, q] for p, q in list(zip(x, y))])


def x_y_to_xy(x, y):
    pair = []
    for xx in x:
        for yy in y:
            pair.append([xx, yy])
    return np.array(pair)


def gaussian(npix=101, sig2=1.):
    x = np.linspace(-5, 5, npix)
    y = np.linspace(-5, 5, npix)
    X, Y = np.meshgrid(x, y)  # 2D grid for interpolation
    Z = exp(-(X*X + Y*Y) / sig2)
    return x, y, X, Y, Z


def cylinder(r=1, npix=101):
    x = np.linspace(-5, 5, npix)
    y = np.linspace(-5, 5, npix)
    X, Y = np.meshgrid(x, y)
    Z = ((X*X + Y*Y) < r) * 1.
    return x, y, X, Y, Z


class Image():
    """
    Format of the image (x, y, I) and (rho, theta, I)
    at one wavelength
    """
    def __init__(self, x=None, y=None,
                 rho=None, theta=None,
                 lamda=None,
                 Ipolar=None,
                 Icart=None,  # Source total Intensity energy/surface
                 Flux=None,  # Source total FLux in energy/surface/sr
                 dA=None,  # pixel area projected on the sky in steradian
                 dOmega=None) -> None:
        self.x = x
        self.y = y
        self.rho = rho
        self.theta = theta
        self.lamda = lamda
        self.Ipolar = Ipolar
        self.Icart = Icart
        self.Flux = Flux
        self.dOmega = dOmega
        self.dA = dA

    def compute_rhotheta(self):
        lx = len(self.x)
        ly = len(self.y)
        assert (lx == ly)
        rhotheta, z = [], []
        for i, x in enumerate(self.x):
            for k, y in enumerate(self.y):
                c = x + 1j * y
                rhotheta.append([abs(c), angle(c)])
                z.append(self.Icart[i, k])
        self.rhotheta = np.array(rhotheta)
        self.Irhotheta = np.array(z)

    def compute_Ipolar(self, rho, dtheta):
        """
        - Convert the requested polar grid into a cartesian grid
        - Interpolate the intensity in the cartesian grid
        """
        theta = np.arange(0, 360, dtheta) * np.pi / 180. - np.pi
        # Find all the (rho, theta) pairs
        rt = x_y_to_xy(rho, theta)
        # Transform the polar to catesian coordinate
        xx, yy = pol2cart(rt[:, 0], rt[:, 1])
        if xx.max() > self.x.max():
            print("Choose a lower maximum rho value")
        xy = x_y_to_xy(self.x, self.y)
        lx = len(self.x)
        ly = len(self.y)
        z = np.reshape(self.Icart.T, lx * ly)
        # Interpolate in the cartesian coordinate system
        # It uses the griddata with the cubic method
        Ipol = griddata(xy, z, zip_xy(xx, yy), method='cubic')
        # create the mesh grid for x, y, and z
        Rho, Theta = np.meshgrid(rho, theta)
        Ipolar = Ipol.reshape(Rho.T.shape).T
        self.rho = rho
        self.theta = theta
        self.Rho = Rho
        self.Theta = Theta
        self.Ipolar = Ipolar

    def compute_dOmega(self, rho, dtheta, dist):
        """
        Compute the area of between rho and rho + dtheta
        given the angle dtheta

        rho in au
        dtheta in radian
        d in pc
        dOmega in au^2 / d^2 = sr

        The boundary is at 1/2 between consercutive grid point in
        radius
        """
        drplus = 0.5 * (rho[1] - rho[0])
        if rho[0] == 0.:
            drmin = 0.
        else:
            drmin = drplus
        dOmega = [(rho[0] + drplus)**2 - (rho[0] - drmin)**2]
        for i, r in enumerate(rho[1:-1]):
            drmin = drplus
            drplus = 0.5 * (rho[i + 2] - r)
            dOmega.append((r + drplus)**2 - (r - drmin)**2)
        dr = 0.5 * (rho[-1] - rho[-2])
        dOmega.append((rho[-1] + dr)**2 - (rho[-1] - dr)**2)
        dOmega = dtheta * np.pi * np.array(dOmega) / dist**2
        self.dOmega = dOmega

    def compute_flux(self):
        """
        Compute the total flux
        """
        flux = 0.
        for Ipolar, dOmega in zip(np.transpose(self.Ipolar), self.dOmega):
            flux += (Ipolar * dOmega).sum()
        self.flux = flux


class Target():
    """
    Class container for target information
    """
    def __init__(self, dist=None, PA=None,
                 dec=None, ) -> None:
        self.dist = dist
        self.PA = PA
        self.dec = dec
        pass


class Observation():

    def __init__(self,
                 target=None,
                 station=None,
                 image=None,
                 nproc=1) -> None:
        self.target = target  # Target object
        self.image = image
        self.station = station  # Station object
        self.nproc = nproc
        pass

    def compute_uv_tracks(self, hourAngle):
        """
        Compute the uv track given the Hour Angle the
        target has been observed

        Ref.
        ----
        Damien Segransan: https://www.jmmc.fr/mirrors/obsvlti/
        https://scienceworld.wolfram.com/physics/TransferFunction.html
        """
        if isinstance(hourAngle, float):
            hourAngle = [hourAngle]
        dec = self.target.dec
        lamda = self.image.lamda
        dX = self.station.dX
        dY = self.station.dY
        u_HA, v_HA = [], []
        for h in hourAngle:
            u = sin(h) * dX + cos(h) * dY
            v = -sin(dec) * cos(h) * dX + sin(dec) * sin(h) * dY
            u /= lamda
            v /= lamda
            u_HA.append(u)
            v_HA.append(v)
        self.u = np.array(u_HA)
        self.v = np.array(v_HA)

    def compute_V2(self):
        """
        Compute Vsibility^2
        """
        self.V2 = (self.Re**2 + self.Im**2) / self.image.flux**2

    def expose(self, hourAngle):
        if isinstance(hourAngle, float):
            hourAngle = [hourAngle]
        targetDistance = self.target.dist
        targetPA = self.target.PA
        image = self.image
        station = self.station
        obs_Re, obs_Im = [], []
        for h in hourAngle:
            obsTargetPA = targetPA + station.FieldRotationRate * h
            if self.nproc == 1:
                station.compute_vCZ(image, targetDistance, obsTargetPA)
            else:
                station.multiprocess_vCZ(image, targetDistance, obsTargetPA)
            obs_Re.append(station.Re)
            obs_Im.append(station.Im)
        obs_Re = np.array(obs_Re).flatten()
        obs_Im = np.array(obs_Im).flatten()
        self.Re = obs_Re
        self.Im = obs_Im
        self.Phi = np.arctan(obs_Im / obs_Re)
        self.compute_V2()


class Station():

    def __init__(self, id=None,
                 baseline=None,
                 PA=None,
                 lat=None) -> None:
        self.id = id
        self.baseline = baseline
        self.PA = PA
        if baseline is not None and PA is not None:
            self.compute_dXdY()
        self.lat = lat  # radian
        if lat is not None:
            self.compute_FieldRotationRate()

    def compute_FieldRotationRate(self):
        """
        Field of view rotation in degress/hour
        Angular Rate of Rotation of the Earth (degrees/hour)
        x cos(Observer's latitude)
        (15.04106858 degrees/hour = 360° / hours in a Sidereal day)
        """
        self.FieldRotationRate = 15.04106858 * cos(self.lat)

    def compute_dXdY(self):
        """
        The dX, dY values of the baselines given
        the baseline separations and Postion Angles

        Assume dZ = 0.
        """
        self.dX = self.baseline * cos(self.PA)
        self.dY = self.baseline * sin(self.PA)

    def compute_vCZ_1baseline(self, i, b, pa, image, targetPA, d):
        # Total Positon Angle
        cosPAtot = -cos(pa + image.theta + targetPA)
        kx_fac = 2. * np.pi * b / image.lamda * cosPAtot / d
        intensity = np.transpose(image.Ipolar)
        Re_baseline, Im_baseline = 0., 0.
        for rho, dOmega, Ipolar in zip(image.rho, image.dOmega,
                                       intensity):
            kx = kx_fac * rho
            Re_baseline += Ipolar * cos(kx) * dOmega
            Im_baseline += Ipolar * sin(kx) * dOmega
        return i, Re_baseline, Im_baseline

    def multiprocess_vCZ(self, image, targetDistance, targetPA, nproc=8):
        d = targetDistance  # pc
        Re, Im, pos = [], [], []
        pool = Pool(nproc)
        processes = [pool.apply_async(self.compute_vCZ_1baseline,
                                      args=(i, b, pa,
                                            image, targetPA, d,))
                     for i, (b, pa) in enumerate(zip(self.baseline, self.PA))]
        for pr in processes:
            p = pr.get()
            pos.append(p[0])
            Re.append(p[1].sum())
            Im.append(p[2].sum())
        self.Re = np.array(Re)
        self.Re = self.Re[pos]
        self.Im = np.array(Im)
        self.Im = self.Im[pos]
        # Compute also the phase
        self.Phi = np.arctan(self.Im / self.Re)

    def compute_vCZ(self, image, targetDistance, targetPA):
        """
        Approximate form of van Cittert-Zernike theorem

        There are Nant x (Nant - 1) x 0.5 distinct baselines

        ALMA 50 x 12m + 12 x 7m + 4 Total Power -> 1891 stations

        https://www.eso.org/observing/etc/doc/viscalc/vltistations.html
        """
        d = targetDistance  # pc
        Re, Im = [], []
        # loop over the stations - This loop can be parallelized
        intensity = np.transpose(image.Ipolar)
        for b, pa in zip(self.baseline, self.PA):
            # Total Positon Angle
            cosPAtot = -cos(pa + image.theta + targetPA)
            kx_fac = 2. * np.pi * b / image.lamda * cosPAtot / d
            Re_baseline, Im_baseline = 0., 0.
            for rho, dOmega, Ipolar in zip(image.rho, image.dOmega,
                                           intensity):
                kx = kx_fac * rho
                Re_baseline += Ipolar * cos(kx) * dOmega
                Im_baseline += Ipolar * sin(kx) * dOmega
            Re.append(Re_baseline.sum())
            Im.append(Im_baseline.sum())
        self.Re = np.array(Re)
        self.Im = np.array(Im)
        # Compute also the phase
        self.Phi = np.arctan(self.Im / self.Re)


def run_example(name):
    if name == 'gaussian':
        x, y, X, Y, Z = gaussian()
    else:
        x, y, X, Y, Z = cylinder(r=2.0, npix=501)

    # Instantiate a target object with an arbitray distance
    # declination, PA, and distance
    source = Target(dist=1., PA=0., dec=-45.)
    #
    plt.pcolormesh(X, Y, Z, shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cartesian coordinates - ' + name)
    cbar = plt.colorbar()
    cbar.set_label('Intensity')
    plt.show()

    # instantiate an image object
    img = Image(x=x, y=y, Icart=Z)

    # Steps to convert the (x, y, Ixy) to (rho, theta, Ipol)

    # Interpolation to a finer grid close to the maximum flux location
    # The user defines the new rho and dtheta (for a regular theta grid)
    rho = np.array([0., 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3,
                    8e-3, 9e-3,
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                    0.8, 0.9,
                    1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
    dtheta = 2.5
    img.compute_Ipolar(rho, dtheta)
    plt.pcolormesh(img.Rho, img.Theta,
                   img.Ipolar, shading='auto')
    plt.xlabel('r')
    plt.ylabel(r'$\theta$')
    plt.title('Polar coordinates')
    cbar = plt.colorbar()
    cbar.set_label('Intensity')
    plt.show()

    img.lamda = 1.  # arbitrary value
    img.compute_dOmega(rho, dtheta, source.dist)
    img.compute_flux()

    # Create artificial baselines
    baseline = np.arange(0, 3.0, 0.01)
    PA = np.random.random(len(baseline)) * 180.
    PA = [45.] * len(baseline)

    # Paranal VLTI latitutude in radians
    lat = -24.62794830 / 180. * np.pi

    # vlti is a Station object
    vlti = Station(id=id, baseline=baseline, PA=PA, lat=lat)

    # Instantiate an Observation object
    Obsrun = Observation(image=img,
                         target=source,
                         station=vlti,
                         nproc=1)

    # A grid of hour angle, here one hour before and after meridian transit
    hourAngle = (np.arange(0, 120, 10.) - 60.) / 60.
    hourAngle = 0.
    # Compute the UV track
    Obsrun.compute_uv_tracks(hourAngle)

    # Expose during the HourAngle with multiple exposures
    Obsrun.expose(hourAngle)

    u = Obsrun.u.flatten()
    v = Obsrun.v.flatten()
    plt.scatter(u, v, s=1)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('uv track')
    plt.show()

    uv2 = Obsrun.u.flatten()**2 + Obsrun.v.flatten()**2
    plt.scatter(uv2, Obsrun.V2, s=1)
    plt.xlabel(r'u$^2$ + v$^2$')
    plt.ylabel('V$^2$')
    plt.show()

    uv2 = Obsrun.u.flatten()**2 + Obsrun.v.flatten()**2
    plt.scatter(u, np.sqrt(Obsrun.V2), s=1)
    plt.xlabel('baseline')
    plt.ylabel('|V|')
    plt.show()


if __name__ == "__main__":
    run_example('gaussian')
    run_example('cylinder')
