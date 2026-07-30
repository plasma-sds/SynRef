import numpy as np
from scipy.interpolate import RegularGridInterpolator


def pyramidal_horn_near_field_E(x, y, z, a1, b1, rho1, rho2, freq,
                                 E1=1.0, Nx=512, Ny=512, pad=4):
    """
    Near-field (and far-field) E-field of a pyramidal horn via the
    angular-spectrum (plane-wave-spectrum) propagation method.

    1. Sample the horn APERTURE field E_a(x', y') on the horn mouth
       (z'=0 plane): TE10-like amplitude x Balanis quadratic
       phase-error terms (same physics used to build I1, I2 in the
       far-field eq. 13-46/13-48).
    2. 2D FFT -> angular spectrum A(kx, ky)  (exact, no far-field
       approximation).
    3. Propagate each plane-wave component to distance z with the
       exact kz = sqrt(k^2 - kx^2 - ky^2); evanescent modes
       (kx^2+ky^2 > k^2) decay as exp(-|kz| z).
    4. Inverse FFT -> E(x, y, z), valid in reactive near field,
       radiating near field (Fresnel region), and far field alike.

    Parameters
    ----------
    x, y : ndarray [m]  observation-plane coordinates
    z    : float [m]    distance from aperture plane to observation plane
    a1, b1, rho1, rho2, freq, E1 : same meaning as in pyramidal_horn_E
    Nx, Ny : int         FFT grid resolution
    pad    : float       zero-padding factor (window = pad * aperture);
                         larger pad -> finer angular sampling, needed
                         for accurate near-field propagation

    Returns
    -------
    Ex, Ey : complex ndarray, shape of x,y
    xp, yp : 1D ndarrays, internal FFT grid coordinates
    """
    lam = 3e8 / freq
    k = 2.0 * np.pi / lam

    Lx, Ly = pad * a1, pad * b1
    xp = (np.arange(Nx) - Nx / 2) * (Lx / Nx)
    yp = (np.arange(Ny) - Ny / 2) * (Ly / Ny)
    XP, YP = np.meshgrid(xp, yp, indexing='xy')

    inside = (np.abs(XP) <= a1 / 2.0) & (np.abs(YP) <= b1 / 2.0)
    amp = np.zeros_like(XP)
    amp[inside] = np.cos(np.pi * XP[inside] / a1)

    phase = np.exp(-1j * k * (XP**2 / (2.0 * rho2) + YP**2 / (2.0 * rho1)))
    Ea = E1 * amp * phase
    Ea[~inside] = 0.0

    dx, dy = Lx / Nx, Ly / Ny
    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ea))) * dx * dy

    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    kz2 = k**2 - KX**2 - KY**2
    kz = np.where(kz2 >= 0, np.sqrt(np.maximum(kz2, 0)),
                  1j * np.sqrt(np.maximum(-kz2, 0)))
    prop = np.exp(1j * kz.real * z) * np.exp(-kz.imag * z)

    A_prop = A * prop
    Ea_z = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A_prop))) / (dx * dy)

    interp_re = RegularGridInterpolator((yp, xp), Ea_z.real,
                                         bounds_error=False, fill_value=0.0)
    interp_im = RegularGridInterpolator((yp, xp), Ea_z.imag,
                                         bounds_error=False, fill_value=0.0)
    pts = np.stack([np.asarray(y).ravel(), np.asarray(x).ravel()], axis=-1)
    Ex = (interp_re(pts) + 1j * interp_im(pts)).reshape(np.shape(x))
    Ey = np.zeros_like(Ex)

    return Ex, Ey, xp, yp
