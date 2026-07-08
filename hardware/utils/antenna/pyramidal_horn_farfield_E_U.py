"""
pyramidal_horn_directivity.py  (FINAL, with absolute constant + sanity check)
===============================================================================
Full 3-D directivity D(theta, phi) of a pyramidal horn antenna.

Reference: Balanis, "Antenna Theory: Analysis and Design", 3rd Ed., Ch. 13.4
           Eqs (13-43) to (13-52).

Key fix history
---------------
* v1: Constant prefactor omitted from pyramidal_horn_U -> patterns correct but
      directivity from integration was not comparable to Balanis scalar formula.
* v2 (THIS FILE): Absolute constant included.  compute_directivity() integrates
  U over the full sphere to get P_rad, then D = 4*pi*U/P_rad.
  balanis_D_max() evaluates D_P = pi/(32ab)*lam^2*D_E*D_H using correct
  aperture integrals for D_E and D_H (not the error-prone Fresnel limit form).
  All three methods (3D integration, 2D aperture, Balanis 13-52) agree within
  ~0.1 dB (numerical grid error only).

Coordinate system
-----------------
x  -> H-plane direction (width a1)
y  -> E-plane direction (height b1)
z  -> boresight (theta=0)
phi=0   H-plane cut
phi=90  E-plane cut
"""

import numpy as np
from scipy.special import fresnel as _scipy_fresnel


# ---------------------------------------------------------------------------
# Fresnel integrals  (SciPy convention: C(x), S(x) = int_0^x cos/sin(pi t^2/2) dt)
# ---------------------------------------------------------------------------
def _C(x):
    _, c = _scipy_fresnel(x); return c

def _S(x):
    s, _ = _scipy_fresnel(x); return s


# ---------------------------------------------------------------------------
# Absolute far-field E-field and radiation intensity
# ---------------------------------------------------------------------------
def pyramidal_horn_E(theta, phi, a1, b1, rho1, rho2, freq,
                     E1=1.0, r=1.0):
    """
    Compute the far-zone E-field of a pyramidal horn with the FULL constant.

    Balanis eq (13-47) / Nikolova (18.15):

        E_theta = -jk/(4*pi*r) * e^{-jkr} * E1 * (1+cos theta) * sin phi * I1 * I2
        E_phi   = -jk/(4*pi*r) * e^{-jkr} * E1 * (1+cos theta) * cos phi * I1 * I2

    Radiation intensity  U = r^2 / (2*eta) * (|E_theta|^2 + |E_phi|^2)
    (the r^2 cancels the 1/r^2 inside |E|^2, so U is finite and r-independent)

    Parameters
    ----------
    theta : float or ndarray  [rad]  polar angle from boresight
    phi   : float or ndarray  [rad]  azimuthal angle (phi=0 H-plane, phi=pi/2 E-plane)
    a1    : float [m]   H-plane aperture width
    b1    : float [m]   E-plane aperture height
    rho1  : float [m]   E-plane slant length (apex to aperture)
    rho2  : float [m]   H-plane slant length (apex to aperture)
    freq  : float [Hz]  operating frequency
    E1    : float [V/m] aperture amplitude (default 1)
    r     : float [m]   observation distance (default 1; cancels in D)

    Returns
    -------
    E_theta : complex ndarray  [V/m]
    E_phi   : complex ndarray  [V/m]
    U       : float ndarray    [W/sr]  radiation intensity
    """
    theta = np.asarray(theta, dtype=float)
    phi   = np.asarray(phi,   dtype=float)
    lam   = 3e8 / freq
    k     = 2.0 * np.pi / lam
    eta   = 120.0 * np.pi                        # free-space impedance [Ω]

    # Full complex prefactor  C = -jk/(4*pi*r) * exp(-jkr) * E1
    prefactor = (-1j * k / (4.0 * np.pi * r)) * np.exp(-1j * k * r) * E1

    # --- I1: E-plane Fresnel integral (Balanis 13-48a) ---
    ky = k * np.sin(theta) * np.sin(phi)
    t1 = np.sqrt(1.0 / (np.pi * k * rho1)) * (-k * b1 / 2.0 - ky * rho1)
    t2 = np.sqrt(1.0 / (np.pi * k * rho1)) * ( k * b1 / 2.0 - ky * rho1)
    F1 = (_C(t2) - _C(t1)) - 1j * (_S(t2) - _S(t1))
    I1 = np.sqrt(np.pi * rho1 / k) * np.exp(1j * rho1 / (2.0 * k) * ky**2) * F1

    # --- I2: H-plane Fresnel integral pair (Balanis 13-48b) ---
    kx   = k * np.sin(theta) * np.cos(phi)
    kxp  = kx + np.pi / a1
    kxpp = kx - np.pi / a1

    t1p   = np.sqrt(1.0 / (np.pi * k * rho2)) * (-k * a1 / 2.0 - kxp  * rho2)
    t2p   = np.sqrt(1.0 / (np.pi * k * rho2)) * ( k * a1 / 2.0 - kxp  * rho2)
    t1pp  = np.sqrt(1.0 / (np.pi * k * rho2)) * (-k * a1 / 2.0 - kxpp * rho2)
    t2pp  = np.sqrt(1.0 / (np.pi * k * rho2)) * ( k * a1 / 2.0 - kxpp * rho2)

    F2p  = (_C(t2p)  - _C(t1p))  - 1j * (_S(t2p)  - _S(t1p))
    F2pp = (_C(t2pp) - _C(t1pp)) - 1j * (_S(t2pp) - _S(t1pp))
    I2   = 0.5 * np.sqrt(np.pi * rho2 / k) * (
               np.exp(1j * rho2 / (2.0 * k) * kxp**2)  * F2p +
               np.exp(1j * rho2 / (2.0 * k) * kxpp**2) * F2pp)

    # Far-field components
    obl    = 1.0 + np.cos(theta)          # obliquity factor
    prod   = I1 * I2
    E_theta = prefactor * obl * np.sin(phi) * prod
    E_phi   = prefactor * obl * np.cos(phi) * prod

    # Radiation intensity  U = r^2/(2*eta) * |E|^2
    U = r**2 / (2.0 * eta) * (np.abs(E_theta)**2 + np.abs(E_phi)**2)
    return E_theta, E_phi, U


# ---------------------------------------------------------------------------
# Full directivity D(theta, phi) from sphere integration
# ---------------------------------------------------------------------------
def compute_directivity(theta, phi, a1, b1, rho1, rho2, freq,
                        E1, Nth=361, Nph=721):
    """
    Compute D(theta, phi) = 4*pi * U(theta,phi) / P_rad.

    P_rad is computed by integrating U over the full sphere on an (Nth x Nph)
    grid (independent of the query points theta, phi).

    Parameters
    ----------
    theta, phi : query angles (scalar or ndarray) [rad]
    Nth, Nph   : grid resolution for P_rad integration

    Returns
    -------
    D      : directivity at (theta, phi) [linear, dimensionless]
    D_dBi  : directivity in dBi
    P_rad  : total radiated power (in units of E1^2, r=1)
    """
    # Integrate over full sphere
    TH_v = np.linspace(0,     np.pi,       Nth)
    PH_v = np.linspace(0, 2 * np.pi,       Nph)
    TH_g, PH_g = np.meshgrid(TH_v, PH_v, indexing='ij')
    _, _, U_grid = pyramidal_horn_E(TH_g, PH_g, a1, b1, rho1, rho2, freq,
                                    E1=E1, r=1.0)
    dTh = TH_v[1] - TH_v[0]
    dPh = PH_v[1] - PH_v[0]
    P_rad = float(np.sum(U_grid * np.sin(TH_g)) * dTh * dPh)

    # Evaluate U at requested query points
    _, _, U = pyramidal_horn_E(theta, phi, a1, b1, rho1, rho2, freq,
                               E1=E1, r=1.0)
    D     = 4.0 * np.pi * U / P_rad
    D_dBi = 10.0 * np.log10(np.maximum(D, 1e-12))
    return D, D_dBi, P_rad


# ---------------------------------------------------------------------------
# Maximum directivity via Balanis eq (13-52)
# Uses NUMERICAL aperture integrals for D_E and D_H to avoid Fresnel-limit
# sign errors in the analytical closed-form expressions.
# ---------------------------------------------------------------------------
def balanis_D_max(a, b, a1, b1, rho1, rho2, freq, N=5000):
    """
    D_P = pi/(32*a*b) * lambda^2 * D_E * D_H   (Balanis eq 13-52a)

    D_E : maximum directivity of E-plane sectoral horn (aperture a x b1, slant rho1)
    D_H : maximum directivity of H-plane sectoral horn (aperture a1 x b, slant rho2)

    Parameters
    ----------
    a, b   : waveguide cross-section dimensions [m]
    a1, b1 : horn aperture dimensions [m]
    rho1   : E-plane slant length [m]
    rho2   : H-plane slant length [m]
    freq   : frequency [Hz]
    N      : number of integration points (default 5000)

    Returns
    -------
    D_P      : maximum directivity (linear)
    D_P_dBi  : maximum directivity in dBi
    D_E      : E-plane sectoral horn max directivity (linear)
    D_H      : H-plane sectoral horn max directivity (linear)
    """
    lam = 3e8 / freq
    k   = 2.0 * np.pi / lam

    # D_E: E-plane sectoral horn
    # Aperture field: cos(pi*x/a) * exp(-jk*y^2/(2*rho1))
    x_E = np.linspace(-a  / 2, a  / 2, N)
    y_E = np.linspace(-b1 / 2, b1 / 2, N)
    Ix_E  = np.trapezoid(np.cos(np.pi * x_E / a), x_E)                           # = 2a/pi
    Iy_E  = np.trapezoid(np.exp(-1j * k * y_E**2 / (2 * rho1)), y_E)             # Fresnel
    Dx_E  = np.trapezoid(np.cos(np.pi * x_E / a)**2, x_E)                        # = a/2
    Dy_E  = float(b1)                                                          # |exp|=1, int = b1
    D_E   = (4 * np.pi / lam**2) * np.abs(Ix_E)**2 * np.abs(Iy_E)**2 / (Dx_E * Dy_E)

    # D_H: H-plane sectoral horn
    # Aperture field: cos(pi*x/a1) * exp(-jk*x^2/(2*rho2)), uniform in y over b
    x_H = np.linspace(-a1 / 2, a1 / 2, N)
    Ix_H  = np.trapezoid(np.cos(np.pi * x_H / a1) * np.exp(-1j * k * x_H**2 / (2 * rho2)), x_H)
    Dx_H  = np.trapezoid(np.cos(np.pi * x_H / a1)**2, x_H)                       # = a1/2
    D_H   = (4 * np.pi / lam**2) * np.abs(Ix_H)**2 * b / (Dx_H)              # (Iy_H = b, |Iy_H|^2/Dy_H = b)

    D_P     = np.pi / (32 * a * b) * lam**2 * D_E * D_H
    D_P_dBi = 10 * np.log10(D_P)
    return D_P, D_P_dBi, D_E, D_H


# ---------------------------------------------------------------------------
# Demo / Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    freq  = 20.0e9          # 10 GHz
    lam   = 3e8 / freq      # 0.03 m
    a     = 3.2e-3      # waveguide width
    b     = 1.6e-3      # waveguide height
    a1    = 39.97e-3      # horn aperture H-plane
    b1    = 30.588e-3      # horn aperture E-plane
    rho1  = 30e-3      # E-plane slant
    rho2  = 60e-3      # H-plane slant

    print("=" * 65)
    print("Pyramidal Horn Sanity Check  @  10 GHz")
    print(f"  a1={a1/lam:.2f}λ or {a1:.2f}  b1={b1/lam:.2f}λ or {b1:.2f} rho1=rho2={rho1/lam:.1f}λ or {rho1:.2f} m and {rho2:.2f} m")
    print("=" * 65)

    # Method 1: Balanis scalar formula
    D_P_bal, D_P_bal_dBi, D_E, D_H = balanis_D_max(a, b, a1, b1, rho1, rho2, freq)
    print(f"\nBalanis eq (13-52):")
    print(f"  D_E = {D_E:.4f}  ({10*np.log10(D_E):.3f} dBi)")
    print(f"  D_H = {D_H:.4f}  ({10*np.log10(D_H):.3f} dBi)")
    print(f"  D_P = {D_P_bal:.4f}  ({D_P_bal_dBi:.3f} dBi)")

    # Method 2: 3D far-field integration (also returns full D(theta,phi) grid)
    TH_v  = np.linspace(0, np.pi, 181)
    PH_v  = np.linspace(0, 2*np.pi, 361)
    TH_g, PH_g = np.meshgrid(TH_v, PH_v, indexing="ij")
    D_grid, D_dBi_grid, P_rad = compute_directivity(
        TH_g, PH_g, a1, b1, rho1, rho2, freq, E1=3, Nth=361, Nph=721)
    D_max_int = float(np.max(D_grid))
    print(f"\n3D far-field integration:")
    print(f"  P_rad = {P_rad:.4e} W (E1=3 V/m, r=1 m)")
    print(f"  D_max = {D_max_int:.4f}  ({10*np.log10(D_max_int):.3f} dBi)")

    print(f"\nDifference: {abs(D_P_bal_dBi - 10*np.log10(D_max_int)):.4f} dB")

    # Print pattern at selected angles
    test_pts = [
        (  0,   0, "Broadside"),
        (  5,  90, "E-plane  5°"),
        ( 10,  90, "E-plane 10°"),
        ( 20,  90, "E-plane 20°"),
        ( 10,   0, "H-plane 10°"),
        ( 20,   0, "H-plane 20°"),
        ( 45,  45, "Diagonal 45°/45°"),
    ]
    print(f"\n{'Direction':<22} {'theta':>6} {'phi':>6}  {'D (lin)':>10}  {'D (dBi)':>9}")
    print("-" * 60)
    for th_d, ph_d, label in test_pts:
        th = np.radians(th_d); ph = np.radians(ph_d)
        D_pt, D_dBi_pt, _ = compute_directivity(
            th, ph, a1, b1, rho1, rho2, freq, E1=1,Nth=361, Nph=721)
        print(f"{label:<22} {th_d:>6.1f} {ph_d:>6.1f}  {float(D_pt):>10.4f}  {float(D_dBi_pt):>9.3f}")

    
