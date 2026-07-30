"""
Test function comparing E_theta on the E-plane (phi = pi/2) between:
  1) pyramidal_horn_E        -- closed-form Balanis far-field (Fresnel-integral) formula
  2) pyramidal_horn_near_field_E -- angular-spectrum near/far-field propagator

At a fixed observation radius r deep enough in the far field (r >> 2D^2/lambda),
the two should agree closely near boresight, where the propagator's FFT window
fully captures the relevant angular spectrum content. Discrepancies grow at wide
angles/large r unless the FFT aperture window (`pad`) and resolution (`Nx`,`Ny`)
are increased -- this is a numerical (window-truncation/aliasing) artifact of the
near-field propagator, not a physical one.

Usage
-----
    df = compare_Etheta_Eplane(a1, b1, rho1, rho2, freq, r)
    print(df)
"""
import numpy as np
import pandas as pd
from scipy.special import fresnel
from scipy.interpolate import RegularGridInterpolator


def _C(t):
    S, C = fresnel(t)
    return C


def _S(t):
    S, C = fresnel(t)
    return S


def pyramidal_horn_E(theta, phi, a1, b1, rho1, rho2, freq, E1=1.0, r=1.0):
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    lam = 3e8 / freq
    k = 2.0 * np.pi / lam
    eta = 120.0 * np.pi

    prefactor = (-1j * k / (4.0 * np.pi * r)) * np.exp(-1j * k * r) * E1

    ky = k * np.sin(theta) * np.sin(phi)
    t1 = np.sqrt(1.0 / (np.pi * k * rho1)) * (-k * b1 / 2.0 - ky * rho1)
    t2 = np.sqrt(1.0 / (np.pi * k * rho1)) * (k * b1 / 2.0 - ky * rho1)
    F1 = (_C(t2) - _C(t1)) - 1j * (_S(t2) - _S(t1))
    I1 = np.sqrt(np.pi * rho1 / k) * np.exp(1j * rho1 / (2.0 * k) * ky ** 2) * F1

    kx = k * np.sin(theta) * np.cos(phi)
    kxp = kx + np.pi / a1
    kxpp = kx - np.pi / a1

    t1p = np.sqrt(1.0 / (np.pi * k * rho2)) * (-k * a1 / 2.0 - kxp * rho2)
    t2p = np.sqrt(1.0 / (np.pi * k * rho2)) * (k * a1 / 2.0 - kxp * rho2)
    t1pp = np.sqrt(1.0 / (np.pi * k * rho2)) * (-k * a1 / 2.0 - kxpp * rho2)
    t2pp = np.sqrt(1.0 / (np.pi * k * rho2)) * (k * a1 / 2.0 - kxpp * rho2)

    F2p = (_C(t2p) - _C(t1p)) - 1j * (_S(t2p) - _S(t1p))
    F2pp = (_C(t2pp) - _C(t1pp)) - 1j * (_S(t2pp) - _S(t1pp))
    I2 = 0.5 * np.sqrt(np.pi * rho2 / k) * (
        np.exp(1j * rho2 / (2.0 * k) * kxp ** 2) * F2p +
        np.exp(1j * rho2 / (2.0 * k) * kxpp ** 2) * F2pp)

    obl = 1.0 + np.cos(theta)
    prod = I1 * I2
    E_theta = prefactor * obl * np.sin(phi) * prod
    E_phi = prefactor * obl * np.cos(phi) * prod

    U = r ** 2 / (2.0 * eta) * (np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2)
    return E_theta, E_phi, U


def pyramidal_horn_near_field_E(x, y, z, a1, b1, rho1, rho2, freq,
                                 E1=1.0, Nx=512, Ny=512, pad=4):
    lam = 3e8 / freq
    k = 2.0 * np.pi / lam

    Lx, Ly = pad * a1, pad * b1
    xp = (np.arange(Nx) - Nx / 2) * (Lx / Nx)
    yp = (np.arange(Ny) - Ny / 2) * (Ly / Ny)
    XP, YP = np.meshgrid(xp, yp, indexing='xy')

    inside = (np.abs(XP) <= a1 / 2.0) & (np.abs(YP) <= b1 / 2.0)
    amp = np.zeros_like(XP)
    amp[inside] = np.cos(np.pi * XP[inside] / a1)

    phase = np.exp(-1j * k * (XP ** 2 / (2.0 * rho2) + YP ** 2 / (2.0 * rho1)))
    Ea = E1 * amp * phase
    Ea[~inside] = 0.0

    dx, dy = Lx / Nx, Ly / Ny
    A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ea))) * dx * dy

    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
    KX, KY = np.meshgrid(kx, ky, indexing='xy')

    kz2 = k ** 2 - KX ** 2 - KY ** 2
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
    Ey = (interp_re(pts) + 1j * interp_im(pts)).reshape(np.shape(x))
    Ex = np.zeros_like(Ey)

    return Ex, Ey, xp, yp


def compare_Etheta_Eplane(a1, b1, rho1, rho2, freq, r, theta_deg_max=60,
                           n_theta=10, Nx=768, Ny=768, pad=None, E1=1.0):
    """
    Compares E_theta on the E-plane (phi=pi/2) between the closed-form
    far-field formula and the angular-spectrum near/far-field propagator,
    at a fixed observation radius r, over theta in [0.5, theta_deg_max] deg.

    IMPORTANT: `pad` sets the FFT window size = pad * aperture. It MUST be
    large enough that the window covers the requested lateral observation
    offset r*sin(theta_deg_max); otherwise the interpolator silently
    returns 0 outside the window. If pad=None, it is auto-sized with a
    20% safety margin.
    """
    theta_vals = np.deg2rad(np.linspace(0.5, theta_deg_max, n_theta))
    phi_ep = np.pi / 2

    if pad is None:
        y_max_needed = r * np.sin(theta_vals.max())
        pad = 1.2 * 2 * y_max_needed / b1

    Etheta_far = np.array([
        complex(pyramidal_horn_E(th, phi_ep, a1, b1, rho1, rho2, freq,
                                  E1=E1, r=r)[0]) for th in theta_vals])

    Etheta_near = []
    for th in theta_vals:
        y_obs = r * np.sin(th)
        z_obs = r * np.cos(th)
        Eap, _, xp, yp = pyramidal_horn_near_field_E(
            0.0, y_obs, z_obs, a1, b1, rho1, rho2, freq,
            E1=E1, Nx=Nx, Ny=Ny, pad=pad)
        Etheta_near.append(complex(Eap) * np.cos(th))
    Etheta_near = np.array(Etheta_near)

    mag_far = np.abs(Etheta_far)
    mag_near = np.abs(Etheta_near)
    norm_far = mag_far / mag_far.max()
    norm_near = mag_near / mag_near.max() if mag_near.max() > 0 else mag_near

    return pd.DataFrame({
        "theta_deg": np.degrees(theta_vals),
        "farfield": mag_far,
        "farfield_norm": norm_far,
        "nearfield": mag_near,
        "nearfield_norm": norm_near,
        "abs_diff": np.abs(mag_far - mag_near),
        "abs_norm_diff": np.abs(norm_far - norm_near),
    })


def run_fixed_dx_sweep(a1, b1, rho1, rho2, freq, R_far,
                        r_factors=None, Nx_base=256, Ny_base=256,
                        pad_base=20.0, Nx_cap=10000, Ny_cap=10000,
            ):
            """
            compare_fn must be compare_Etheta_Eplane(a1,b1,rho1,rho2,freq,r,
                                                    Nx=,Ny=,pad=) -> DataFrame
            with an 'abs_diff' column (see main conversation for definition).
            """
            if r_factors is None:
                r_factors = np.arange(0.5, 6.01, 0.5)

            dx_target = (pad_base * a1) / Nx_base
            dy_target = (pad_base * b1) / Ny_base

            rows = []
            for rf in r_factors:
                r_test_i = rf * R_far
                y_max_needed = r_test_i * np.sin(np.deg2rad(60))
                pad_i = max(pad_base, 1.3 * 2 * y_max_needed / b1)

                Nx_i = int(np.ceil((pad_i * a1) / dx_target))
                Ny_i = int(np.ceil((pad_i * b1) / dy_target))
                Nx_i = min(Nx_i + (Nx_i % 2), Nx_cap)
                Ny_i = min(Ny_i + (Ny_i % 2), Ny_cap)

                df_i = compare_Etheta_Eplane(a1, b1, rho1, rho2, freq, r_test_i, n_theta=8,
                                Nx=Nx_i, Ny=Ny_i, pad=pad_i)
                rows.append({
                    "r_factor_x_Rfar": rf, "r_test_m": round(r_test_i, 3),
                    "pad": round(pad_i, 1), "Nx": Nx_i, "Ny": Ny_i,
                    "dx_mm": round((pad_i * a1) / Nx_i * 1000, 4),
                    "max_abs_diff": df_i.abs_diff.max(),
                    "mean_abs_diff": df_i.abs_diff.mean(),
                    "max_abs_norm_diff": df_i.abs_norm_diff.max(),
                    "mean_abs_norm_diff": df_i.abs_norm_diff.mean(),
                })
                print(f"test distance = {r_test_i}")
                print(df_i.to_string(index=False))
            return pd.DataFrame(rows)


if __name__ == "__main__":
    a1, b1 = 0.10, 0.08
    rho1, rho2 = 0.20, 0.22
    freq = 10e9
    lam = 3e8 / freq
    D = max(a1, b1)
    R_far = 2 * D ** 2 / lam
    v2 = False
    if v2:
        r_factors = np.arange(0.5, 6.01, 0.5)
        results = []
        for rf in r_factors:
            r_test_i = rf * R_far
            y_max_needed = r_test_i * np.sin(np.deg2rad(60))
            pad_i = max(20, 1.3 * 2 * y_max_needed / b1)
            df_i = compare_Etheta_Eplane(a1, b1, rho1, rho2, freq, r_test_i,
                                        n_theta=8, Nx=2048, Ny=2048, pad=pad_i)
            results.append({"r_factor_x_Rfar": rf, "r_test_m": r_test_i,
                            "max_abs_diff": df_i.abs_diff.max(),
                            "mean_abs_diff": df_i.abs_diff.mean()})
            print(f"test distance = {r_test_i}")
            print(df_i.to_string(index=False))
            

        summary = pd.DataFrame(results)
        print(summary.to_string(index=False))

    v3 = True
    if v3:
        summary = run_fixed_dx_sweep(a1, b1, rho1, rho2, freq, R_far)
        print(summary.to_string(index=False))
