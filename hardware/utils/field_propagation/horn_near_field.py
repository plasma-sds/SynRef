"""
horn_near_field.py
==================

Near-field propagation of horn-antenna aperture fields.

Contents
--------
1) pyramidal_horn_near_field_E(...)
       Original idealised Balanis aperture (cosine taper + quadratic phase)
       propagated with the scalar angular-spectrum method. Unchanged API.

2) Loading measured / simulated aperture fields from text files
       load_field_txt(path)            -> FieldPlane  (CST "ASCII export" format)
       load_aperture(e_file, h_file)   -> ApertureField (E and optionally H)
       ideal_aperture_field(...)       -> ApertureField built from the Balanis model

3) Antenna pose (position + tilt of the antenna w.r.t. the global frame)
       rotation_matrix(rx_deg, ry_deg, rz_deg)
       AntennaPose.from_tilt(...)

4) Propagation
       propagate_parallel(ap, z)       fast FFT angular spectrum, untilted antenna,
                                       target plane parallel to the aperture
       propagate_to_points(ap, pts, pose, method=...)
                                       arbitrary global points, tilted antenna
       propagate_to_z_plane(ap, z, x, y, pose, method=...)
                                       convenience wrapper: global plane z = const

5) fw2d bridge
       fw2d_pose(...), aperture_to_fw2d_line(...)   Ez, ampl, phase on the fw2d
                                       source line (angle 0 -> propagate_parallel,
                                       angle != 0 -> propagate_to_points)
       load_aperture_cached(...), select_cst_export(...)

   method="spectrum"      vector plane-wave (angular-spectrum) summation evaluated
                          directly at the rotated target points. Uses only the
                          tangential aperture E-field (exact for the half space
                          in front of the aperture). H is derived from E, or
                          (h_source="loaded") from the loaded tangential H.
   method="stratton_chu"  Love / Stratton-Chu equivalence-principle integral with
                          J = n x H and M = -n x E on the aperture plane.
                          Uses BOTH loaded E and H, valid at every point in front
                          of the plane (including large angles).

Conventions
-----------
* Time dependence exp(+j w t) (CST convention), +z propagation ~ exp(-j k z).
* All lengths inside the module are in metres. The text loader converts the
  unit given in the column header (mm, cm, um, m) automatically.
* Field arrays are shaped (3, Ny, Nx) with components (x, y, z).
* Antenna frame (primed): aperture plane z' = 0, boresight +z', aperture
  sampled on the (x', y') grid of the text file, centred on `ap.center`.
* AntennaPose maps antenna frame -> global frame:
        r_global = pose.R @ r_antenna + pose.t
  Field vectors transform the same way: E_global = R @ E_antenna.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0 ** 2)
ETA0 = np.sqrt(MU0 / EPS0)

_UNIT = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "um": 1e-6, "µm": 1e-6, "mil": 25.4e-6,
         "in": 25.4e-3}


# =============================================================================
# 1) Original idealised aperture propagator (kept for backwards compatibility)
# =============================================================================
def pyramidal_horn_near_field_E(x, y, z, a1, b1, rho1, rho2, freq,
                                E1=1.0, Nx=512, Ny=512, pad=4):
    """Idealised Balanis aperture propagated by the scalar angular spectrum.

    Returns (Ex, Ey, xp, yp); the aperture field is y-polarised (Ex = 0).
    NOTE: kept unchanged. For tilted antennas / measured apertures use
    ideal_aperture_field(...) + propagate_to_z_plane(...).
    """
    lam = C0 / freq
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


# =============================================================================
# 2) Data containers and text-file loading
# =============================================================================
@dataclass
class FieldPlane:
    """One vector field sampled on a uniform z = const grid."""
    x: np.ndarray            # (Nx,) [m]
    y: np.ndarray            # (Ny,) [m]
    z: float                 # [m]
    F: np.ndarray            # (3, Ny, Nx) complex
    quantity: str            # "E" or "H"
    unit: str = ""           # e.g. "V/m"

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])


@dataclass
class ApertureField:
    """Complex aperture field in the ANTENNA frame (aperture plane z' = 0).

    x, y   : 1-D grids [m] relative to `center` (i.e. the aperture centre is 0,0)
    E, H   : (3, Ny, Nx) complex arrays, H may be None
    freq   : frequency [Hz]
    center : position of the grid origin in the file's coordinate system [m]
             (x0, y0, z0). Used as the default antenna position in the global
             frame when no pose is given.
    """
    x: np.ndarray
    y: np.ndarray
    E: np.ndarray
    freq: float
    H: Optional[np.ndarray] = None
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def k(self):
        return 2.0 * np.pi * self.freq / C0

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])

    def power_z(self) -> float:
        """Forward power 0.5*Re(E x H*)_z through the aperture plane [W]."""
        if self.H is None:
            raise ValueError("H is required for the Poynting flux")
        Sz = 0.5 * np.real(self.E[0] * np.conj(self.H[1]) - self.E[1] * np.conj(self.H[0]))
        return float(Sz.sum() * abs(self.dx * self.dy))


def _parse_header(line: str):
    """Return [(name, unit), ...] from a CST header like 'x [mm]  ExRe [V/m] ...'."""
    return [(n.strip(), u.strip()) for n, u in re.findall(r"([^\s\[\]]+)\s*\[([^\]]*)\]", line)]


def load_field_txt(path: str, quantity: Optional[str] = None) -> FieldPlane:
    """Load a 2-D CST ASCII field export (x y z  FxRe FxIm FyRe FyIm FzRe FzIm).

    Supported column layouts (detected from the header):
      * Re/Im : xRe xIm yRe yIm zRe zIm   (CST default, e.g. 'ExRe [V/m]')
      * no header: 9 numeric columns in the above order, coordinates in mm.
    The grid ordering in the file is irrelevant; points are sorted onto a
    regular (y, x) grid. Coordinates are converted to metres.
    """
    with open(path, "r", errors="replace") as fh:
        first = fh.readline()
    cols = _parse_header(first)
    skip = 0
    if cols:
        skip = 1
        with open(path, "r", errors="replace") as fh:
            fh.readline()
            second = fh.readline()
        if set(second.strip()) <= set("-= "):
            skip = 2
    data = np.loadtxt(path, skiprows=skip)
    if data.ndim != 2 or data.shape[1] < 9:
        raise ValueError(f"{path}: expected >= 9 numeric columns, got {data.shape}")

    if cols:
        names = [c[0] for c in cols]
        coord_unit = _UNIT.get(cols[0][1], None)
        if coord_unit is None:
            raise ValueError(f"Unknown coordinate unit '{cols[0][1]}' in {path}")
        fname = names[3]
        q = quantity or fname[0].upper()
        field_unit = cols[3][1]
        # map component columns by name, robust against column order
        idx = {}
        for i, n in enumerate(names):
            m = re.match(r"^[A-Za-z]([xyz])(Re|Im)$", n)
            if m:
                idx[(m.group(1), m.group(2))] = i
        if len(idx) != 6:
            raise ValueError(f"{path}: could not identify Re/Im columns in header {names}")
        comp = [data[:, idx[(c, "Re")]] + 1j * data[:, idx[(c, "Im")]] for c in "xyz"]
    else:
        coord_unit, q, field_unit = 1e-3, (quantity or "E"), ""
        comp = [data[:, 3] + 1j * data[:, 4], data[:, 5] + 1j * data[:, 6],
                data[:, 7] + 1j * data[:, 8]]

    xs, ys, zs = data[:, 0] * coord_unit, data[:, 1] * coord_unit, data[:, 2] * coord_unit
    if np.ptp(zs) > 1e-9:
        raise ValueError(f"{path}: points are not on a single z plane (ptp z = {np.ptp(zs)} m)")

    # snap to regular grid (tolerant against print rounding)
    def _axis(v):
        u = np.unique(np.round(v, 9))
        return u
    xu, yu = _axis(xs), _axis(ys)
    Nx, Ny = len(xu), len(yu)
    if Nx * Ny != len(xs):
        raise ValueError(f"{path}: {len(xs)} points do not form a full {Nx}x{Ny} grid")
    if Nx > 2 and np.ptp(np.diff(xu)) > 1e-3 * abs(xu[1] - xu[0]):
        warnings.warn(f"{path}: x grid is not uniform; FFT methods assume uniform spacing")
    if Ny > 2 and np.ptp(np.diff(yu)) > 1e-3 * abs(yu[1] - yu[0]):
        warnings.warn(f"{path}: y grid is not uniform; FFT methods assume uniform spacing")
    ix = np.searchsorted(xu, np.round(xs, 9))
    iy = np.searchsorted(yu, np.round(ys, 9))
    F = np.zeros((3, Ny, Nx), dtype=complex)
    for c in range(3):
        F[c, iy, ix] = comp[c]
    return FieldPlane(x=xu, y=yu, z=float(zs.mean()), F=F, quantity=q, unit=field_unit)


def load_aperture(e_file: str, freq: float, h_file: Optional[str] = None,
                  center_xy: Optional[Tuple[float, float]] = (0.0, 0.0)) -> ApertureField:
    """Load aperture E (and optionally H) from CST text exports.

    Parameters
    ----------
    e_file, h_file : paths of the E and H exports on the SAME plane and grid
    freq           : frequency [Hz] (not stored in CST ASCII exports)
    center_xy      : antenna axis position (x0, y0) in the file's coordinate
                     system [m]. The local aperture grid is x - x0, y - y0.
                     None -> use the |E|^2-weighted centroid of the plane.
    """
    pe = load_field_txt(e_file, "E")
    H = None
    if h_file is not None:
        ph = load_field_txt(h_file, "H")
        if (ph.F.shape != pe.F.shape or not np.allclose(ph.x, pe.x) or
                not np.allclose(ph.y, pe.y) or abs(ph.z - pe.z) > 1e-9):
            raise ValueError("E and H files must be sampled on the same plane and grid")
        H = ph.F
    if center_xy is None:
        w = np.sum(np.abs(pe.F) ** 2, axis=0)
        X, Y = np.meshgrid(pe.x, pe.y)
        center_xy = (float((w * X).sum() / w.sum()), float((w * Y).sum() / w.sum()))
    x0, y0 = center_xy
    return ApertureField(x=pe.x - x0, y=pe.y - y0, E=pe.F, H=H, freq=freq,
                         center=np.array([x0, y0, pe.z]))


def ideal_aperture_field(a1, b1, rho1, rho2, freq, E1=1.0, polarization="y",
                         taper_axis="x", dx=None, dy=None, extent=3.0,
                         with_H=True) -> ApertureField:
    """Balanis pyramidal-horn aperture as an ApertureField (so it can be tilted).

    a1 : aperture width along `taper_axis` (cosine taper, H-plane)
    b1 : aperture height along the other axis (uniform, E-plane)
    rho2 / rho1 : phase radii along the a1 / b1 directions.
    polarization : 'x' or 'y'  (CST horn exports above are x-polarised with
                   the taper along y -> polarization='x', taper_axis='y').
    H is the local plane-wave approximation H = z x E / eta0 (with_H=True).
    """
    lam = C0 / freq
    k = 2 * np.pi / lam
    dx = dx or lam / 8
    dy = dy or lam / 8
    Lx = extent * (a1 if taper_axis == "x" else b1)
    Ly = extent * (b1 if taper_axis == "x" else a1)
    x = (np.arange(int(np.ceil(Lx / dx))) - np.ceil(Lx / dx) / 2) * dx
    y = (np.arange(int(np.ceil(Ly / dy))) - np.ceil(Ly / dy) / 2) * dy
    X, Y = np.meshgrid(x, y)
    T, U = (X, Y) if taper_axis == "x" else (Y, X)          # T: taper coordinate
    inside = (np.abs(T) <= a1 / 2) & (np.abs(U) <= b1 / 2)
    Ea = E1 * np.cos(np.pi * T / a1) * np.exp(-1j * k * (T ** 2 / (2 * rho2) + U ** 2 / (2 * rho1)))
    Ea[~inside] = 0
    E = np.zeros((3,) + X.shape, complex)
    E[0 if polarization == "x" else 1] = Ea
    H = None
    if with_H:
        H = np.zeros_like(E)
        H[0], H[1] = -E[1] / ETA0, E[0] / ETA0
    return ApertureField(x=x, y=y, E=E, H=H, freq=freq)


# =============================================================================
# 3) Antenna pose
# =============================================================================
def rotation_matrix(rx_deg=0.0, ry_deg=0.0, rz_deg=0.0, order="xyz") -> np.ndarray:
    """Rotation matrix from successive rotations about the GLOBAL axes.

    order="xyz": first about x by rx, then about y by ry, then about z by rz
    (extrinsic, right-handed, positive = counter-clockwise looking down the
    axis towards the origin).
      ry > 0 tilts the boresight (+z) towards +x   (rotation in the x-z plane)
      rx > 0 tilts the boresight (+z) towards -y   (rotation in the y-z plane)
      rz     rotates the aperture (polarisation) about the boresight when
             rx = ry = 0
    """
    a = np.deg2rad
    cx, sx = np.cos(a(rx_deg)), np.sin(a(rx_deg))
    cy, sy = np.cos(a(ry_deg)), np.sin(a(ry_deg))
    cz, sz = np.cos(a(rz_deg)), np.sin(a(rz_deg))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    mats = {"x": Rx, "y": Ry, "z": Rz}
    R = np.eye(3)
    for ax in order:
        R = mats[ax] @ R
    return R


@dataclass
class AntennaPose:
    """Antenna frame -> global frame:  r_g = R @ r_a + t.

    t is the global position of the aperture centre (antenna-frame origin).
    """
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_tilt(cls, aperture_center, rx_deg=0.0, ry_deg=0.0, rz_deg=0.0,
                  pivot=None, order="xyz"):
        """Pose of an antenna whose untilted aperture centre is `aperture_center`
        (global, [m]) and which is rotated about `pivot` (global, default: the
        aperture centre itself, e.g. use the horn throat or a mount point)."""
        c = np.asarray(aperture_center, float)
        R = rotation_matrix(rx_deg, ry_deg, rz_deg, order)
        p = c if pivot is None else np.asarray(pivot, float)
        return cls(R=R, t=R @ (c - p) + p)

    @property
    def boresight(self):
        return self.R[:, 2]

    def to_antenna(self, pts_g):
        return (np.asarray(pts_g) - self.t) @ self.R          # == R^T (r - t)

    def boresight_hit_z(self, z_plane):
        """Global point where the boresight axis crosses the plane z = z_plane."""
        b = self.boresight
        if abs(b[2]) < 1e-12:
            return None
        s = (z_plane - self.t[2]) / b[2]
        return self.t + s * b


# =============================================================================
# 4) Propagation
# =============================================================================
def _spectrum(ap: ApertureField, pad: int, comps=("E",)):
    """Zero-padded FFT spectra of the tangential components.

    Returns kx (Mx,), ky (My,), dict name->(2, My, Mx) spectra, x0, y0
    The spectrum S is defined such that
        F(x, y, 0) = 1/(Mx*My) * sum S[m,n] exp(+j(kx_n (x-x0) + ky_m (y-y0)))
    i.e. numpy's ifft convention referenced to the first grid sample (x0, y0).
    """
    Ny, Nx = ap.E.shape[1:]
    Mx, My = int(pad * Nx), int(pad * Ny)
    kx = 2 * np.pi * np.fft.fftfreq(Mx, d=ap.dx)
    ky = 2 * np.pi * np.fft.fftfreq(My, d=ap.dy)
    out = {}
    for name in comps:
        F = ap.E if name == "E" else ap.H
        S = np.zeros((2, My, Mx), complex)
        for c in range(2):
            S[c] = np.fft.fft2(F[c], s=(My, Mx))
        out[name] = S
    return kx, ky, out, ap.x[0], ap.y[0]


def _kz(k, KX, KY):
    kz2 = k ** 2 - KX ** 2 - KY ** 2
    # exp(+jwt): propagating kz > 0, evanescent kz = -j|kz|  -> exp(-j kz z) decays
    return np.where(kz2 >= 0, np.sqrt(np.abs(kz2)), -1j * np.sqrt(np.abs(kz2)))


def _complete_vector(KX, KY, KZ, Sx, Sy):
    """Ez of plane waves exp(j(kx x + ky y) - j kz z) from the divergence
    condition. Wave vector kappa = (-kx, -ky, kz) in exp(-j kappa.r) notation:
    kappa.E = 0  ->  Ez = (kx Ex + ky Ey) / kz."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Sz = np.where(np.abs(KZ) > 0, (KX * Sx + KY * Sy) / KZ, 0)
    return Sx, Sy, Sz


def _forward_E(k, KX, KY, KZ, Ex, Ey, Hx, Hy, kz_min_rel=1e-3):
    """Tangential E spectrum of the FORWARD (+z) travelling part only.

    At the aperture plane E_t = E+ + E-,  H_t = Y E+ - Y E-  (Y = spectral
    admittance of +z plane waves). Hence  E+ = (E_t + Y^-1 H_t) / 2  with
        Y^-1 = eta0/(k kz) [[kx ky, ky^2+kz^2], [-(kx^2+kz^2), -kx ky]].
    This is the spectral-domain equivalent of Love's equivalence principle
    (J = n x H, M = -n x E) and removes waves reflected back towards the horn
    and the field-sampling inconsistencies between E and H.
    """
    kzr = np.where(np.abs(KZ) < kz_min_rel * k, kz_min_rel * k, KZ)
    c = ETA0 / (k * kzr)
    Exf = 0.5 * (Ex + c * (KX * KY * Hx + (KY ** 2 + KZ ** 2) * Hy))
    Eyf = 0.5 * (Ey + c * (-(KX ** 2 + KZ ** 2) * Hx - KX * KY * Hy))
    return Exf, Eyf


def _build_spectra(ap, pad, h_source, wave):
    """Full (Ex,Ey,Ez,Hx,Hy,Hz) spectra on the padded k-grid."""
    if wave == "auto":
        wave = "forward" if ap.H is not None else "total"
    if (wave == "forward" or h_source == "loaded") and ap.H is None:
        raise ValueError("wave='forward' / h_source='loaded' need the aperture H-field")
    comps = ("E", "H") if (wave == "forward" or h_source == "loaded") else ("E",)
    kx, ky, S, x0, y0 = _spectrum(ap, pad, comps)
    KX, KY = np.meshgrid(kx, ky)
    KZ = _kz(ap.k, KX, KY)
    Sx, Sy = S["E"][0], S["E"][1]
    if wave == "forward":
        Sx, Sy = _forward_E(ap.k, KX, KY, KZ, Sx, Sy, S["H"][0], S["H"][1])
    Ex, Ey, Ez = _complete_vector(KX, KY, KZ, Sx, Sy)
    if h_source == "loaded":
        Hx, Hy = S["H"][0], S["H"][1]
        with np.errstate(divide="ignore", invalid="ignore"):
            Hz = np.where(np.abs(KZ) > 0, (KX * Hx + KY * Hy) / KZ, 0)
    else:
        Hx, Hy, Hz = _H_from_E(ap.k, KX, KY, KZ, Ex, Ey, Ez)
    return KX, KY, KZ, (Ex, Ey, Ez, Hx, Hy, Hz), x0, y0, wave


def _H_from_E(k, KX, KY, KZ, Ex, Ey, Ez):
    kap = (-KX, -KY, KZ)
    Hx = (kap[1] * Ez - kap[2] * Ey) / (k * ETA0)
    Hy = (kap[2] * Ex - kap[0] * Ez) / (k * ETA0)
    Hz = (kap[0] * Ey - kap[1] * Ex) / (k * ETA0)
    return Hx, Hy, Hz


def _mode_filter(ap, KX, KY, KZ, z_max, span_x, span_y, Lx, Ly, s_max, s_roll):
    """Weights applied to the plane-wave spectrum.

    1) Band limit (Matsushima & Shimobaba 2009): a mode whose lateral walk-off
       z*tan(theta) exceeds the free part of the padded FFT window would wrap
       around (periodic images) -> removed.
    2) Smooth angular cut-off: raised-cosine roll-off between
       sin(theta) = s_max - s_roll and s_max. Near-grazing modes carry the
       1/kz singularity of Ez (and of the forward-wave operator) and are
       sampled very poorly on any finite k-grid; they hardly reach a target
       plane in front of the horn anyway. s_max=None disables the cut-off
       (evanescent modes are then kept by propagate_parallel).
    """
    W = np.ones(KX.shape)
    kzr = np.where(np.real(KZ) > 0, np.real(KZ), np.inf)
    if z_max > 0:
        free_x, free_y = Lx - span_x, Ly - span_y
        W *= (z_max * np.abs(KX) / kzr <= free_x) & (z_max * np.abs(KY) / kzr <= free_y)
    if s_max is not None:
        s = np.sqrt(KX ** 2 + KY ** 2) / ap.k
        t = np.clip((s_max - s) / s_roll, 0.0, 1.0)
        W *= 0.5 - 0.5 * np.cos(np.pi * t)
    return W


def _pad_for(span_x, span_y, Lsx, Lsy, z_max, s_max, pad_min=2, pad_max=16):
    """Smallest padding whose band limit still passes all modes up to s_max."""
    s = 0.999 if s_max is None else s_max
    walk = z_max * s / np.sqrt(1 - s ** 2)
    p = max((span_x + walk) / Lsx, (span_y + walk) / Lsy, pad_min)
    if p > pad_max:
        warnings.warn(f"required padding {p:.1f} > pad_max={pad_max}; band limit will "
                      "remove some wide-angle modes (accuracy at large angles drops)")
    return int(np.ceil(min(p, pad_max)))


def propagate_parallel(ap: ApertureField, d: float, pad: Optional[int] = None,
                       h_source: str = "E", wave: str = "auto",
                       s_max: Optional[float] = 0.95, s_roll: float = 0.1,
                       x_out: Optional[np.ndarray] = None,
                       y_out: Optional[np.ndarray] = None):
    """FFT angular-spectrum propagation over distance d along the antenna axis.

    Target plane is parallel to the aperture (untilted case), sampled on the
    aperture grid (ap.x, ap.y). Fastest method; use it whenever possible.

    pad      : zero-padding factor (None -> automatic from d and s_max)
    wave     : "forward" -> use E AND H to keep only the +z travelling wave
                            (recommended when H is loaded; == Stratton-Chu)
               "total"   -> use the tangential E only
               "auto"    -> "forward" if ap.H is available, else "total"
    h_source : "E"      -> output H derived from the (forward) E spectrum
               "loaded" -> tangential H propagated independently from ap.H
                           (cross check of E/H consistency)
    s_max, s_roll : angular cut-off, see _mode_filter
    x_out, y_out  : optional 1-D output grids in the antenna frame [m]. They
                    may extend BEYOND the exported aperture window (the beam
                    spreads); the padding is enlarged accordingly and the
                    padded FFT result is interpolated onto (y_out, x_out).
    Returns E, H  each (3, Ny, Nx) complex on the aperture grid, or
            (3, len(y_out), len(x_out)) if x_out / y_out are given.
    """
    Ny, Nx = ap.E.shape[1:]
    Lsx, Lsy = Nx * abs(ap.dx), Ny * abs(ap.dy)
    windowed = x_out is not None or y_out is not None
    if windowed:
        x_out = np.atleast_1d(np.asarray(ap.x if x_out is None else x_out, float))
        y_out = np.atleast_1d(np.asarray(ap.y if y_out is None else y_out, float))
        span_x = max(x_out.max(), ap.x.max()) - min(x_out.min(), ap.x.min()) + abs(ap.dx)
        span_y = max(y_out.max(), ap.y.max()) - min(y_out.min(), ap.y.min()) + abs(ap.dy)
    else:
        span_x, span_y = Lsx, Lsy
    if pad is None:
        pad = _pad_for(span_x, span_y, Lsx, Lsy, d, s_max)
    KX, KY, KZ, spec, x0, y0, _ = _build_spectra(ap, pad, h_source, wave)
    W = _mode_filter(ap, KX, KY, KZ, d, span_x, span_y, pad * Lsx, pad * Lsy, s_max, s_roll)
    P = np.exp(-1j * KZ * d) * W
    if not windowed:
        out = np.stack([np.fft.ifft2(A * P)[:Ny, :Nx] for A in spec])
        return out[:3], out[3:]
    # padded window is periodic: roll so that it is centred on the source window
    My, Mx = KX.shape
    sx, sy = (Mx - Nx) // 2, (My - Ny) // 2
    xg = x0 + (np.arange(Mx) - sx) * ap.dx
    yg = y0 + (np.arange(My) - sy) * ap.dy
    if (x_out.min() < xg.min() or x_out.max() > xg.max() or
            y_out.min() < yg.min() or y_out.max() > yg.max()):
        raise ValueError("output window exceeds padded FFT window; increase pad")
    Xq, Yq = np.meshgrid(x_out, y_out)
    q = np.column_stack([Yq.ravel(), Xq.ravel()])
    # crop to the bounding box of the requested points (+4 cells) before interpolating
    i0 = max(np.searchsorted(xg, x_out.min()) - 4, 0); i1 = min(np.searchsorted(xg, x_out.max()) + 4, Mx)
    j0 = max(np.searchsorted(yg, y_out.min()) - 4, 0); j1 = min(np.searchsorted(yg, y_out.max()) + 4, My)
    xc, yc = xg[i0:i1], yg[j0:j1]
    out = []
    for A in spec:
        F = np.roll(np.fft.ifft2(A * P), (sy, sx), axis=(0, 1))[j0:j1, i0:i1]
        meth = "cubic" if min(F.shape) >= 4 else "linear"
        fr = RegularGridInterpolator((yc, xc), F.real, method=meth)
        fi = RegularGridInterpolator((yc, xc), F.imag, method=meth)
        out.append((fr(q) + 1j * fi(q)).reshape(Xq.shape))
    out = np.stack(out)
    return out[:3], out[3:]


def _spectrum_points(ap, pts_a, pad=None, h_source="E", wave="auto",
                     s_max=0.95, s_roll=0.1, rel_threshold=1e-5, chunk=256):
    """Direct plane-wave summation at arbitrary antenna-frame points.

    The plane-wave expansion obtained from the padded FFT is periodic with the
    padded window, so the band limit uses the joint lateral span of the source
    window and of the target points, and the largest z' of the targets.
    """
    Ny, Nx = ap.E.shape[1:]
    Lsx, Lsy = Nx * abs(ap.dx), Ny * abs(ap.dy)
    span_x = max(pts_a[:, 0].max(), ap.x.max()) - min(pts_a[:, 0].min(), ap.x.min()) + abs(ap.dx)
    span_y = max(pts_a[:, 1].max(), ap.y.max()) - min(pts_a[:, 1].min(), ap.y.min()) + abs(ap.dy)
    z_max = float(pts_a[:, 2].max())
    if pad is None:
        pad = _pad_for(span_x, span_y, Lsx, Lsy, z_max, s_max)
    KX, KY, KZ, spec, x0, y0, wave = _build_spectra(ap, pad, h_source, wave)
    W = _mode_filter(ap, KX, KY, KZ, z_max, span_x, span_y, pad * Lsx, pad * Lsy, s_max, s_roll)
    W = W * (np.imag(KZ) == 0)                      # evanescent modes: negligible at z' > ~lambda
    Mtot = KX.size
    mag = (np.abs(spec[0]) + np.abs(spec[1])) * W
    keep = mag > rel_threshold * mag.max()
    kxs, kys, kzs = KX[keep], KY[keep], KZ[keep].real
    S = np.stack([(A * W)[keep] for A in spec]) / Mtot                             # (6, K)
    out = np.empty((pts_a.shape[0], 6), complex)
    for i in range(0, pts_a.shape[0], chunk):
        p = pts_a[i:i + chunk]
        ph = np.exp(1j * (np.outer(p[:, 0] - x0, kxs) + np.outer(p[:, 1] - y0, kys)
                          - np.outer(p[:, 2], kzs)))                                   # (n, K)
        out[i:i + chunk] = ph @ S.T
    return out[:, :3].T, out[:, 3:].T, {"pad": pad, "modes": int(keep.sum()), "wave": wave}


def _stratton_chu_points(ap, pts_a, chunk=128, src_threshold=1e-3, src_step=1):
    """Love equivalence: J = z x H, M = -z x E on z'=0, radiating in free space.
    Exact for the half space z' > 0 if the plane were infinite; truncation of
    the exported plane is the only approximation."""
    if ap.H is None:
        raise ValueError("method='stratton_chu' needs the aperture H-field (h_file)")
    k = ap.k
    st = int(src_step)        # source decimation; keep spacing <= ~lambda/6
    X, Y = np.meshgrid(ap.x[::st], ap.y[::st])
    E, H = ap.E[:, ::st, ::st], ap.H[:, ::st, ::st]
    # z x H = (-Hy, Hx, 0),   -z x E = (Ey, -Ex, 0)
    J = np.stack([-H[1], H[0], np.zeros_like(H[0])]).reshape(3, -1)
    M = np.stack([E[1], -E[0], np.zeros_like(E[0])]).reshape(3, -1)
    w = np.abs(J[0]) + np.abs(J[1]) + (np.abs(M[0]) + np.abs(M[1])) / ETA0
    keep = w > src_threshold * w.max()
    dS = abs(ap.dx * ap.dy) * st * st
    J, M = J[:, keep] * dS, M[:, keep] * dS
    sx, sy = X.ravel()[keep], Y.ravel()[keep]
    Eo = np.empty((3, pts_a.shape[0]), complex)
    Ho = np.empty((3, pts_a.shape[0]), complex)
    for i in range(0, pts_a.shape[0], chunk):
        p = pts_a[i:i + chunk]
        Rx = p[:, 0:1] - sx[None, :]
        Ry = p[:, 1:2] - sy[None, :]
        Rz = np.broadcast_to(p[:, 2:3], Rx.shape)
        R = np.sqrt(Rx ** 2 + Ry ** 2 + Rz ** 2)
        ux, uy, uz = Rx / R, Ry / R, Rz / R
        kR = k * R
        G = np.exp(-1j * kR) / (4 * np.pi * R)
        a = 1 - 1j / kR - 1 / kR ** 2
        b = 1 - 3j / kR - 3 / kR ** 2
        g = (1j * k + 1 / R) * G
        # J, M only have x, y components
        uJ = ux * J[0] + uy * J[1]
        uM = ux * M[0] + uy * M[1]
        cE = -1j * k * ETA0 * G
        cH = -1j * k / ETA0 * G
        # E = cE[a J - b (u.J) u] - g (M x u)  ;  H = cH[a M - b (u.M) u] + g (J x u)
        MxU = (M[1] * uz, -M[0] * uz, M[0] * uy - M[1] * ux)
        JxU = (J[1] * uz, -J[0] * uz, J[0] * uy - J[1] * ux)
        Ec = (cE * (a * J[0] - b * uJ * ux) - g * MxU[0],
              cE * (a * J[1] - b * uJ * uy) - g * MxU[1],
              cE * (-b * uJ * uz) - g * MxU[2])
        Hc = (cH * (a * M[0] - b * uM * ux) + g * JxU[0],
              cH * (a * M[1] - b * uM * uy) + g * JxU[1],
              cH * (-b * uM * uz) + g * JxU[2])
        for c in range(3):
            Eo[c, i:i + chunk] = Ec[c].sum(axis=1)
            Ho[c, i:i + chunk] = Hc[c].sum(axis=1)
    return Eo, Ho, {"sources": int(keep.sum())}


def propagate_to_points(ap: ApertureField, pts_global, pose: Optional[AntennaPose] = None,
                        method: str = "spectrum", **kw):
    """Fields at arbitrary GLOBAL points for an antenna placed with `pose`.

    pts_global : (N, 3) array [m]
    pose       : AntennaPose; default = untilted antenna at ap.center
    method     : "spectrum" | "stratton_chu"
    kw         : spectrum     -> pad, wave ("auto"|"forward"|"total"), s_max, s_roll,
                                 h_source ("E"|"loaded"), rel_threshold, chunk
                 stratton_chu -> chunk, src_threshold, src_step
    Returns E, H as (3, N) complex arrays in GLOBAL components, and an info dict.
    Points behind or on the aperture plane (z' <= 0) are returned as NaN.
    """
    pose = pose or AntennaPose(t=np.asarray(ap.center, float))
    pts_g = np.atleast_2d(np.asarray(pts_global, float))
    pts_a = pose.to_antenna(pts_g)
    front = pts_a[:, 2] > 1e-9
    if not front.all():
        warnings.warn(f"{(~front).sum()} target points lie behind/on the aperture plane -> NaN")
    E = np.full((3, len(pts_g)), np.nan + 0j)
    H = np.full((3, len(pts_g)), np.nan + 0j)
    if method == "spectrum":
        Ea, Ha, info = _spectrum_points(ap, pts_a[front], **kw)
    elif method in ("stratton_chu", "sc"):
        Ea, Ha, info = _stratton_chu_points(ap, pts_a[front], **kw)
    else:
        raise ValueError("method must be 'spectrum' or 'stratton_chu'")
    E[:, front] = pose.R @ Ea                  # rotate vectors back to global frame
    H[:, front] = pose.R @ Ha
    info["z_antenna_range"] = (float(pts_a[front, 2].min()), float(pts_a[front, 2].max())) \
        if front.any() else None
    return E, H, info


def propagate_to_z_plane(ap: ApertureField, z: float, x: Sequence[float], y: Sequence[float],
                         pose: Optional[AntennaPose] = None, method: str = "spectrum", **kw):
    """Fields on the GLOBAL plane z = const, sampled on the grid x (Nx,), y (Ny,).

    Untilted pose and target grid == aperture grid -> use propagate_parallel()
    (identical physics, much faster). Returns E, H (3, Ny, Nx) and info.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    Xg, Yg = np.meshgrid(x, y)
    pts = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z)])
    E, H, info = propagate_to_points(ap, pts, pose, method, **kw)
    return E.reshape(3, *Xg.shape), H.reshape(3, *Xg.shape), info


def interp_plane(x, y, F, xq, yq):
    """Bilinear complex interpolation of (3, Ny, Nx) field onto grid xq, yq."""
    Xq, Yq = np.meshgrid(xq, yq)
    pts = np.column_stack([Yq.ravel(), Xq.ravel()])
    out = []
    for c in range(F.shape[0]):
        f = RegularGridInterpolator((y, x), F[c], bounds_error=False, fill_value=np.nan)
        out.append(f(pts).reshape(Xq.shape))
    return np.stack(out)


def nrmse(pred, ref, fit_complex=True):
    """Vector NRMSE after optional single complex scale fit; returns (nrmse, alpha)."""
    p, r = pred.reshape(-1), ref.reshape(-1)
    m = np.isfinite(p) & np.isfinite(r)
    p, r = p[m], r[m]
    a = (np.vdot(p, r) / np.vdot(p, p)) if fit_complex else 1.0
    return float(np.linalg.norm(a * p - r) / np.linalg.norm(r)), a



# =============================================================================
# 5) Bridge to the 2-D fw2d solver (O-mode, Ez out of plane)
# =============================================================================
# fw2d frame (X, Y, Z):  X = propagation direction into the plasma,
#                        Y = in-plane transverse direction,
#                        Z = out of plane, O-mode field Ez (parallel to B).
# Antenna frame (CST):   x' = polarisation axis (Ex dominant),
#                        y' = transverse axis lying in the fw2d plane,
#                        z' = boresight.
# For a boresight elevation `angle` (fw2d convention: 0 -> +X, >0 -> towards +Y)
#     R = [p | t | b],  p = (0, 0, -1), t = (-sin a, cos a, 0), b = (cos a, sin a, 0)
# (right handed: p x t = b).  The fw2d plane is the cut x' = 0 of the 3-D
# horn field, i.e. the horn's H-plane for an Ex-polarised horn, and
# Ez_fw2d = -Ex'.  The 2-D solver then continues this cut as a 2-D beam.

def fw2d_pose(ant_x: float, ant_y: float, angle_deg: float,
              aperture_offset: float = 0.0) -> AntennaPose:
    """AntennaPose (antenna/CST frame -> fw2d frame).

    ant_x, ant_y    : antenna reference point in fw2d coordinates [m]
    angle_deg       : boresight elevation from +X towards +Y [deg]
    aperture_offset : distance from the reference point to the aperture
                      centre along the boresight [m] (0 if ant_x, ant_y already
                      denote the aperture centre)
    """
    a = np.deg2rad(angle_deg)
    b = np.array([np.cos(a), np.sin(a), 0.0])
    t = np.array([-np.sin(a), np.cos(a), 0.0])
    p = np.array([0.0, 0.0, -1.0])
    R = np.column_stack([p, t, b])
    c = np.array([ant_x, ant_y, 0.0]) + aperture_offset * b
    return AntennaPose(R=R, t=c)


def aperture_to_fw2d_line(ap: ApertureField, x_line: float, y_line: np.ndarray,
                          ant_x: float, ant_y: float, angle_deg: float = 0.0,
                          aperture_offset: float = 0.0, field_scale="cst",
                          E_ref: Optional[float] = None, angle_tol_deg: float = 1e-6,
                          **kw):
    """Ez of the propagated horn field on the fw2d source line X = x_line.

    angle == 0 -> propagate_parallel (FFT, output window = source line)
    angle != 0 -> propagate_to_points (tilted plane-wave summation)

    field_scale : "cst"           keep the exported CST amplitude [V/m]
                  "aperture_peak" scale so that max|E| on the aperture = E_ref
                                  (same meaning as E1 of the Balanis model)
                  "line_peak"     scale so that max|Ez| on the line = E_ref
                  float           multiply the CST field by this factor
    kw          : passed to the propagator (pad, wave, s_max, method, ...)

    Returns ampl (N,), phase (N,) in (-pi, pi], Ez (N,) complex, info dict.
    phase follows the fw2d source convention  ampl*cos(wt + phase).
    """
    y_line = np.asarray(y_line, float)
    pose = fw2d_pose(ant_x, ant_y, angle_deg, aperture_offset)
    c = pose.t
    if abs(angle_deg) < angle_tol_deg:
        d = x_line - c[0]
        if d <= 0:
            raise ValueError(f"source line X={x_line} lies behind the aperture (X_ap={c[0]})")
        kw.pop("method", None)
        E, _ = propagate_parallel(ap, d, x_out=np.array([0.0]), y_out=y_line - c[1], **kw)
        Ez = -E[0][:, 0]                                   # Ez_fw2d = -Ex'
        info = {"method": "propagate_parallel", "distance": d}
    else:
        pts = np.column_stack([np.full_like(y_line, x_line), y_line, np.zeros_like(y_line)])
        E, _, info = propagate_to_points(ap, pts, pose, **kw)
        Ez = E[2]
        info["method"] = "propagate_to_points/" + kw.get("method", "spectrum")
        bad = ~np.isfinite(Ez)
        if bad.any():
            warnings.warn(f"{bad.sum()} source-line points are behind the tilted aperture -> 0")
            Ez = np.where(bad, 0.0, Ez)
    if isinstance(field_scale, (int, float)) and not isinstance(field_scale, bool):
        Ez = Ez * float(field_scale)
    elif field_scale == "aperture_peak":
        Ez = Ez * (E_ref / np.abs(ap.E).max())
    elif field_scale == "line_peak":
        Ez = Ez * (E_ref / np.abs(Ez).max())
    elif field_scale != "cst":
        raise ValueError("field_scale must be 'cst', 'aperture_peak', 'line_peak' or a float")
    b = pose.boresight
    info["boresight_hit_y"] = float(c[1] + (x_line - c[0]) * b[1] / b[0])
    return np.abs(Ez), np.angle(Ez), Ez, info


_APERTURE_CACHE = {}


def load_aperture_cached(e_file, freq, h_file=None, center_xy=(0.0, 0.0)):
    """load_aperture with an in-memory cache (CST text files load slowly)."""
    key = (str(e_file), str(h_file), float(freq), tuple(center_xy) if center_xy else None)
    if key not in _APERTURE_CACHE:
        _APERTURE_CACHE[key] = load_aperture(e_file, freq, h_file=h_file, center_xy=center_xy)
    return _APERTURE_CACHE[key]


def select_cst_export(exports: dict, freq: float, rel_tol: float = 5e-3):
    """Pick the CST export for `freq` from {freq_Hz: (e_file, h_file), ...}.

    Raises if no export lies within rel_tol: an aperture field is valid only
    at the frequency it was simulated at.
    """
    f_best = min(exports, key=lambda f: abs(f - freq))
    if abs(f_best - freq) > rel_tol * freq:
        raise ValueError(f"no CST aperture export near {freq/1e9:.3f} GHz "
                         f"(available: {[f/1e9 for f in sorted(exports)]} GHz)")
    return f_best, exports[f_best]


# =============================================================================
# Example
# =============================================================================
if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(description="Propagate a CST aperture E/H export")
    pa.add_argument("e_file"); pa.add_argument("h_file")
    pa.add_argument("--freq", type=float, default=29e9)
    pa.add_argument("--z", type=float, default=158.0, help="target plane z [mm]")
    pa.add_argument("--rx", type=float, default=0.0, help="tilt about x [deg]")
    pa.add_argument("--ry", type=float, default=10.0, help="tilt about y [deg]")
    pa.add_argument("--rz", type=float, default=0.0, help="rotation about z [deg]")
    pa.add_argument("--half", type=float, default=50.0, help="target half width [mm]")
    pa.add_argument("--n", type=int, default=101, help="target points per axis")
    pa.add_argument("--out", default="propagated_field.npz")
    a = pa.parse_args()

    ap = load_aperture(a.e_file, a.freq, h_file=a.h_file)
    print(f"aperture z = {ap.center[2]*1e3:.3f} mm, grid {ap.E.shape[2]} x {ap.E.shape[1]}, "
          f"P = {ap.power_z():.4f} W")
    pose = AntennaPose.from_tilt(ap.center, rx_deg=a.rx, ry_deg=a.ry, rz_deg=a.rz)
    xq = np.linspace(-a.half, a.half, a.n) * 1e-3
    E, H, info = propagate_to_z_plane(ap, a.z * 1e-3, xq, xq, pose)
    hit = pose.boresight_hit_z(a.z * 1e-3)
    Sz = 0.5 * np.real(E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    print(info)
    print("boresight crosses target plane at", np.round(hit * 1e3, 2), "mm")
    print(f"power through target window: {np.nansum(Sz) * (xq[1]-xq[0])**2:.4f} W")
    np.savez(a.out, x=xq, y=xq, z=a.z * 1e-3, E=E, H=H, R=pose.R, t=pose.t)
    print("saved", a.out)
