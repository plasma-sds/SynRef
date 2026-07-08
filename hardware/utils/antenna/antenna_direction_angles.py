import numpy as np

def antenna_look_angles(
    ant_pos,       # (3,) array  — antenna phase centre in lab frame [m]
    boresight,     # (3,) array  — antenna boresight direction (need not be unit)
    e_plane_up,    # (3,) array  — E-plane "up" direction  (need not be unit)
    src_pos,       # (3,) or (N,3) array — source / wavefront point(s) in lab frame [m]
):
    """
    Convert lab-frame geometry to (theta, phi) in the antenna's boresight frame.

    Convention (matches Balanis Ch.13 / pyramidal_horn_E):
        theta = 0       : boresight
        phi   = 0°      : H-plane  (boresight x e_plane_up)
        phi   = 90°     : E-plane  (e_plane_up direction)

    Parameters
    ----------
    ant_pos    : array-like (3,)      antenna phase centre [m]
    boresight  : array-like (3,)      boresight direction (any length)
    e_plane_up : array-like (3,)      E-plane polarisation / "up" direction
    src_pos    : array-like (3,) or (N,3)
                                      incoming wavefront source position(s) [m]

    Returns
    -------
    theta : ndarray [rad]   polar angle from boresight,  0 <= theta <= pi
    phi   : ndarray [rad]   azimuth in antenna frame,   -pi < phi <= pi
                            (use np.degrees() to convert to degrees)
    """
    ant_pos   = np.asarray(ant_pos,   dtype=float)
    boresight = np.asarray(boresight, dtype=float)
    e_up      = np.asarray(e_plane_up, dtype=float)
    src_pos   = np.asarray(src_pos,   dtype=float)

    # --- Build orthonormal local frame ---
    z_loc = boresight / np.linalg.norm(boresight)   # boresight  (+z local)
    # orthogonalise e_up against z_loc (Gram-Schmidt)
    e_up_orth = e_up - np.dot(e_up, z_loc) * z_loc
    if np.linalg.norm(e_up_orth) < 1e-10:
        raise ValueError("e_plane_up is parallel to boresight — choose a different vector.")
    y_loc = e_up_orth / np.linalg.norm(e_up_orth)  # E-plane up (+y local)
    x_loc = np.cross(y_loc, z_loc)                 # H-plane    (+x local)
    # x_loc is already unit length because y_loc and z_loc are orthonormal

    # --- Direction vector(s) from antenna to source ---
    if src_pos.ndim == 1:
        delta = src_pos - ant_pos                   # (3,)
    else:
        delta = src_pos - ant_pos[np.newaxis, :]    # (N,3)

    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Source position coincides with antenna phase centre.")
    r_hat = delta / norm                            # unit vector(s) to source

    # --- Project onto local frame ---
    if r_hat.ndim == 1:
        rx = np.dot(r_hat, x_loc)
        ry = np.dot(r_hat, y_loc)
        rz = np.dot(r_hat, z_loc)
    else:
        rx = r_hat @ x_loc
        ry = r_hat @ y_loc
        rz = r_hat @ z_loc

    # --- Spherical angles ---
    rz_clipped = np.clip(rz, -1.0, 1.0)   # guard against floating-point >1
    theta = np.arccos(rz_clipped)          # [0, pi]
    phi   = np.arctan2(ry, rx)             # (-pi, pi]

    return theta, phi