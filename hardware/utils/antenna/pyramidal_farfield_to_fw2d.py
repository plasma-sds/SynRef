import numpy as np
from scipy.special import fresnel as _scipy_fresnel
from hardware.utils.antenna.pyramidal_horn_farfield_E_U import pyramidal_horn_E
from reflectometer.conversions import antenna_pos_to_unit
from reflectometer.conversions import meter_to_unit

# -----------------------------------------------------------------------
# BRIDGE: horn far-field → fw2d ampl_inc / phase_inc
# -----------------------------------------------------------------------
def pyramidal_farfield_to_fw2d(
    y,                # y grid of simulation space
    ny,               # number of fw2d grid points along y (0..ny)
    dy,               # fw2d grid spacing along y  [m]
    dx,               # fw2d grid spacing along x  [m]
    x_horn,           # horn x-position [m] (must be <= 0)
    antenna_pos,           # horn y-position [m] (can be <0, 0..ny, or >ny)
    yante,            
    angle,            # beam steering angle from the C code [rad]
                      #   = 0 → boresight along +x
                      #   > 0 → beam tilted toward +y
    a1, b1,           # horn aperture H- and E-plane widths [m]
    rho1, rho2,       # horn E- and H-plane slant lengths   [m]
    freq,             # frequency [Hz]
    E1=1.0
):
    """
    Sample the pyramidal horn far-field at each fw2d antenna-plane grid point
    j = 0 .. ny, using the full 2-D geometry of horn placement.

    Geometry
    --------
    The fw2d antenna plane is the line x = 0, with grid points at y = j * dy.
    The horn is located at physical position:

        x_horn_phys = x_horn * dx   (must be negative or zero, i.e. left of the fw2d frame)
        y_horn_phys = y_horn * dy   (can be below/above the fw2d frame)

    The horn boresight direction is defined by 'angle' (same convention as
    the original C code):
        boresight = (cos(angle), sin(angle))   in (x, y) physical space

    For each grid point j the code:
      1. Computes the vector from the horn to that grid point.
      2. Resolves it into components ALONG and PERPENDICULAR to the boresight.
      3. Derives the off-boresight angle  theta_j  seen by the horn.
      4. Evaluates the complex horn far-field F(theta_j).
      5. Adds the free-space propagation phase  -k*(R_j - R_ref)
         so the phase ramp across j correctly reflects path-length differences.
      6. Normalises amplitude to 1 at the reference point.

    Parameters
    ----------
    ny, yante : fw2d grid parameters (same as C code)
    dy, dx    : physical grid spacings [m]
    x_horn    : horn x-position in grid indices  (e.g. -30 → 30 cells left)
    y_horn    : horn y-position in grid indices  (e.g. -20 → 20 cells below)
    angle     : beam steering angle [rad]  (matches C code 'angle' variable)
    a1, b1    : horn aperture dimensions [m]
    rho1,rho2 : horn slant lengths [m]
    freq      : frequency [Hz]
    E1        : aperture excitation (normalised out)

    Returns
    -------
    ampl_inc  : ndarray shape (ny+1,)   normalised amplitude  [0, 1]
    phase_inc : ndarray shape (ny+1,)   phase [rad] in (-pi, pi]
    """
    if x_horn > 0:
        raise ValueError(f"x_horn = {x_horn} > 0: horn must be left of the fw2d frame.")
    
    k   = 2.0 * np.pi * freq / 3e8
    j_arr = np.arange(ny + 1, dtype=float)

    j_ref_ind = meter_to_unit(antenna_pos, dx) - meter_to_unit(x_horn, dx) * np.tan(angle)
    j_ref_m   = antenna_pos - x_horn * np.tan(angle)

    # Vector from horn to each grid point j
    vec_x_ind = 0.0 - meter_to_unit(x_horn, dx)                    # same for all j (scalar)
    vec_y_ind = j_arr - meter_to_unit(antenna_pos, dx)             # varies with j

    vec_x_m = 0.0 - x_horn
    vec_y_m = j_arr * dx - antenna_pos

    # Distance from horn to each grid point j
    R_j_ind = np.sqrt(vec_x_ind**2 + vec_y_ind**2)
    R_j_m = np.sqrt(vec_x_m**2 + vec_y_m**2)

    # -------------------------------------------------------------------
    # Reference point: grid point at j ref (beam centre)
    # -------------------------------------------------------------------
    vec_x_ref_ind = vec_x_ind
    vec_y_ref_ind = j_ref_ind - meter_to_unit(antenna_pos, dx)

    vec_x_ref_m = vec_x_m
    vec_y_ref_m = j_ref_m - antenna_pos


    R_ref_ind     = np.sqrt(vec_x_ref_ind**2 + vec_y_ref_ind**2)
    R_ref_m     = np.sqrt(vec_x_ref_m**2 + vec_y_ref_m**2)

    # -------------------------------------------------------------------
    # Decompose each vec_j into boresight and perpendicular components.
    #
    # Boresight unit vector:  b = (cos(angle), sin(angle))
    # Perpendicular unit vector: p = (-sin(angle), cos(angle))
    #
    # Component along boresight (= "range" in horn coords):
    #   along_j = vec_x*cos(angle) + vec_y*sin(angle)
    #
    # Component perpendicular to boresight (= E-plane offset):
    #   perp_j  = -vec_x*sin(angle) + vec_y*cos(angle)
    #
    # Off-boresight angle:
    #   theta_j = atan2(perp_j, along_j)
    #
    # For the horn pattern phi is ALWAYS pi/2 (E-plane) because the
    # fw2d grid is 2-D and the transverse direction IS the E-plane.
    # -------------------------------------------------------------------
    along_j = vec_x_m * np.cos(angle) + vec_y_m * np.sin(angle)
    perp_j  = -vec_x_m * np.sin(angle) + vec_y_m * np.cos(angle)

    theta_j = np.arctan2(np.abs(perp_j), along_j)   # polar, 0..pi
    sign_j  = np.sign(perp_j)                        # which side of boresight

    # phi_j encodes the side: +perp → phi=pi/2, -perp → phi=3pi/2 (or -pi/2)
    # For a symmetric horn the amplitude is the same on both sides,
    # but the Fresnel phase is symmetric too, so using |perp| + tracking sign
    # in the propagation phase is sufficient.
    phi_j = np.where(sign_j >= 0, np.pi / 2.0, -np.pi / 2.0)

    # -------------------------------------------------------------------
    # Evaluate complex horn far-field at each (theta_j, phi_j)
    # -------------------------------------------------------------------
    E_theta, E_phi, _ = pyramidal_horn_E(
        theta_j, phi_j, a1, b1, rho1, rho2, freq, E1=E1, r=R_j_m
    )

    # -------------------------------------------------------------------
    # Choose the dominant field component.
    #
    # The horn radiates a linearly polarised field.  On the E-plane cut
    # (phi = ±pi/2), the dominant component is E_theta.
    # On the H-plane cut (phi = 0 or pi), it would be E_phi.
    #
    # Since the fw2d grid is 2-D in the (x, y) plane and the transverse
    # direction is always the y-axis (= E-plane of the horn as placed),
    # E_theta is the physically relevant component here.
    #
    # Concretely:
    #   E_theta ∝ (1+cosθ) * sin(phi) * I1 * I2
    #   E_phi   ∝ (1+cosθ) * cos(phi) * I1 * I2
    #
    # At phi = ±pi/2:  sin(phi) = ±1, cos(phi) = 0
    #   → E_theta is NON-ZERO, E_phi is ZERO.
    #
    # At phi = 0:       sin(phi) = 0,  cos(phi) = 1
    #   → E_theta is ZERO,  E_phi is NON-ZERO.
    #
    # So: always use E_theta for an E-plane cut (phi=±pi/2), which is
    # the correct choice for a 2-D simulation where the horn is oriented
    # with its E-plane parallel to the fw2d y-axis.
    # -------------------------------------------------------------------
    F_complex = E_theta

    # -------------------------------------------------------------------
    # Add free-space propagation phase  -k*(R_j - R_ref)
    #
    # The horn far-field formula gives the PATTERN phase (Fresnel terms).
    # But as the wave travels from horn to grid point j it also accumulates
    # -k*R_j worth of phase (the e^{-jkr} propagation factor).
    # We factor out the reference phase -k*R_ref so that:
    #   - The reference grid point (j = yante) has zero relative propagation phase.
    #   - Points closer to the horn get positive relative phase.
    #   - Points farther from the horn get negative relative phase.
    # This reproduces the effect of the original linear phase ramp but now
    # correctly accounts for the curved (spherical) phase fronts of the horn.
    # -------------------------------------------------------------------
    prop_phase = -k * (R_j_m - R_ref_m)
    F_complex  = F_complex * np.exp(1j * prop_phase)

    # -------------------------------------------------------------------
    # Normalise: divide by the value at the reference point j_ref
    # -------------------------------------------------------------------
    max_amp = np.max(np.abs(F_complex))
    if max_amp < 1e-30:
        raise ValueError("Pattern is zero everywhere — check horn parameters.")
    F_norm = F_complex / max_amp 

    ampl_inc  = np.abs(E_theta)
    phase_inc = np.angle(E_theta)          # wrapped to (-pi, pi] automatically

    return ampl_inc, phase_inc


"""import matplotlib.pyplot as plt
y =  np.arange(0,400,1) * 0.001
freq = 10.0e9
lam  = 3e8 / freq
a1   = 12.0 * lam
b1   =  6.0 * lam
rho1 =  6.0 * lam
rho2 =  6.0 * lam

ny    = 300
yante = 150
dy    = 0.5  * lam
dx    = 0.5  * lam

x_horn = -80
y_horn = 150
angle_rad = np.radians(40.0)

ampl, phase = pyramidal_farfield_to_fw2d(
            y, ny, dy, dx, x_horn, y_horn, angle_rad,
            a1, b1, rho1, rho2, freq)

# -----------------------------------------------------------------------
# Self-test and visualisation
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    freq = 10.0e9
    lam  = 3e8 / freq
    a1   = 12.0 * lam
    b1   =  6.0 * lam
    rho1 =  6.0 * lam
    rho2 =  6.0 * lam

    ny    = 300
    yante = 150
    dy    = 0.5  * lam
    dx    = 0.5  * lam

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    fig.suptitle("Horn far-field → fw2d aperture: geometry variations", fontsize=13)

    cases = [
        # (x_horn, y_horn, angle_deg, label)
        ( -80,  150,   0.0, "Left -80, on-axis,\nangle=0°"),
        ( -80,  150,  40.0, "Left -80, on-axis,\nangle=+40°"),
        ( -80,  150, -40.0, "Left -80, on-axis,\nangle=-40°"),
        ( -80,   -30,   0.0, "Left -80, BELOW frame,\nangle=0°"),
        ( -80,  330,   0.0, "Left -80, ABOVE frame,\nangle=0°"),
        ( -80,   -30,  40.0, "Left -80, BELOW frame,\nangle=+40°"),
    ]

    j_arr  = np.arange(ny + 1)
    y_lam  = (j_arr - yante) * dy / lam

    for ax_row, ax_col, (x_horn, y_horn, angle_deg, label) in zip(
            [0,0,0,1,1,1], [0,1,2,0,1,2], cases):
        angle_rad = np.radians(angle_deg)
        ampl, phase = pyramidal_farfield_to_fw2d(
            ny, dy, dx, x_horn, y_horn, angle_rad,
            a1, b1, rho1, rho2, freq)

        ax1 = axes[ax_row, ax_col]
        color_a = "steelblue"
        color_p = "tomato"
        l1, = ax1.plot(y_lam, ampl, color=color_a, lw=1.8)
        ax1.set_ylabel("Amplitude", color=color_a, fontsize=9)
        ax1.tick_params(axis='y', labelcolor=color_a)
        ax1.set_ylim(0, 1.05)

        ax2 = ax1.twinx()
        l2, = ax2.plot(y_lam, np.degrees(phase), color=color_p, lw=1.4, ls='--')
        ax2.set_ylabel("Phase [°]", color=color_p, fontsize=9)
        ax2.tick_params(axis='y', labelcolor=color_p)

        ax1.set_title(label, fontsize=9)
        if ax_row == 1:
            ax1.set_xlabel("y offset [λ]", fontsize=9)

    plt.tight_layout()
    plt.savefig("horn_aperture_v4.png", dpi=150)
    print("Saved horn_aperture_v4.png")"""
