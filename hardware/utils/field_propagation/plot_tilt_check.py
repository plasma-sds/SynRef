"""
plot_tilt_check.py
Compare |Ex| on the global plane z = 158 mm:
  panel 1 : CST reference (untilted horn)
  panel 2-4: aperture field from z = 52 mm propagated with horn tilts ry = 0, 10, 20 deg
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                     # remove this line to show the figure interactively
import matplotlib.pyplot as plt

from horn_near_field import (load_aperture, load_field_txt,
                             AntennaPose, propagate_to_z_plane)

# --- paths (adjust parents[N] so it points to your repo root) --------------
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPORT_DIR = REPO_ROOT / "hardware/utils/antenna/field_exports/W7X_PCR"
E52 = EXPORT_DIR / "output_E_52.txt"
H52 = EXPORT_DIR / "output_H_52.txt"
E158 = EXPORT_DIR / "output_E_158.txt"

FREQ = 29e9
Z_TARGET = 0.158                          # [m] global target plane
TILTS_RY = (0, 10, 40)                    # [deg] tilt about the global y axis
HALF = 0.065                              # [m] half width of the target grid
N = 131                                   # target points per axis

# --- load data ---------------------------------------------------------------
ap = load_aperture(E52, FREQ, h_file=H52)          # aperture plane, antenna frame
ref = load_field_txt(E158)                          # CST reference at z = 158 mm

xq = np.linspace(-HALF, HALF, N)

# --- plot ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 1 + len(TILTS_RY), figsize=(17, 4.3),
                       constrained_layout=True)

ax[0].pcolormesh(ref.x * 1e3, ref.y * 1e3, np.abs(ref.F[0]), shading="auto")
ax[0].set_title(f"CST, z={ref.z*1e3:.0f} mm, untilted")

for i, ry in enumerate(TILTS_RY, start=1):
    pose = AntennaPose.from_tilt(ap.center, ry_deg=ry)     # rotate about aperture centre
    E, H, info = propagate_to_z_plane(ap, Z_TARGET, xq, xq, pose)   # E: (3, Ny, Nx)

    ax[i].pcolormesh(xq * 1e3, xq * 1e3, np.abs(E[0]), shading="auto")
    hit = pose.boresight_hit_z(Z_TARGET) * 1e3
    ax[i].plot(hit[0], hit[1], "w+", ms=12)
    ax[i].set_title(f"propagated from z={ap.center[2]*1e3:.0f} mm, ry={ry}°")
    print(f"ry={ry:>3} deg  {info}  boresight hit = {np.round(hit, 1)} mm")

for a in ax:
    a.set_aspect("equal")
    a.set_xlabel("x [mm]")
    a.set_ylabel("y [mm]")

fig.suptitle(f"|Ex| [V/m] on the global plane z = {Z_TARGET*1e3:.0f} mm "
             "(+ = boresight intersection)")
plt.savefig("tilt_check_v2.png", dpi=110)
# plt.show()