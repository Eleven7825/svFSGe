"""
Visualise the interface Gaussian filter kernel on the inner-wall nodes.

Shows kernel weights for representative nodes (centre, quarter, near-end,
boundary) as scatter plots on the unrolled cylinder (s = r*theta vs z).
Overlays 1-sigma and 3-sigma ellipses so it is easy to verify that the
filter spans approximately the desired number of neighbours.

Usage:
    python visualize_interface_kernel.py [interface.vtp] [--l_s LS] [--l_z LZ]

Defaults to the mesh in the most recent partition found in /svFSGe.
"""

import sys, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n

from filter_interface import kernel_weights, _node_cylindrical

# ── locate interface mesh ────────────────────────────────────────────────────
if len(sys.argv) > 1 and sys.argv[1].endswith(".vtp"):
    vtp_path = sys.argv[1]
else:
    partitions = sorted(glob.glob("/svFSGe/partitioned_*/mesh_tube_fsi/solid/mesh-surfaces/interface.vtp"))
    if not partitions:
        raise FileNotFoundError("No interface.vtp found under /svFSGe/partitioned_*/")
    vtp_path = partitions[-1]

print(f"Using interface mesh: {vtp_path}")

r = vtk.vtkXMLPolyDataReader()
r.SetFileName(vtp_path)
r.Update()
pts_ref = v2n(r.GetOutput().GetPoints().GetData())  # (N, 3)
N = len(pts_ref)

# ── filter parameters ────────────────────────────────────────────────────────
params = {
    "l_s":    0.127,   # circumferential length [cm]  ≈ 1 element width  → 3σ spans 3 el
    "l_z":    0.75,    # axial length           [cm]  ≈ 1 element height → 3σ spans 3 el
    "beta":   0.5,
    "t_nl":   1.0,
    "lo":     15.0,
    "curve":  0.0,
    "ro_tol": 0.01,
}

# parse any command-line overrides
for arg in sys.argv[1:]:
    if arg.startswith("--l_s="):  params["l_s"]  = float(arg.split("=")[1])
    if arg.startswith("--l_z="):  params["l_z"]  = float(arg.split("=")[1])

# ── cylindrical coordinates of all nodes ────────────────────────────────────
theta, z, ro = _node_cylindrical(pts_ref, params["lo"], params["curve"])
ro_mean = ro.mean()
s_arr   = ro_mean * theta   # unrolled arc coordinate

# ── pick representative nodes (inner layer) ──────────────────────────────────
inner = np.where(np.abs(ro - ro.mean()) < params["ro_tol"])[0]
targets = [
    (7.5,  "centre  z≈7.5"),
    (3.75, "quarter z≈3.75"),
    (1.0,  "near-end z≈1.0"),
    (0.3,  "boundary z≈0.3"),
]
effect_nodes = []
for zt, lbl in targets:
    i0 = inner[np.argmin(np.abs(z[inner] - zt))]
    effect_nodes.append((i0, lbl))

# seam node: mid-z, closest to theta = pi (or -pi)
seam_mid_z = inner[np.argmin(np.abs(z[inner] - params["lo"] / 2))]
# among nodes near mid-z, pick the one with largest |theta|
mid_z_val = z[seam_mid_z]
mid_z_nodes = inner[np.abs(z[inner] - mid_z_val) < 1e-6]
i_seam = mid_z_nodes[np.argmax(np.abs(theta[mid_z_nodes]))]
effect_nodes.append((i_seam, f"seam  θ≈{theta[i_seam]:.2f} rad  z≈{z[i_seam]:.1f}"))

# ── element size estimates (for neighbour count annotation) ──────────────────
z_uniq = np.unique(np.round(z[inner], 4))
h_z_est = np.diff(z_uniq).mean() if len(z_uniq) > 1 else params["l_z"]
theta_uniq = np.unique(np.round(theta[inner], 5))
h_s_est = ro_mean * np.diff(theta_uniq).mean() if len(theta_uniq) > 1 else params["l_s"]

print(f"Interface: {N} nodes,  h_z≈{h_z_est:.3f} cm,  h_s≈{h_s_est:.3f} cm")
print(f"Filter:    l_s={params['l_s']} cm ({params['l_s']/h_s_est:.1f} elements),  "
      f"l_z={params['l_z']} cm ({params['l_z']/h_z_est:.1f} elements)")

# ── Figure 1: kernel weights for 4 representative nodes ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)
fig.suptitle(
    f"Interface Gaussian filter kernel  "
    f"(l_s={params['l_s']} cm={params['l_s']/h_s_est:.1f}el, "
    f"l_z={params['l_z']} cm={params['l_z']/h_z_est:.1f}el)  —  unrolled cylinder",
    fontsize=12)

cmap = plt.cm.YlOrRd

for ax, (i0, label) in zip(axes.flat, effect_nodes):
    w, eff_ls, eff_lz, gam = kernel_weights(i0, pts_ref, params)

    vmax = w.max()
    norm = Normalize(vmin=0, vmax=vmax)

    # scatter: all inner-layer nodes coloured by weight
    mask = np.abs(ro - ro_mean) < params["ro_tol"]
    colors = [cmap(norm(w[j])) if w[j] > 1e-8 else (0.88, 0.88, 0.88, 1.0)
              for j in np.where(mask)[0]]
    sc = ax.scatter(z[mask], s_arr[mask], c=colors, s=40, zorder=2)

    # highlight the effect node
    ax.scatter(z[i0], s_arr[i0], s=120, facecolors="none",
               edgecolors="blue", linewidths=2.0, zorder=3, label="effect node")

    # 1σ and 3σ ellipses
    t_ang = np.linspace(0, 2 * np.pi, 300)
    for nsig, ls_, color in [(1, "-", "navy"), (3, "--", "steelblue")]:
        ez = z[i0]    + nsig * eff_lz * np.cos(t_ang)
        es = s_arr[i0] + nsig * eff_ls * np.sin(t_ang)
        ax.plot(ez, es, ls_, color=color, linewidth=1.5,
                label=f"{nsig}σ ellipse")

    ax.set_xlim(-0.3, params["lo"] + 0.3)
    ax.set_ylim(-ro_mean * np.pi - 0.3, ro_mean * np.pi + 0.3)
    ax.set_xlabel("z  (axial, cm)")
    ax.set_ylabel("s = r·θ  (circumferential arc, cm)")

    n_nb = (w > 1e-6).sum()
    ax.set_title(
        f"{label}\n"
        f"γ={gam:.3f}  eff_ls={eff_ls:.3f} cm  eff_lz={eff_lz:.3f} cm  "
        f"neighbours={n_nb}",
        fontsize=9)
    ax.legend(fontsize=7, loc="upper right")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="normalised weight", shrink=0.7)

out1 = os.path.join(os.path.dirname(vtp_path.split("partitioned_")[0]),
                    "svFSGe/kernel_visualization.png") \
       if "partitioned_" in vtp_path else "/svFSGe/kernel_visualization.png"
out1 = "/svFSGe/kernel_visualization.png"
fig.savefig(out1, dpi=150)
print(f"Saved: {out1}")

# ── Figure 2: gamma correction profile ───────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 3), constrained_layout=True)
z_line = np.linspace(0, params["lo"], 500)
from filter_interface import _gamma
gam_line = [_gamma(min(z_, params["lo"] - z_), params["l_z"],
                   params["beta"], params["t_nl"]) for z_ in z_line]
ax2.plot(z_line, gam_line, "b-", linewidth=2, label="γ(z)")
ax2.axhline(params["beta"], color="r", linestyle="--",
            label=f"β={params['beta']} (boundary minimum)")
ax2.axhline(1.0, color="g", linestyle="--", label="γ=1 (interior)")
ax2.fill_between(z_line, 0, gam_line, alpha=0.12)
ax2.set_xlabel("z  (axial position, cm)")
ax2.set_ylabel("γ(z)  (boundary correction)")
ax2.set_title(f"Boundary correction profile  —  "
              f"t_nl={params['t_nl']}, β={params['beta']}, l_z={params['l_z']} cm")
ax2.legend()
ax2.grid(True, alpha=0.3)
out2 = "/svFSGe/gamma_profile.png"
fig2.savefig(out2, dpi=150)
print(f"Saved: {out2}")

plt.show()
