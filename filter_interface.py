"""
Boundary-corrected Gaussian spatial filter for the FSI interface displacement.

Mirrors the kernel from the C++ gr_nonlocal.cpp (Appendix C), applied here at
the nodal level on the inner-wall interface mesh instead of at element Gauss
points.  All length inputs are in the same physical units as the mesh (cm).

Parameters (passed as a dict):
    l_s    : circumferential characteristic length  [cm]
    l_z    : axial characteristic length            [cm]
    beta   : boundary floor factor (gamma at wall)  [–]  default 0.5
    t_nl   : transition parameter                   [–]  default 1.0
    lo     : vessel length                          [cm]
    curve  : centreline curvature parameter         [–]  default 0.0
    ro_tol : radial layer tolerance                 [cm] default 0.01
"""

import numpy as np


def _node_cylindrical(pts, lo, curve=0.0):
    """Convert Cartesian interface node coords to (theta, z, ro)."""
    N = len(pts)
    theta = np.empty(N)
    z_arr = np.empty(N)
    ro_arr = np.empty(N)
    for i in range(N):
        X = pts[i]
        xcl = curve / 2.0 * (1.0 - np.cos(2.0 * np.pi * X[2] / lo))
        nx = X[0] - xcl
        ny = X[1]
        ro = np.hypot(nx, ny)
        ro_arr[i] = ro
        theta[i] = np.arctan2(nx / max(ro, 1e-14), -ny / max(ro, 1e-14))
        z_arr[i] = X[2]
    return theta, z_arr, ro_arr


def _gamma(d_x, l, beta, t_nl):
    """Boundary correction factor (Eq. C.5)."""
    tl = t_nl * l
    if tl < 1e-30:
        return beta
    return 1.0 if d_x >= tl else (1.0 - beta) / tl * d_x + beta


def gaussian_filter_interface(disp_int, pts_ref, params):
    """
    Apply boundary-corrected Gaussian filter to interface displacement.

    Parameters
    ----------
    disp_int : (N, 3) ndarray  — converged interface displacement
    pts_ref  : (N, 3) ndarray  — reference (undeformed) node coordinates
    params   : dict             — filter parameters (see module docstring)

    Returns
    -------
    disp_filtered : (N, 3) ndarray
    """
    l_s    = params["l_s"]
    l_z    = params["l_z"]
    beta   = params.get("beta",   0.5)
    t_nl   = params.get("t_nl",   1.0)
    lo     = params["lo"]
    curve  = params.get("curve",  0.0)
    ro_tol = params.get("ro_tol", 0.01)

    theta, z, ro = _node_cylindrical(pts_ref, lo, curve)
    N = len(disp_int)
    disp_filtered = np.empty_like(disp_int)

    for i in range(N):
        d_x    = min(z[i], lo - z[i])
        gam    = _gamma(d_x, l_z, beta, t_nl)
        eff_ls = gam * l_s
        eff_lz = gam * l_z

        w_sum = 0.0
        acc   = np.zeros(3)
        for j in range(N):
            if abs(ro[j] - ro[i]) > ro_tol:
                continue
            dz = abs(z[j] - z[i])
            if dz > 3.0 * eff_lz:
                continue
            dth = (theta[j] - theta[i] + np.pi) % (2.0 * np.pi) - np.pi
            ds  = ro[i] * abs(dth)
            if ds > 3.0 * eff_ls:
                continue
            w = np.exp(-0.5 * ((ds / eff_ls) ** 2 + (dz / eff_lz) ** 2))
            w_sum += w
            acc   += w * disp_int[j]

        disp_filtered[i] = acc / w_sum if w_sum > 0.0 else disp_int[i]

    return disp_filtered


def kernel_weights(i0, pts_ref, params):
    """
    Return the normalised kernel weight vector w (length N) for node i0.
    Useful for visualisation.
    """
    l_s    = params["l_s"]
    l_z    = params["l_z"]
    beta   = params.get("beta",   0.5)
    t_nl   = params.get("t_nl",   1.0)
    lo     = params["lo"]
    curve  = params.get("curve",  0.0)
    ro_tol = params.get("ro_tol", 0.01)

    theta, z, ro = _node_cylindrical(pts_ref, lo, curve)
    N = len(pts_ref)

    d_x    = min(z[i0], lo - z[i0])
    gam    = _gamma(d_x, l_z, beta, t_nl)
    eff_ls = gam * l_s
    eff_lz = gam * l_z

    w = np.zeros(N)
    for j in range(N):
        if abs(ro[j] - ro[i0]) > ro_tol:
            continue
        dz  = abs(z[j] - z[i0])
        dth = (theta[j] - theta[i0] + np.pi) % (2.0 * np.pi) - np.pi
        ds  = ro[i0] * abs(dth)
        d2  = (ds / eff_ls) ** 2 + (dz / eff_lz) ** 2
        if d2 <= 9.0:
            w[j] = np.exp(-0.5 * d2)

    s = w.sum()
    return w / s if s > 0 else w, eff_ls, eff_lz, gam
