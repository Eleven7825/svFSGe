#!/usr/bin/env python
# coding=utf-8
"""
Post-processing script for pulsatile fluid simulation
Extracts pressure, velocity magnitude, and WSS for the last cardiac cycle
Time-averages all quantities over the cycle
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from vtk.util.numpy_support import vtk_to_numpy as v2n

from vtk_functions import read_geo

# use same matplotlib settings as post.py
plt.rcParams.update({"text.usetex": False})

# field descriptions (same as post.py)
f_labels = {
    "pressure": ["Pressure [mmHg]"],
    "velocity": ["Velocity $u$ [mm/s]"],
    "wss": ["WSS [dyne/cm²]"],
}
f_comp = {key: len(value) for key, value in f_labels.items()}
f_scales = {"pressure": [1.0/0.1333], "wss": [10000.0]}  # pressure: kg/(mm·s²) to mmHg, WSS: kg/(mm·s²) to dyne/cm²

fields = ["pressure", "velocity", "wss"]


def xyz2cra(xyz):
    """Convert Cartesian to cylindrical coordinates (same as post.py)"""
    return np.array([
        (np.arctan2(xyz[0], xyz[1]) + 2.0 * np.pi) % (2.0 * np.pi),
        np.sqrt(xyz[0] ** 2.0 + xyz[1] ** 2.0),
        xyz[2],
    ])


def get_ids(pts_xyz, domain="fluid"):
    """Get point IDs at specific locations (same as post.py for fluid)"""
    # coordinates of all points (in reference configuration)
    pts_cra = xyz2cra(pts_xyz.T).T

    # cylinder dimensions
    ro = np.max(pts_cra[:, 1])
    ri = np.min(pts_cra[:, 1])
    h = np.max(pts_cra[:, 2])

    if domain == "fluid":
        # plot along centerline
        p_cir = {0: 0.0}
        p_rad = {"center": 0.0}
        p_axi = {"start": 0.0, "mid": h / 2, "end": h}
    elif domain == "interface":
        # plot at interface (OUTER wall where WSS is defined)
        p_cir = {0: 0.0, 3: 0.5 * np.pi, 6: np.pi, 9: 1.5 * np.pi}
        p_rad = {"wall": ro}  # WSS is at outer wall, not inner!
        p_axi = {"start": 0.0, "mid": h / 2, "end": h}
    else:
        raise ValueError("Unknown domain: " + domain)

    # collect all point coordinates
    locations = {}
    for cn, cp in p_cir.items():
        for rn, rp in p_rad.items():
            for an, ap in p_axi.items():
                identifiers = [
                    (cn, rn, an),
                    (":", rn, an),
                    (cn, ":", an),
                    (cn, rn, ":"),
                ]
                for i in identifiers:
                    locations[i] = [cp, rp, ap]

    # collect all point ids
    ids = {}
    coords = {}
    for loc, pt in locations.items():
        chk = [np.isclose(pts_cra[:, i], pt[i]) for i in range(3) if loc[i] != ":"]
        ids[loc] = np.where(np.logical_and.reduce(np.array(chk)))[0]
        if len(ids[loc]) == 0:
            print("no points found: " + str(loc))
            continue

        # sort according to coordinate
        if ":" in loc:
            dim = list(loc).index(":")
            crd = pts_cra[ids[loc], dim]
            sort = np.argsort(crd)
            ids[loc] = ids[loc][sort]
            coords[loc] = crd[sort]
            assert len(np.unique(crd)) == len(crd), "coordinates not unique: " + str(crd)

    return ids, coords


def extract_results_fluid_centerline(post, res, pts, ids):
    """Extract fluid results at centerline (same as post.py)"""
    pressure = v2n(res.GetPointData().GetArray("Pressure"))
    velocity = v2n(res.GetPointData().GetArray("Velocity"))
    velocity = np.linalg.norm(velocity, axis=1)

    for loc, pt in ids.items():
        post[loc]["pressure"] += [pressure[pt]]
        post[loc]["velocity"] += [velocity[pt]]


def extract_results_wall(post, res, pts, ids):
    """Extract pressure at wall interface"""
    pressure = v2n(res.GetPointData().GetArray("Pressure"))

    for loc, pt in ids.items():
        post[loc]["pressure"] += [pressure[pt]]


def extract_wss(post, res, pts, ids):
    """Extract wall shear stress at interface"""
    # Check available arrays
    if res.GetPointData().HasArray("WSS"):
        wss = v2n(res.GetPointData().GetArray("WSS"))
        wss = np.linalg.norm(wss, axis=1)
    elif res.GetPointData().HasArray("Traction"):
        # Traction might contain WSS
        traction = v2n(res.GetPointData().GetArray("Traction"))
        wss = np.linalg.norm(traction, axis=1)
    else:
        # WSS not available, set to zero
        wss = np.zeros(pts.shape[0])
        print("Warning: WSS array not found in VTU file")

    for loc, pt in ids.items():
        post[loc]["wss"] += [wss[pt]]


def get_results(results, pts, ids_fluid, ids_interface, start_step):
    """Get post-processed quantities at all extracted locations"""
    post_fluid = {}
    post_interface = {}
    post_interface_time = {}  # Keep time series for temporal plots

    for loc in ids_fluid.keys():
        post_fluid[loc] = defaultdict(list)
    for loc in ids_interface.keys():
        post_interface[loc] = defaultdict(list)
        post_interface_time[loc] = defaultdict(list)

    # get results only for the last cardiac cycle (from start_step onwards)
    print(f"Processing steps {start_step} to {len(results)} for time averaging")
    for i, res in enumerate(results):
        if i >= start_step:
            extract_results_fluid_centerline(post_fluid, res, pts, ids_fluid)
            extract_results_wall(post_interface_time, res, pts, ids_interface)
            extract_wss(post_interface_time, res, pts, ids_interface)

    # convert to numpy arrays and compute time average
    for loc in post_fluid.keys():
        for f in post_fluid[loc].keys():
            data = np.array(post_fluid[loc][f])
            # Average over time (axis=0)
            post_fluid[loc][f] = np.mean(data, axis=0)

    for loc in post_interface_time.keys():
        for f in post_interface_time[loc].keys():
            data = np.array(post_interface_time[loc][f])
            # Store time series
            post_interface_time[loc][f] = data
            # Average over time (axis=0) for spatial plots
            post_interface[loc][f] = np.mean(data, axis=0)

    return post_fluid, post_interface, post_interface_time


def read_res(fname):
    """Read all simulation results at all time steps"""
    res = []
    for fn in sorted(glob.glob(fname)):
        geo = read_geo(fn).GetOutput()
        res += [geo]
    return res


def post_process(f_out, start_step):
    """Main post-processing function"""
    fname = os.path.join(f_out, "steady_*.vtu")

    # read results from file
    res = read_res(fname)
    if not len(res):
        raise RuntimeError("No results found in " + f_out)

    print(f"Total time steps found: {len(res)}")

    if start_step >= len(res):
        raise RuntimeError(f"Start step {start_step} is beyond available steps {len(res)}")

    # extract points
    pts = v2n(res[0].GetPoints().GetData())

    # get point and line ids for fluid domain
    ids_fluid, coords_fluid = get_ids(pts, "fluid")

    # get point and line ids for interface (for WSS)
    ids_interface, coords_interface = get_ids(pts, "interface")

    # extract results and time-average over last cardiac cycle
    data_fluid, data_interface, data_interface_time = get_results(res, pts, ids_fluid, ids_interface, start_step)

    # extract WSS at mid ring for all time steps (needed for heartbeat overlay)
    wss_mid_all = extract_wss_mid_timeseries(res, pts, ids_interface)

    return data_fluid, data_interface, data_interface_time, coords_fluid, coords_interface, len(res), wss_mid_all


def plot_single(data, coords, quant, out, locations, plot_combined=False):
    """Plot single quantity (similar to post.py)"""
    # For combined plots, use single subplot; otherwise use multiple
    if plot_combined:
        nx = 1
        ny = 1
        h = 5
        fs = (10, h)
        fig, ax = plt.subplots(ny, nx, figsize=fs, dpi=300)
        ax = [ax]
    else:
        nx = 1
        ny = len(locations)
        h = 3.5 if ny == 1 else 3
        fs = (10, ny * h)
        fig, ax = plt.subplots(ny, nx, figsize=fs, dpi=300, sharex=False)
        if ny == 1:
            ax = [ax]

    # Color map for different circumferential positions (same as post.py)
    colors = {0: plt.cm.tab10(0), 3: plt.cm.tab10(1), 6: plt.cm.tab10(3), 9: plt.cm.tab10(2)}
    labels_circ = {0: "0 o'clock", 3: "3 o'clock", 6: "6 o'clock", 9: "9 o'clock"}

    for i_loc, loc in enumerate(locations):
        # skip if no data available
        if quant not in data[loc]:
            print(f"No {quant} data for location {loc}")
            continue

        # get data for y-axis (already time-averaged)
        ydata = np.array(data[loc][quant]).copy()

        # apply scaling
        if quant in f_scales:
            ydata *= f_scales[quant][0]

        if np.isscalar(ydata):
            continue

        # get data for x-axis
        xdata = coords[loc].copy()
        dim = list(loc).index(":")

        if dim == 0:
            xlabel = "Vessel circumference $\\varphi$ [°]"
            xdata *= 180 / np.pi
            xdata = np.append(xdata, 360)
            ydata = np.append(ydata, [ydata[0]], axis=0)
            xticks = np.arange(0, 360 + 45, 45).astype(int)
        elif dim == 1:
            xlabel = "Vessel radius $r$ [mm]"
            xticks = [xdata[0], xdata[-1]]
        elif dim == 2:
            xlabel = "Vessel axial $z$ [mm]"
            xticks = np.linspace(xdata[0], xdata[-1], 5)

        # Determine which axis to plot on
        ax_idx = 0 if plot_combined else i_loc

        # plot with color coding for circumferential position if combined
        if plot_combined and dim == 2:  # axial plots at different circumferential positions
            circ_pos = loc[0]
            color = colors.get(circ_pos, 'k')
            label = labels_circ.get(circ_pos, str(circ_pos))
            ax[ax_idx].plot(xdata, ydata, "-", color=color, linewidth=2, label=label)
        else:
            ax[ax_idx].plot(xdata, ydata, "k-", linewidth=2)

        ax[ax_idx].grid(True)
        ax[ax_idx].set_xticks(xticks)
        ax[ax_idx].set_xlim([np.min(xdata), np.max(xdata)])
        ax[ax_idx].set_xlabel(xlabel)
        ax[ax_idx].set_ylabel(f_labels[quant][0])

        if plot_combined:
            ax[ax_idx].set_title(f"Time-averaged {quant} at wall")
            if dim == 2:  # Add legend for axial plots
                ax[ax_idx].legend(loc='best', frameon=True)
        else:
            # add location as title
            loc_str = f"{loc[0]}, {loc[1]}, {loc[2]}"
            ax[ax_idx].set_title(f"Time-averaged {quant} at location: {loc_str}")

    plt.tight_layout()

    # assemble filename
    dim_names = ["cir", "rad", "axi"]
    if ":" in locations[0]:
        dim = list(locations[0]).index(":")
        loc = list(locations[0])

        # Add position identifier for axial plots at different circumferential positions
        if dim == 2:  # axial direction
            if plot_combined:
                fname = f"{quant}_{dim_names[dim]}_wall_averaged.pdf"
            else:
                circ_pos = loc[0]  # 0, 3, 6, or 9
                fname = f"{quant}_{dim_names[dim]}_{circ_pos}oclock_averaged.pdf"
        # Add position identifier for circumferential plots at different axial positions
        elif dim == 0:  # circumferential direction
            axial_pos = loc[2]  # start, mid, or end
            fname = f"{quant}_{dim_names[dim]}_{axial_pos}_averaged.pdf"
        else:
            fname = f"{quant}_{dim_names[dim]}_averaged.pdf"
    else:
        fname = f"{quant}_points_averaged.pdf"

    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def extract_wss_mid_timeseries(results, pts, ids_interface):
    """Extract mean WSS at mid wall ring for every time step"""
    loc = (":", "wall", "mid")
    if loc not in ids_interface:
        print("Warning: mid wall ring location not found")
        return None

    pt = ids_interface[loc]
    wss_all = []
    for res in results:
        if res.GetPointData().HasArray("WSS"):
            wss = v2n(res.GetPointData().GetArray("WSS"))
            wss = np.linalg.norm(wss, axis=1)
        elif res.GetPointData().HasArray("Traction"):
            wss = v2n(res.GetPointData().GetArray("Traction"))
            wss = np.linalg.norm(wss, axis=1)
        else:
            wss = np.zeros(pts.shape[0])
        wss_all.append(np.mean(wss[pt]))

    return np.array(wss_all)


def plot_wss_heartbeat_overlay(wss_all, steps_per_beat, out):
    """Overlay WSS at mid wall ring for each heart beat in one plot"""
    wss_dyne = wss_all * f_scales["wss"][0]  # convert to dyne/cm²

    n_beats = len(wss_dyne) // steps_per_beat
    if n_beats == 0:
        print("Not enough steps for one full beat; skipping heartbeat overlay plot")
        return

    colors = plt.cm.tab10.colors
    beat_steps = np.arange(steps_per_beat)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    for i in range(n_beats):
        start = i * steps_per_beat
        end = start + steps_per_beat
        ax.plot(beat_steps, wss_dyne[start:end],
                color=colors[i % len(colors)], linewidth=2, label=f"Beat {i + 1}")

    ax.set_xlabel("Step within Beat")
    ax.set_ylabel("WSS [dyne/cm²]")
    ax.set_title("WSS at Middle Wall — Heart Beat Overlay")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = "wss_heartbeat_overlay_mid.pdf"
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_wss_time_series(data_interface_time, out, start_step):
    """Plot WSS averaged over circumferential rings as function of time"""
    # Locations: inlet, mid, outlet
    locations = [
        (":", "wall", "start"),
        (":", "wall", "mid"),
        (":", "wall", "end")
    ]
    labels = ["Inlet", "Mid", "Outlet"]
    colors = ["blue", "green", "red"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

    for loc, label, color in zip(locations, labels, colors):
        if loc in data_interface_time and "wss" in data_interface_time[loc]:
            # Get WSS time series [n_time, n_points_on_ring]
            wss_data = data_interface_time[loc]["wss"]

            # Average over the circumferential ring at each time step
            wss_avg = np.mean(wss_data, axis=1)

            # Apply scaling to dyne/cm²
            wss_avg *= f_scales["wss"][0]

            # Time steps (offset by start_step to show actual step numbers)
            time_steps = np.arange(len(wss_avg)) + start_step + 1

            # Plot
            ax.plot(time_steps, wss_avg, "-", color=color, linewidth=2, label=label)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("WSS [dyne/cm²]")
    ax.set_title("Wall Shear Stress vs Time (averaged over circumferential rings)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    fname = "wss_time_series_rings.pdf"
    fig.savefig(os.path.join(out, fname), bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_res(data_fluid, data_interface, coords_fluid, coords_interface, out):
    """Plot all results"""
    # fluid domain locations
    loc_cir = [0]
    loc_rad = ["center"]
    loc_axi = ["start", "mid", "end"]

    # interface locations for wall quantities - find which circumferential positions exist
    loc_cir_interface = []
    for lc in [0, 3, 6, 9]:
        if any(lc == loc[0] and "wall" in str(loc) for loc in data_interface.keys()):
            loc_cir_interface.append(lc)

    loc_rad_interface = ["wall"]
    loc_axi_interface = ["start", "mid", "end"]

    print(f"\nAvailable circumferential positions at wall: {loc_cir_interface}")

    # plot velocity along axial line at centerline
    if (0, "center", ":") in coords_fluid:
        plot_single(data_fluid, coords_fluid, "velocity", out, [(0, "center", ":")])

    # plot wall quantities (pressure and WSS) along circumferential rings at interface
    for f in ["pressure", "wss"]:
        for la in loc_axi_interface:
            if (":", "wall", la) in coords_interface:
                plot_single(data_interface, coords_interface, f, out, [(":", "wall", la)])

    # plot wall quantities (pressure and WSS) along axial lines at interface - COMBINED PLOT
    for f in ["pressure", "wss"]:
        # Collect all axial locations that exist
        axial_locs = []
        for lc in loc_cir_interface:
            if (lc, "wall", ":") in coords_interface:
                axial_locs.append((lc, "wall", ":"))

        # Plot all 4 curves together if we have data
        if len(axial_locs) > 0:
            plot_single(data_interface, coords_interface, f, out, axial_locs, plot_combined=True)



def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Post-process pulsatile fluid simulation results")
    parser.add_argument("input_dir", nargs="?", default="steady",
                        help="Directory containing VTU files (default: steady)")
    parser.add_argument("--start-step", type=int, default=96,
                        help="First step index (0-based) for time averaging (default: 96)")
    parser.add_argument("--steps-per-beat", type=int, default=None,
                        help="Steps per heart beat for beat overlay plot")
    parser.add_argument("--output-dir", default="post_results_pulsatile",
                        help="Output directory for plots (default: post_results_pulsatile)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    start_step = args.start_step
    os.makedirs(output_dir, exist_ok=True)

    # post-process
    data_fluid, data_interface, data_interface_time, coords_fluid, coords_interface, n_times, wss_mid_all = post_process(input_dir, start_step)

    print(f"Time-averaged over steps {start_step+1} to {n_times} ({n_times - start_step} steps)")
    print(f"Fluid locations: {list(data_fluid.keys())}")
    print(f"Interface locations: {list(data_interface.keys())}")

    # plot time-averaged spatial results
    plot_res(data_fluid, data_interface, coords_fluid, coords_interface, output_dir)

    # plot WSS time series on circumferential rings
    plot_wss_time_series(data_interface_time, output_dir, start_step)

    # plot WSS heartbeat overlay at mid wall
    if args.steps_per_beat is not None and wss_mid_all is not None:
        plot_wss_heartbeat_overlay(wss_mid_all, args.steps_per_beat, output_dir)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
