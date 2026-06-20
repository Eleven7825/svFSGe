"""
Neural operator surrogate for WSS and pressure prediction in svFSGe.

Replaces the MESH + FLUID svFSI solvers within each FSI coupling sub-iteration.

Per-sub-iteration inference pipeline:
  1. Compute current solid interface positions = solid_xyz + solid_disp
  2. Write surface as legacy ASCII VTK (triangulated)
  3. Call MATLAB LDDMM: register cylinder.vtk → current surface
     → reads back 1-shoot-16.vtk (deformed cylinder nodes in physical space)
  4. Compute LDDMM displacement at cylinder nodes:
       disp_cyl = shoot16_pts - cylinder_pts
  5. Project onto SVD basis → geometry branch coefficients
  6. Pull back solid FEM nodes to reference (cylinder) space:
       For each FEM node, find k nearest shoot16 nodes;
       the corresponding cylinder nodes give the reference position
       (cylinder_node[i] ↔ shoot16_node[i] by LDDMM particle correspondence)
  7. NN forward passes (WSS + pressure): trunk = pulled-back FEM positions,
     branch = coefficients
     → WSS (N_solid, 3) and pressure (N_solid,) at solid FEM interface nodes

Required config keys (under "neural_operator" in JSON):
  enabled           – bool
  pt_file           – path to WSS shear_stress_model.pt
  pressure_pt_file  – path to pressure shear_stress_model.pt (required)
  svd_basis_file    – path to *_basis.npz  (Ux, Uy, Uz)
  svd_template_vtk  – path to cylinder.vtk  (672 nodes, LDDMM template)
  matlab_exe        – path to MATLAB executable
  lddmm_script_dir  – directory containing lddmm_register_single.m
  work_dir          – writable scratch directory for VTK files and LDDMM output
  branch_dims       – list[int]
  trunk_dims        – list[int]
  final_dim         – int
  mode              – int   (SVD modes; default = branch_dims[0] // 3)
  idw_k             – int   (IDW neighbours; default 8)
  idw_power         – float (IDW exponent;  default 2)
  model_dir         – path to ShapeOperatorLearning (optional)
  matlab_timeout    – int   MATLAB timeout in seconds (default 600)
"""

import os
import sys
import importlib
import subprocess
import tempfile

import numpy as np
import torch
from scipy.spatial import cKDTree

import vtk
from vtk.util.numpy_support import vtk_to_numpy


# ---------------------------------------------------------------------------
# VTK I/O helpers
# ---------------------------------------------------------------------------

def _load_vtk_points(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VTK not found: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float32)


def _write_surface_vtk(polydata, xyz, out_path):
    """
    Write a vtkPolyData with updated node positions to a legacy ASCII VTK file.
    Applies vtkTriangleFilter to ensure triangulated cells (required by fshapesTk).
    """
    # Update points on a copy
    poly = vtk.vtkPolyData()
    poly.DeepCopy(polydata)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(len(xyz))
    for i, (x, y, z) in enumerate(xyz):
        pts.SetPoint(i, float(x), float(y), float(z))
    poly.SetPoints(pts)

    # Triangulate (interface.vtp cells may be quads)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(tri.GetOutput())
    writer.SetFileTypeToASCII()
    writer.Write()


# ---------------------------------------------------------------------------
# IDW helpers
# ---------------------------------------------------------------------------

def _idw_weights(src_xyz, dst_xyz, k, power):
    """Return (indices, weights) for IDW from src to dst."""
    tree = cKDTree(src_xyz)
    dists, idxs = tree.query(dst_xyz, k=k)
    dists = dists.astype(np.float32)

    exact = dists[:, 0] == 0.0
    dists[exact, :] = 1.0
    w = 1.0 / dists ** power
    w[exact, :] = 0.0
    w[exact, 0] = 1.0
    w /= w.sum(axis=1, keepdims=True)
    return idxs, w


def _apply_idw(values, indices, weights):
    """(N_dst, C) = IDW interpolation of (N_src, C) values."""
    return (values[indices] * weights[:, :, None]).sum(axis=1)


# ---------------------------------------------------------------------------
# NeuralOperator
# ---------------------------------------------------------------------------

class NeuralOperator:
    def __init__(self, cfg):
        model_dir = cfg.get(
            "model_dir",
            os.path.expanduser("~/ShapeOperatorLearning"),
        )
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        model_module = importlib.import_module("models.shearStressNN_coeff")
        ShearStressNN = getattr(model_module, "ShearStressNN")

        branch_dims = list(cfg["branch_dims"])
        self.mode   = cfg.get("mode", branch_dims[0] // 3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SVD basis — lives in cylinder template space (N_cyl, mode)
        basis    = np.load(os.path.expanduser(cfg["svd_basis_file"]))
        self.Ux  = basis["Ux"][:, :self.mode].astype(np.float32)
        self.Uy  = basis["Uy"][:, :self.mode].astype(np.float32)
        self.Uz  = basis["Uz"][:, :self.mode].astype(np.float32)

        # Cylinder template node positions (N_cyl, 3) — reference for SVD and pull-back
        self._source_vtk_path = os.path.expanduser(cfg["svd_template_vtk"])
        self.cyl_xyz = _load_vtk_points(self._source_vtk_path)

        # LDDMM / MATLAB settings
        self.matlab_exe      = cfg.get("matlab_exe", "/home/shiyi/matlab/bin/matlab")
        self.lddmm_script_dir = os.path.expanduser(
            cfg.get("lddmm_script_dir", "~/TAA_CFD_pipeline")
        )
        self.work_dir        = os.path.expanduser(cfg.get("work_dir", "~/fsg_no_work"))
        self.matlab_timeout  = cfg.get("matlab_timeout", 600)
        os.makedirs(self.work_dir, exist_ok=True)

        # IDW parameters
        self.idw_k     = cfg.get("idw_k", 8)
        self.idw_power = cfg.get("idw_power", 2.0)

        trunk_dims = list(cfg["trunk_dims"])
        final_dim  = cfg.get("final_dim", 64)

        # WSS model (out_dim=3)
        self.model = self._load_model(
            ShearStressNN, branch_dims, trunk_dims, final_dim, out_dim=3,
            pt_file=cfg["pt_file"],
        )

        # Pressure model (out_dim=1) — required: solid solver always needs interface pressure
        pressure_pt = cfg["pressure_pt_file"]
        self.pressure_model = self._load_model(
            ShearStressNN, branch_dims, trunk_dims, final_dim, out_dim=1,
            pt_file=pressure_pt,
        )
        print(f"[NeuralOperator] pressure model: {pressure_pt}")

        print(f"[NeuralOperator] WSS model: {cfg['pt_file']}")
        print(f"[NeuralOperator] cylinder template: {len(self.cyl_xyz)} nodes, modes: {self.mode}")
        print(f"[NeuralOperator] MATLAB: {self.matlab_exe}")
        print(f"[NeuralOperator] work dir: {self.work_dir}")

    def _load_model(self, cls, branch_dims, trunk_dims, final_dim, out_dim, pt_file):
        model = cls(
            branch_dims=branch_dims,
            trunk_dims=trunk_dims,
            final_dim=final_dim,
            out_dim=out_dim,
        )
        ckpt  = torch.load(os.path.expanduser(pt_file), map_location="cpu")
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # LDDMM
    # ------------------------------------------------------------------

    def _run_lddmm(self, target_vtk, output_dir):
        """
        Call MATLAB to register cylinder → target_vtk.
        Returns shoot16_pts (N_cyl, 3): deformed positions of cylinder nodes.
        """
        source_vtk = self._source_vtk_path

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            self.matlab_exe,
            "-sd", self.lddmm_script_dir,
            "-batch",
            (f"lddmm_register_single("
             f"'{source_vtk}', "
             f"'{target_vtk}', "
             f"'{output_dir}')")
        ]
        print(f"[NeuralOperator] running LDDMM ...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.matlab_timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB LDDMM failed (exit {result.returncode}):\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        shoot_vtk = os.path.join(output_dir, "1-shoot-16.vtk")
        if not os.path.exists(shoot_vtk):
            raise FileNotFoundError(
                f"LDDMM output not found: {shoot_vtk}\n"
                f"MATLAB stdout:\n{result.stdout}"
            )

        return _load_vtk_points(shoot_vtk)   # (N_cyl, 3)

    # ------------------------------------------------------------------
    # Pull-back
    # ------------------------------------------------------------------

    def _pull_back(self, solid_xyz, shoot16_pts):
        """
        Pull solid FEM interface nodes from current (physical) space back to
        cylinder reference space.

        LDDMM gives a particle correspondence: cyl_xyz[i] ↔ shoot16_pts[i].
        For each solid FEM node (in physical space), find k nearest shoot16 nodes,
        then IDW-interpolate the corresponding cylinder positions.

        Parameters
        ----------
        solid_xyz   : (N_solid, 3) solid FEM reference positions (physical)
        shoot16_pts : (N_cyl, 3)  deformed cylinder node positions (physical)

        Returns
        -------
        (N_solid, 3) solid FEM nodes pulled back to cylinder reference space
        """
        idxs, weights = _idw_weights(
            shoot16_pts, solid_xyz, self.idw_k, self.idw_power
        )
        return _apply_idw(self.cyl_xyz, idxs, weights)

    # ------------------------------------------------------------------
    # Batched NN forward pass
    # ------------------------------------------------------------------

    def _forward(self, model, coeffs_t, xyz_t, batch=1000):
        """Run model in batches; returns numpy array."""
        preds = []
        with torch.no_grad():
            for start in range(0, len(xyz_t), batch):
                out = model(
                    coeffs_t[start:start+batch],
                    xyz_t[start:start+batch],
                )
                preds.append(out.cpu())
        return torch.cat(preds).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Main inference entry points
    # ------------------------------------------------------------------

    def predict_wss_and_pressure(self, solid_disp, solid_xyz, solid_mesh, call_id=0):
        """
        Predict WSS and (optionally) pressure at solid interface nodes.

        Runs LDDMM once, then does one forward pass per active model.

        Parameters
        ----------
        solid_disp : (N_solid, 3) displacement of solid interface nodes
        solid_xyz  : (N_solid, 3) reference positions of solid interface nodes
        solid_mesh : vtkPolyData  solid interface mesh (for surface connectivity)
        call_id    : int          iteration counter (used to name temp files)

        Returns
        -------
        wss      : (N_solid, 3)
        pressure : (N_solid,)
        """
        solid_disp = np.asarray(solid_disp, dtype=np.float32)
        solid_xyz  = np.asarray(solid_xyz,  dtype=np.float32)

        # 1. Current surface positions
        current_xyz = solid_xyz + solid_disp

        # 2. Write current surface as VTK
        target_vtk = os.path.join(self.work_dir, f"target_{call_id}.vtk")
        output_dir  = os.path.join(self.work_dir, f"lddmm_{call_id}")
        _write_surface_vtk(solid_mesh, current_xyz, target_vtk)

        # 3. Run LDDMM (once per sub-iteration)
        shoot16_pts = self._run_lddmm(target_vtk, output_dir)   # (N_cyl, 3)

        # 4. LDDMM displacement at cylinder nodes → SVD coefficients
        disp_cyl = shoot16_pts - self.cyl_xyz                   # (N_cyl, 3)
        coeff_x  = self.Ux.T @ disp_cyl[:, 0]                  # (mode,)
        coeff_y  = self.Uy.T @ disp_cyl[:, 1]
        coeff_z  = self.Uz.T @ disp_cyl[:, 2]
        coeffs   = np.concatenate([coeff_x, coeff_y, coeff_z]).astype(np.float32)

        # 5. Pull back solid FEM nodes to cylinder reference space
        trunk_input = self._pull_back(solid_xyz, shoot16_pts)   # (N_solid, 3)

        # 6. Shared tensors
        n_pts    = len(trunk_input)
        coeffs_t = torch.from_numpy(np.tile(coeffs, (n_pts, 1))).to(self.device)
        xyz_t    = torch.from_numpy(trunk_input).to(self.device)

        # 7. WSS forward pass → (N_solid, 3)
        wss = self._forward(self.model, coeffs_t, xyz_t)

        # 8. Pressure forward pass → (N_solid,)
        p_raw    = self._forward(self.pressure_model, coeffs_t, xyz_t)  # (N_solid, 1)
        pressure = p_raw.squeeze(-1)                                     # (N_solid,)

        return wss, pressure

    def predict_wss(self, solid_disp, solid_xyz, solid_mesh, call_id=0):
        """Backward-compatible wrapper that returns only WSS."""
        wss, _ = self.predict_wss_and_pressure(solid_disp, solid_xyz, solid_mesh, call_id)
        return wss
