"""
Neural operator surrogate for wall shear stress (WSS) prediction.

Replaces the MESH + FLUID svFSI solvers within each FSI coupling sub-iteration.

Geometry encoding pipeline (per sub-iteration):
  1. Solid interface displacement (N_solid, 3) at solid FEM mesh nodes
  2. IDW interpolation → LDDMM reference nodes (N_lddmm, 3)
  3. Project onto SVD basis (Ux, Uy, Uz) → geometry coefficients (3*mode,)
  4. DeepONet forward pass at ref_xyz → WSS (N_lddmm, 3)
  5. IDW interpolation back → solid interface nodes (N_solid, 3)

The LDDMM reference geometry and SVD basis are precomputed by:
  TAA_CFD_pipeline/coefficients_convert.py  (basis)
  TAA_CFD_pipeline/vtk/cylinder.vtk         (ref_xyz_vtk_path)

Config keys (all under "neural_operator" in the JSON):
  enabled         – bool, set true to activate
  pt_file         – path to shear_stress_model.pt
  svd_basis_file  – path to *_basis.npz (Ux, Uy, Uz)
  ref_xyz_vtk_path – path to cylinder.vtk (LDDMM template, 672 nodes)
  branch_dims     – list[int]
  trunk_dims      – list[int]
  final_dim       – int
  mode            – int, SVD modes (default = branch_dims[0] // 3)
  idw_k           – int, IDW nearest neighbours (default 8)
  idw_power       – float, IDW power (default 2)
  model_dir       – path to ShapeOperatorLearning (optional)
"""

import os
import sys
import importlib

import numpy as np
import torch
from scipy.spatial import cKDTree

import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _load_vtk_points(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"VTK not found: {path}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float32)


def _idw_weights(src_xyz, dst_xyz, k, power):
    """
    Precompute IDW weights: for each point in dst, a sparse set of (indices, weights)
    into src.

    Returns
    -------
    indices : (N_dst, k)  int
    weights : (N_dst, k)  float32
    """
    tree = cKDTree(src_xyz)
    dists, idxs = tree.query(dst_xyz, k=k)
    dists = dists.astype(np.float32)

    # Handle exact matches (dist == 0)
    exact = dists[:, 0] == 0.0
    dists[exact, :] = 1.0         # avoid division by zero
    w = 1.0 / dists ** power
    w[exact, :] = 0.0
    w[exact, 0] = 1.0             # exact match → weight 1 on nearest neighbour
    w /= w.sum(axis=1, keepdims=True)
    return idxs, w


def _apply_idw(values, indices, weights):
    """
    Apply precomputed IDW weights to interpolate values.

    Parameters
    ----------
    values  : (N_src, C) float32
    indices : (N_dst, k) int
    weights : (N_dst, k) float32

    Returns
    -------
    (N_dst, C) float32
    """
    # values[indices] → (N_dst, k, C)
    return (values[indices] * weights[:, :, None]).sum(axis=1)


class NeuralOperator:
    def __init__(self, cfg):
        """
        Parameters
        ----------
        cfg : dict  (from p["neural_operator"] in JSON config)
        """
        model_dir = cfg.get(
            "model_dir",
            os.path.expanduser("~/ShapeOperatorLearning"),
        )
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        model_module = importlib.import_module("models.shearStressNN_coeff")
        ShearStressNN = getattr(model_module, "ShearStressNN")

        branch_dims = list(cfg["branch_dims"])
        self.mode = cfg.get("mode", branch_dims[0] // 3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- SVD basis (Ux, Uy, Uz): needed to project displacement → coefficients ---
        basis = np.load(os.path.expanduser(cfg["svd_basis_file"]))
        self.Ux = basis["Ux"][:, :self.mode].astype(np.float32)   # (N_lddmm, mode)
        self.Uy = basis["Uy"][:, :self.mode].astype(np.float32)
        self.Uz = basis["Uz"][:, :self.mode].astype(np.float32)

        # --- LDDMM reference node positions (cylinder template) ---
        self.ref_xyz = _load_vtk_points(cfg["ref_xyz_vtk_path"])  # (N_lddmm, 3)

        # IDW hyper-parameters
        self.idw_k     = cfg.get("idw_k", 8)
        self.idw_power = cfg.get("idw_power", 2.0)

        # Precomputed IDW weight tables (computed on first call once we know the
        # solid mesh layout; cached thereafter)
        self._fwd_idx = None   # solid → LDDMM
        self._fwd_w   = None
        self._bwd_idx = None   # LDDMM → solid
        self._bwd_w   = None

        # --- Load trained model ---
        model = ShearStressNN(
            branch_dims=branch_dims,
            trunk_dims=list(cfg["trunk_dims"]),
            final_dim=cfg.get("final_dim", 64),
        )
        ckpt = torch.load(
            os.path.expanduser(cfg["pt_file"]),
            map_location="cpu",
        )
        state = {k.replace("module.", ""): v
                 for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

        print(f"[NeuralOperator] model loaded from {cfg['pt_file']}")
        print(f"[NeuralOperator] LDDMM ref nodes: {len(self.ref_xyz)}, SVD modes: {self.mode}")

    def _ensure_idw(self, solid_xyz):
        """Build IDW tables the first time we see the solid mesh layout."""
        if self._fwd_idx is not None:
            return
        solid_xyz = solid_xyz.astype(np.float32)
        self._fwd_idx, self._fwd_w = _idw_weights(
            solid_xyz, self.ref_xyz, self.idw_k, self.idw_power)
        self._bwd_idx, self._bwd_w = _idw_weights(
            self.ref_xyz, solid_xyz, self.idw_k, self.idw_power)
        print(f"[NeuralOperator] IDW tables built: "
              f"solid={len(solid_xyz)} nodes → LDDMM={len(self.ref_xyz)} nodes")

    def predict_wss(self, solid_disp, solid_xyz):
        """
        Predict WSS at solid interface nodes.

        Parameters
        ----------
        solid_disp : np.ndarray (N_solid, 3)
            Displacement of solid interface nodes from reference configuration.
        solid_xyz  : np.ndarray (N_solid, 3)
            Reference positions of solid interface nodes.

        Returns
        -------
        wss_solid : np.ndarray (N_solid, 3)
            Predicted WSS at solid interface nodes.
        """
        solid_disp = np.asarray(solid_disp, dtype=np.float32)
        solid_xyz  = np.asarray(solid_xyz,  dtype=np.float32)

        # Lazily build IDW weight tables
        self._ensure_idw(solid_xyz)

        # 1. Interpolate displacement from solid mesh to LDDMM nodes
        disp_lddmm = _apply_idw(solid_disp, self._fwd_idx, self._fwd_w)  # (N_lddmm, 3)

        # 2. Project onto SVD basis → geometry coefficients
        coeff_x = self.Ux.T @ disp_lddmm[:, 0]   # (mode,)
        coeff_y = self.Uy.T @ disp_lddmm[:, 1]
        coeff_z = self.Uz.T @ disp_lddmm[:, 2]
        coeffs  = np.concatenate([coeff_x, coeff_y, coeff_z]).astype(np.float32)

        # 3. NN forward pass at LDDMM ref nodes → WSS (N_lddmm, 3)
        n_pts    = len(self.ref_xyz)
        coeffs_t = torch.from_numpy(np.tile(coeffs, (n_pts, 1))).to(self.device)
        xyz_t    = torch.from_numpy(self.ref_xyz).to(self.device)

        preds = []
        with torch.no_grad():
            for start in range(0, n_pts, 1000):
                out = self.model(coeffs_t[start:start+1000],
                                 xyz_t[start:start+1000])
                preds.append(out.cpu())
        wss_lddmm = torch.cat(preds).numpy().astype(np.float32)  # (N_lddmm, 3)

        # 4. Interpolate WSS from LDDMM nodes back to solid interface nodes
        return _apply_idw(wss_lddmm, self._bwd_idx, self._bwd_w)  # (N_solid, 3)
