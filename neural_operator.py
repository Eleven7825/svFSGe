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
  enabled              – bool, set true to activate
  pt_file              – path to shear_stress_model.pt
  svd_basis_file       – path to *_basis.npz (Ux, Uy, Uz)
  svd_template_vtk     – path to cylinder.vtk (LDDMM template used to compute SVD basis)
  trunk_xyz_vtk        – path to sample_00001/1-shoot-16.vtk (fixed evaluation grid,
                         must match the ref_xyz_vtk_path used during training)
  branch_dims          – list[int]
  trunk_dims           – list[int]
  final_dim            – int
  mode                 – int, SVD modes (default = branch_dims[0] // 3)
  idw_k                – int, IDW nearest neighbours (default 8)
  idw_power            – float, IDW power (default 2)
  model_dir            – path to ShapeOperatorLearning (optional)

Two distinct 672-node grids:
  svd_template_vtk  – cylinder.vtk: used ONLY for projecting solid displacement
                      onto SVD basis (same nodes the basis was computed on)
  trunk_xyz_vtk     – 1-shoot-16.vtk of sample_00001: fixed collocation points
                      fed as trunk input to the NN (must match training)
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

        # --- SVD basis (Ux, Uy, Uz): project displacement → coefficients ---
        # Basis lives in the cylinder template space (N_template, mode)
        basis = np.load(os.path.expanduser(cfg["svd_basis_file"]))
        self.Ux = basis["Ux"][:, :self.mode].astype(np.float32)
        self.Uy = basis["Uy"][:, :self.mode].astype(np.float32)
        self.Uz = basis["Uz"][:, :self.mode].astype(np.float32)

        # --- SVD template nodes (cylinder.vtk): IDW target for displacement projection ---
        self.svd_template_xyz = _load_vtk_points(cfg["svd_template_vtk"])  # (N_template, 3)

        # --- Trunk evaluation grid (sample_00001/1-shoot-16.vtk) ---
        # Must match ref_xyz_vtk_path used during training — this is the fixed
        # collocation grid fed as trunk input to the NN.
        self.trunk_xyz = _load_vtk_points(cfg["trunk_xyz_vtk"])  # (N_trunk, 3)

        # IDW hyper-parameters
        self.idw_k     = cfg.get("idw_k", 8)
        self.idw_power = cfg.get("idw_power", 2.0)

        # Precomputed IDW weight tables (built on first call; cached thereafter).
        # Three transfers needed:
        #   solid → svd_template  (to project displacement onto SVD basis)
        #   solid → trunk         (not needed; trunk is fixed)
        #   trunk → solid         (to map WSS back to solid interface)
        self._disp_idx = None   # solid → svd_template
        self._disp_w   = None
        self._wss_idx  = None   # trunk → solid
        self._wss_w    = None

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
        print(f"[NeuralOperator] SVD template: {len(self.svd_template_xyz)} nodes, "
              f"trunk grid: {len(self.trunk_xyz)} nodes, modes: {self.mode}")

    def _ensure_idw(self, solid_xyz):
        """Build IDW tables the first time we see the solid mesh layout."""
        if self._disp_idx is not None:
            return
        solid_xyz = solid_xyz.astype(np.float32)
        # solid → SVD template nodes (for displacement projection)
        self._disp_idx, self._disp_w = _idw_weights(
            solid_xyz, self.svd_template_xyz, self.idw_k, self.idw_power)
        # trunk nodes → solid (for WSS back-transfer)
        self._wss_idx, self._wss_w = _idw_weights(
            self.trunk_xyz, solid_xyz, self.idw_k, self.idw_power)
        print(f"[NeuralOperator] IDW tables built: solid={len(solid_xyz)} nodes, "
              f"svd_template={len(self.svd_template_xyz)}, trunk={len(self.trunk_xyz)}")

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

        # 1. Interpolate displacement: solid nodes → SVD template nodes
        disp_template = _apply_idw(solid_disp, self._disp_idx, self._disp_w)

        # 2. Project onto SVD basis → geometry coefficients
        coeff_x = self.Ux.T @ disp_template[:, 0]   # (mode,)
        coeff_y = self.Uy.T @ disp_template[:, 1]
        coeff_z = self.Uz.T @ disp_template[:, 2]
        coeffs  = np.concatenate([coeff_x, coeff_y, coeff_z]).astype(np.float32)

        # 3. NN forward pass at fixed trunk collocation grid → WSS
        n_pts    = len(self.trunk_xyz)
        coeffs_t = torch.from_numpy(np.tile(coeffs, (n_pts, 1))).to(self.device)
        xyz_t    = torch.from_numpy(self.trunk_xyz).to(self.device)

        preds = []
        with torch.no_grad():
            for start in range(0, n_pts, 1000):
                out = self.model(coeffs_t[start:start+1000],
                                 xyz_t[start:start+1000])
                preds.append(out.cpu())
        wss_trunk = torch.cat(preds).numpy().astype(np.float32)  # (N_trunk, 3)

        # 4. Interpolate WSS: trunk nodes → solid interface nodes
        return _apply_idw(wss_trunk, self._wss_idx, self._wss_w)  # (N_solid, 3)
