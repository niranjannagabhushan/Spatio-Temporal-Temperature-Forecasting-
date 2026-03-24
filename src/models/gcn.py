"""
src/models/gcn.py
-----------------
Graph Convolutional Network for spatial temperature prediction.

The station readings are sub-sampled (default 8 000 rows) for computational
feasibility.  A k-NN spatial graph is built from station coordinates and
passed to a two-layer GCN implemented in PyTorch Geometric.

Public API
----------
train_gcn(station_gdf, config)  →  dict
"""

from __future__ import annotations

from typing import Any, Dict

import geopandas as gpd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature specification  (must match ml_models.py)
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "altitude",
    "latitude",
    "longitude",
    "distance_to_lake",
    "distance_to_river",
    "hour",
    "month",
]
_TARGET_COL = "temperature"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subsample(gdf: gpd.GeoDataFrame, n: int, seed: int = 42) -> gpd.GeoDataFrame:
    if len(gdf) <= n:
        return gdf.reset_index(drop=True)
    return gdf.sample(n=n, random_state=seed).reset_index(drop=True)


def _build_knn_edge_index(coords: np.ndarray, k: int):
    """Return a COO-format edge_index tensor for a k-NN graph."""
    try:
        import torch
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for GCN training.  "
            "Install it with: pip install torch"
        ) from exc

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)

    rows, cols = [], []
    for i, neighbours in enumerate(indices):
        for j in neighbours[1:]:  # skip self
            rows.append(i)
            cols.append(j)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index


def _build_gcn_model(in_channels: int, hidden: int = 64, out_channels: int = 1):
    """Return a two-layer GCN model."""
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for GCN training.  "
            "Install it with: pip install torch-geometric"
        ) from exc

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, out_channels)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.3)

        def forward(self, x, edge_index):
            x = self.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            return x.squeeze(-1)

    return GCN()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_gcn(
    station_gdf: gpd.GeoDataFrame,
    config: dict,
) -> Dict[str, Any]:
    """Train a GCN on a spatial k-NN graph of weather stations.

    Parameters
    ----------
    station_gdf:
        Output of :func:`src.data.preprocessor.build_pipeline`.
    config:
        Dictionary loaded from ``config.yaml``.

    Returns
    -------
    dict
        Keys: ``model``, ``mse``, ``mae``, ``r2``.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for GCN training.  "
            "Install it with: pip install torch"
        ) from exc

    gcn_cfg = config.get("models", {}).get("gcn", {})
    n_subsample = gcn_cfg.get("n_subsample", 8_000)
    k_neighbours = gcn_cfg.get("k_neighbours", 6)
    hidden_dim = gcn_cfg.get("hidden_dim", 64)
    lr = gcn_cfg.get("lr", 1e-3)
    epochs = gcn_cfg.get("epochs", 150)
    seed = gcn_cfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1.  Prepare data
    # ------------------------------------------------------------------
    available = [c for c in _FEATURE_COLS if c in station_gdf.columns]
    df = station_gdf[available + [_TARGET_COL]].dropna()
    df = _subsample(df, n_subsample, seed=seed)

    X_raw = df[available].values.astype(np.float32)
    y_raw = df[_TARGET_COL].values.astype(np.float32)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw).astype(np.float32)

    # Coordinates for graph construction (lat/lon columns)
    lat_col = "latitude" if "latitude" in df.columns else available[0]
    lon_col = "longitude" if "longitude" in df.columns else available[1]
    coords = df[[lat_col, lon_col]].values

    # ------------------------------------------------------------------
    # 2.  Build graph
    # ------------------------------------------------------------------
    edge_index = _build_knn_edge_index(coords, k=k_neighbours)

    # ------------------------------------------------------------------
    # 3.  Train / test split (index-based, preserving graph topology)
    # ------------------------------------------------------------------
    n = len(df)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)

    X_tensor = torch.tensor(X_scaled)
    y_tensor = torch.tensor(y_raw)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[test_idx] = True

    # ------------------------------------------------------------------
    # 4.  Model, optimiser, loss
    # ------------------------------------------------------------------
    model = _build_gcn_model(
        in_channels=X_tensor.shape[1],
        hidden=hidden_dim,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # 5.  Training loop
    # ------------------------------------------------------------------
    model.train()
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad()
        out = model(X_tensor, edge_index)
        loss = criterion(out[train_mask], y_tensor[train_mask])
        loss.backward()
        optimiser.step()

        if epoch % 25 == 0:
            print(f"    Epoch {epoch:>4}/{epochs}  train loss = {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 6.  Evaluation
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor, edge_index).numpy()

    y_test_np = y_tensor[test_mask].numpy()
    y_pred_np = predictions[test_mask]

    return {
        "model": "GCN",
        "mse": float(mean_squared_error(y_test_np, y_pred_np)),
        "mae": float(mean_absolute_error(y_test_np, y_pred_np)),
        "r2": float(r2_score(y_test_np, y_pred_np)),
    }
