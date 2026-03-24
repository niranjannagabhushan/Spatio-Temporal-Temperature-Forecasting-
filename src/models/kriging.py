"""
src/models/kriging.py
---------------------
Spatial interpolation via Ordinary Kriging (OK) and Universal Kriging (UK)
using the PyKrige library.

All calls to ``OK.execute`` / ``UK.execute`` are wrapped in a
``try/except`` block catching :class:`numpy.linalg.LinAlgError` and
other numerical exceptions so that a singular covariance matrix (common
with sparse or collocated station grids) never crashes the pipeline.

Public API
----------
run_ordinary_kriging(x, y, z, grid_x, grid_y, variogram_model) → (z_pred, ss)
run_universal_kriging(x, y, z, grid_x, grid_y, variogram_model) → (z_pred, ss)
run_kriging(station_gdf, config)  →  dict
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError


# ---------------------------------------------------------------------------
# Low-level wrappers (Issue 5 fix — LinAlgError guard)
# ---------------------------------------------------------------------------


def run_ordinary_kriging(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    variogram_model: str = "linear",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit an OrdinaryKriging model and interpolate onto *grid_x* / *grid_y*.

    Parameters
    ----------
    x, y:
        1-D arrays of station easting and northing coordinates.
    z:
        1-D array of observed values (e.g. temperature) at each station.
    grid_x, grid_y:
        1-D arrays defining the interpolation grid axes.
    variogram_model:
        Variogram model passed to :class:`pykrige.ok.OrdinaryKriging`
        (``'linear'``, ``'power'``, ``'gaussian'``, ``'spherical'``, …).

    Returns
    -------
    tuple
        ``(z_pred, variance)`` masked arrays on the grid, or
        ``(None, None)`` if the matrix inversion fails.

    Notes
    -----
    **Issue 5 fix** — ``OK.execute`` is wrapped in a
    ``try/except (LinAlgError, ValueError)`` so that singular covariance
    matrices do not crash the pipeline.
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise ImportError(
            "PyKrige is required for Kriging.  "
            "Install it with: pip install pykrige"
        ) from exc

    OK = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )

    # ------------------------------------------------------------------
    # Issue 5 fix: guard against LinAlgError and other numerical errors
    # raised during the internal matrix inversion inside OK.execute.
    # ------------------------------------------------------------------
    try:
        z_pred, ss = OK.execute("grid", grid_x, grid_y)
        return z_pred, ss
    except (LinAlgError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f"OrdinaryKriging.execute failed ({type(exc).__name__}: {exc}). "
            "Returning (None, None).  Consider using a different variogram "
            "model or increasing the nugget parameter.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None


def run_universal_kriging(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    variogram_model: str = "linear",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit a UniversalKriging model and interpolate onto *grid_x* / *grid_y*.

    Parameters
    ----------
    x, y:
        1-D arrays of station easting and northing coordinates.
    z:
        1-D array of observed values at each station.
    grid_x, grid_y:
        1-D arrays defining the interpolation grid axes.
    variogram_model:
        Variogram model passed to :class:`pykrige.uk.UniversalKriging`.

    Returns
    -------
    tuple
        ``(z_pred, variance)`` masked arrays, or ``(None, None)`` on failure.

    Notes
    -----
    **Issue 5 fix** — same try/except guard as :func:`run_ordinary_kriging`.
    """
    try:
        from pykrige.uk import UniversalKriging
    except ImportError as exc:
        raise ImportError(
            "PyKrige is required for Kriging.  "
            "Install it with: pip install pykrige"
        ) from exc

    UK = UniversalKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )

    # ------------------------------------------------------------------
    # Issue 5 fix: guard against LinAlgError and other numerical errors.
    # ------------------------------------------------------------------
    try:
        z_pred, ss = UK.execute("grid", grid_x, grid_y)
        return z_pred, ss
    except (LinAlgError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f"UniversalKriging.execute failed ({type(exc).__name__}: {exc}). "
            "Returning (None, None).  Consider using a different variogram "
            "model or increasing the nugget parameter.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None


# ---------------------------------------------------------------------------
# High-level runner (integrates with the project pipeline)
# ---------------------------------------------------------------------------


def run_kriging(station_gdf, config: dict) -> dict:
    """Run both Ordinary and Universal Kriging on the station GeoDataFrame.

    Reads coordinate and target columns from *station_gdf*, builds a regular
    interpolation grid, then calls :func:`run_ordinary_kriging` and
    :func:`run_universal_kriging`.  Both calls are individually protected
    against numerical failures (see Issue 5 fix in the low-level wrappers).

    Parameters
    ----------
    station_gdf:
        Output of :func:`src.data.preprocessor.build_pipeline`.
        Must contain ``longitude``, ``latitude``, and the regression target
        column defined in ``config['features']['target']``.
    config:
        Dictionary loaded from ``config.yaml``.

    Returns
    -------
    dict
        ``{
            'ok_pred': ndarray | None,
            'ok_var':  ndarray | None,
            'uk_pred': ndarray | None,
            'uk_var':  ndarray | None,
            'grid_x':  ndarray,
            'grid_y':  ndarray,
        }``
    """
    kriging_cfg = config.get("models", {}).get("kriging", {})
    variogram_model = kriging_cfg.get("variogram_model", "linear")
    grid_resolution = kriging_cfg.get("grid_resolution", 50)
    target = config.get("features", {}).get("target", "temperature")

    df = station_gdf[["longitude", "latitude", target]].dropna()

    x = df["longitude"].values.astype(float)
    y = df["latitude"].values.astype(float)
    z = df[target].values.astype(float)

    grid_x = np.linspace(x.min(), x.max(), grid_resolution)
    grid_y = np.linspace(y.min(), y.max(), grid_resolution)

    ok_pred, ok_var = run_ordinary_kriging(
        x, y, z, grid_x, grid_y, variogram_model=variogram_model
    )
    uk_pred, uk_var = run_universal_kriging(
        x, y, z, grid_x, grid_y, variogram_model=variogram_model
    )

    return {
        "ok_pred": ok_pred,
        "ok_var": ok_var,
        "uk_pred": uk_pred,
        "uk_var": uk_var,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }
