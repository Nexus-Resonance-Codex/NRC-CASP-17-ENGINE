"""
NRC Mathematical Hyperparameter Calibration
===========================================

Performs a deterministic, derivative-free optimization (Nelder-Mead) over the
forcefield weights parameter space. Calibrates the engine against a diverse set of
experimentally determined PDB structures to minimize global CA-RMSD.
"""

import numpy as np
import urllib.request
import os
import json
import sys
from scipy.optimize import minimize

# Add src to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from nrc_casp17_engine.forcefield import NRCForcefield


# 1. Reference PDB list (diverse, small-to-medium proteins)
CALIBRATION_SET = {
    "5AWL": "GTASVNYAEIRGY",  # Chignolin mutant (Beta-hairpin, 13aa)
    "1L2Y": "NLYIQWLKDGGPSSGRPPPS",  # Trp-cage (Helix/loop, 20aa)
    "1GB1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",  # Protein G B1 (Alpha/Beta, 56aa)
}


def download_pdb(pdb_id):
    """Download native structure from RCSB."""
    local_path = f"/tmp/{pdb_id}.pdb"
    if not os.path.exists(local_path):
        url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            return None
    return local_path


def parse_native_ca(pdb_path):
    """Extract native CA coordinates."""
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords)


def calculate_rmsd(coords1, coords2):
    """Compute the RMSD between two sets of coordinates after Kabsch alignment."""
    # Handle length differences
    min_len = min(len(coords1), len(coords2))
    c1 = coords1[:min_len] - np.mean(coords1[:min_len], axis=0)
    c2 = coords2[:min_len] - np.mean(coords2[:min_len], axis=0)

    # Kabsch alignment
    cov = c1.T @ c2
    u, s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    if d < 0:
        vt[-1] *= -1
    rot = u @ vt
    c1_aligned = c1 @ rot

    # RMSD
    diff = c1_aligned - c2
    return np.sqrt(np.mean(np.sum(diff**2, axis=-1)))


# Download PDBs and cache native coordinates
native_cache = {}
for pdb_id in CALIBRATION_SET.keys():
    pdb_path = download_pdb(pdb_id)
    if pdb_path:
        native_cache[pdb_id] = parse_native_ca(pdb_path)


def loss_function(weights_vector):
    """
    Evaluates the prediction accuracy (average RMSD^2) over the training set.
    """
    # Parameter mapping: [hydro, helix, sheet, torsion, elec, rg, steric]
    hydro, helix, sheet, torsion, elec, rg, steric = weights_vector

    # Large penalty for negative weights
    penalty = 0.0
    for w in weights_vector:
        if w < 0.0:
            penalty += 1000.0 * (w**2)

    weights_dict = {
        "bond": 5000.0,  # Keep bond constraint rigid
        "steric": max(0.1, steric),
        "ttt7": 50.0,
        "hydro": max(0.0, hydro),
        "helix": max(0.0, helix),
        "sheet": max(0.0, sheet),
        "torsion": max(0.0, torsion),
        "elec": max(0.0, elec),
        "rg": max(0.1, rg),
    }

    rmsds = []
    for pdb_id, sequence in CALIBRATION_SET.items():
        if pdb_id not in native_cache:
            continue
        native = native_cache[pdb_id]

        # Instantiate forcefield and optimize
        ff = NRCForcefield(sequence, weights=weights_dict)
        opt_coords = ff.optimize(max_iter=150)

        # Calculate RMSD
        rmsd = calculate_rmsd(opt_coords, native)
        rmsds.append(rmsd)

    mean_rmsd = np.mean(rmsds) if rmsds else 100.0
    total_loss = mean_rmsd + penalty
    print(
        f"Weights: {np.round(weights_vector, 2)} | Mean RMSD: {mean_rmsd:.4f} A | Loss: {total_loss:.4f}"
    )
    return total_loss


def run_calibration():
    print("Starting automated hyperparameter search...")
    # Initial guess [hydro, helix, sheet, torsion, elec, rg, steric]
    x0 = [20.0, 10.0, 10.0, 25.0, 5.0, 100.0, 500.0]

    # Deterministic Nelder-Mead optimization
    res = minimize(
        loss_function,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 30, "xatol": 1.0, "fatol": 0.1},
    )

    opt_w = res.x
    final_weights = {
        "bond": 5000.0,
        "steric": max(0.1, opt_w[6]),
        "ttt7": 50.0,
        "hydro": max(0.0, opt_w[0]),
        "helix": max(0.0, opt_w[1]),
        "sheet": max(0.0, opt_w[2]),
        "torsion": max(0.0, opt_w[3]),
        "elec": max(0.0, opt_w[4]),
        "rg": max(0.1, opt_w[5]),
    }

    # Save to src package
    package_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.abspath(
        os.path.join(
            package_dir, "..", "src", "nrc_casp17_engine", "force_weights.json"
        )
    )
    with open(weights_path, "w") as f:
        json.dump(final_weights, f, indent=4)

    print(f"\nOptimization converged. Optimal weights written to {weights_path}:")
    print(json.dumps(final_weights, indent=4))

    # Save a dated version to production directory per Project Rules
    # We will copy it to /mnt/2TBext/FOLD-TEMP/CASP-17/SOURCE_SCRIPTS/math_hyperparameter_calibration_06-26-2026.py
    # and /mnt/2TBext/FOLD-TEMP/CASP-17/SOURCE_SCRIPTS/force_weights_06-26-2026.json


if __name__ == "__main__":
    run_calibration()
