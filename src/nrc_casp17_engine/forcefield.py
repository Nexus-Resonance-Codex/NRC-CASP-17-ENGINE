"""
NRC Forcefield — PyTorch-Accelerated All-Atom Energy Minimization
================================================================

A delegation wrapper around energy and optimization modules.
"""

import numpy as np
import os
import json
import torch

from .chemistry import NRCChemistry
from .atoms import NRCAtoms
from .energy import NRCPotential
from .geometry import fragment_based_initialization, reconstruct_backbone_frames_np
from .optimizer import NRCOptimizer


class NRCForcefield:
    """
    All-Atom and CA-Lattice Ab Initio Thermodynamic Forcefield.
    """

    def __init__(self, sequence: str, weights: dict = None, contacts: list = None):
        self.sequence = sequence
        self.N_res = len(sequence)
        self.contacts = contacts

        self.chem = NRCChemistry()
        self.atom_lib = NRCAtoms()

        # Load default weights
        self.weights = {
            "bond": 5000.0,
            "steric": 500.0,
            "ttt7": 50.0,
            "hydro": 20.0,
            "solvation": 20.0,
            "centroid_steric": 100.0,
            "helix": 10.0,
            "sheet": 10.0,
            "torsion": 25.0,
            "elec": 5.0,
            "rg": 100.0,
            "contact": 50.0
        }

        # Try loading calibrated weights
        package_dir = os.path.dirname(os.path.abspath(__file__))
        weights_file = os.path.join(package_dir, "force_weights.json")
        if os.path.exists(weights_file):
            try:
                with open(weights_file, "r") as f:
                    calibrated = json.load(f)
                    self.weights.update(calibrated)
            except Exception:
                pass

        if weights:
            self.weights.update(weights)

        # Get chemistry charges
        _, _, self.charges = self.chem.get_params(sequence)

        # Initialize the potential manager
        self.potential = NRCPotential(
            sequence=self.sequence,
            weights=self.weights,
            contacts=self.contacts,
            charges=self.charges
        )

        # Set up coordinates using the new geometry initializer
        self.x0 = self.fragment_based_initialization(self.N_res)

        # Initialize optimizer
        self.optimizer = NRCOptimizer(self.energy_and_gradient)

    def fragment_based_initialization(self, N: int) -> np.ndarray:
        return fragment_based_initialization(
            self.sequence, 
            self.potential.p_alpha, 
            self.potential.p_beta
        )

    def energy_and_gradient(self, coords_flat: np.ndarray) -> tuple:
        """
        Total energy and gradient calculation using PyTorch autograd.
        """
        coords_t = torch.tensor(coords_flat, dtype=torch.float64, requires_grad=True)
        energy_t = self.potential.compute_energy(coords_t)
        
        # Compute backward gradients
        energy_t.backward()
        grad = coords_t.grad.detach().numpy()
        
        return energy_t.item(), grad

    def optimize(self, max_iter: int = 500, use_annealing: bool = False) -> np.ndarray:
        """Run optimizer."""
        if use_annealing:
            coords_flat = self.optimizer.minimize_annealing(self.x0)
        else:
            coords_flat = self.optimizer.minimize_lbfgs(self.x0, max_iter=max_iter)
        self.x0 = coords_flat
        return coords_flat.reshape(-1, 3)

    def generate_all_atom(self, ca_coords: np.ndarray) -> dict:
        """Flesh out CA coordinates to full atom representation."""
        ca_coords = ca_coords.reshape(-1, 3)
        all_coords = []
        atom_types = []
        res_indices = []
        res_names = []

        # Reconstruct NumPy rotation matrices
        rot = reconstruct_backbone_frames_np(ca_coords)

        for i, aa in enumerate(self.sequence):
            phi, psi = 0.0, 0.0  # Kept for compatibility with get_full_residue signature
            res_dict = self.atom_lib.get_full_residue(
                aa, ca_coords[i], rotation_matrix=rot[i], phi=phi, psi=psi
            )
            for atom_name, coord in res_dict.items():
                all_coords.append(coord)
                atom_types.append(atom_name)
                res_indices.append(i + 1)
                res_names.append(aa)

        return {
            "coords": np.array(all_coords),
            "atom_types": atom_types,
            "res_indices": res_indices,
            "res_names": res_names,
        }
