"""
NRC Forcefield — L-BFGS-B Optimized All-Atom Energy Minimization
================================================================

Implements a complete ab initio thermodynamic potential function with analytical gradients:
- Chou-Fasman secondary structure propensities
- Kyte-Doolittle hydrophobic collapse
- TTT-7 resonance & Tesla 3-6-9 exclusion
- Screened Coulomb electrostatics
- Local pseudo-dihedral distance constraints
- Radius of gyration confinement
"""

import numpy as np
import os
import json
from scipy.optimize import minimize

from .chemistry import NRCChemistry
from .atoms import NRCAtoms


class NRCForcefield:
    """
    All-Atom and CA-Lattice Ab Initio Thermodynamic Forcefield.
    """

    # Chou-Fasman helical (P_alpha) and sheet (P_beta) propensities
    CHOU_FASMAN = {
        "A": (1.42, 0.83), "R": (0.98, 0.93), "N": (0.67, 0.89), "D": (1.01, 0.54),
        "C": (0.70, 1.19), "Q": (1.11, 1.10), "E": (1.51, 0.37), "G": (0.57, 0.75),
        "H": (1.00, 0.87), "I": (1.08, 1.60), "L": (1.21, 1.30), "K": (1.14, 0.74),
        "M": (1.45, 1.05), "F": (1.13, 1.38), "P": (0.57, 0.55), "S": (0.77, 0.75),
        "T": (0.83, 1.19), "W": (1.08, 1.37), "Y": (0.69, 1.47), "V": (1.06, 1.70)
    }

    # Kyte-Doolittle hydrophobicity scale
    KYTE_DOOLITTLE = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
        "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
        "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
    }

    def __init__(self, sequence: str, weights: dict = None):
        self.sequence = sequence
        self.N_res = len(sequence)
        self.phi = (1 + np.sqrt(5)) / 2

        self.chem = NRCChemistry()
        self.atom_lib = NRCAtoms()

        # Load default weights
        self.weights = {
            "bond": 5000.0,
            "steric": 500.0,
            "ttt7": 50.0,
            "hydro": 20.0,
            "helix": 10.0,
            "sheet": 10.0,
            "torsion": 25.0,
            "elec": 5.0,
            "rg": 100.0
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

        # Precompute Kyte-Doolittle positive hydrophobicity matrix
        self.h_pos = np.array([max(0.0, self.KYTE_DOOLITTLE.get(aa, 0.0)) for aa in sequence])

        # Precompute Chou-Fasman secondary structure propensities
        self.p_alpha = np.array([self.CHOU_FASMAN.get(aa, (1.0, 1.0))[0] for aa in sequence])
        self.p_beta = np.array([self.CHOU_FASMAN.get(aa, (1.0, 1.0))[1] for aa in sequence])

        # Get chemistry charges
        _, _, self.charges = self.chem.get_params(sequence)

        self.RG_TARGET = 3.0 * (self.N_res**0.33)
        self.MODULAR_SCALE = 3.8017

        # Seed coordinates on uniform spherical Fibonacci spiral
        self.x0 = self.spherical_fibonacci_initialization(self.N_res)

    def spherical_fibonacci_initialization(self, N: int) -> np.ndarray:
        """uniform points on sphere using Fibonacci spiral."""
        indices = np.arange(1, N + 1)
        z = 1 - (2 * indices - 1) / N
        theta = (2 * np.pi / (self.phi**2)) * indices
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        return (np.column_stack((x, y, z)) * self.RG_TARGET).flatten()

    def energy_and_gradient(self, coords_flat: np.ndarray) -> tuple:
        """
        Total energy and gradient calculation. All interactions are vectorized in CA space.
        """
        coords = coords_flat.reshape(-1, 3)
        N = coords.shape[0]
        grad = np.zeros_like(coords)
        total_e = 0.0

        # 1. Harmonic Backbone Constraints (i to i+1)
        diff_bond = coords[1:] - coords[:-1]
        d_bond = np.linalg.norm(diff_bond, axis=1) + 1e-9
        bond_e = self.weights["bond"] * np.sum((d_bond - 3.8)**2)
        total_e += bond_e

        bond_mag = 2 * self.weights["bond"] * (d_bond - 3.8) / d_bond
        grad[:-1] += -bond_mag[:, np.newaxis] * diff_bond
        grad[1:] += bond_mag[:, np.newaxis] * diff_bond

        # Non-bonded terms mask (j - i >= 3)
        mask = np.triu(np.ones((N, N), dtype=bool), k=3)
        if np.any(mask):
            idx_i, idx_j = np.where(mask)
            diff_nb = coords[idx_i] - coords[idx_j]
            d_nb = np.linalg.norm(diff_nb, axis=1) + 1e-9

            # 2. Steric repulsion (clash check)
            clash_mask = d_nb < 4.0
            if np.any(clash_mask):
                d_clash = d_nb[clash_mask]
                diff_clash = diff_nb[clash_mask]
                steric_e = self.weights["steric"] * np.sum((4.0 - d_clash)**2)
                total_e += steric_e

                steric_mag = -2 * self.weights["steric"] * (4.0 - d_clash) / d_clash
                np.add.at(grad, idx_i[clash_mask], steric_mag[:, np.newaxis] * diff_clash)
                np.add.at(grad, idx_j[clash_mask], -steric_mag[:, np.newaxis] * diff_clash)

            # 3. TTT-7 Resonance & Tesla 3-6-9 exclusion
            dr = d_nb * self.MODULAR_SCALE
            damping = 1.0 / (1.0 + 0.1 * d_nb**2)
            void_penalty = self.weights["steric"] * damping * (1.0 + np.cos(2 * np.pi * dr / 3.0))
            total_e += np.sum(void_penalty)

            p_grad_periodic = -self.weights["steric"] * damping * (2 * np.pi / 3.0) * np.sin(2 * np.pi * dr / 3.0) * self.MODULAR_SCALE
            p_grad_damping = -self.weights["steric"] * (1.0 + np.cos(2 * np.pi * dr / 3.0)) * (0.2 * d_nb) * (damping**2)
            p_grad_total = p_grad_periodic + p_grad_damping

            ttt_factor = 2 * np.pi / 9.0
            ttt_e = -self.weights["ttt7"] * np.cos(ttt_factor * (dr - 7.0))
            total_e += np.sum(ttt_e)
            ttt_grad_mag = self.weights["ttt7"] * ttt_factor * np.sin(ttt_factor * (dr - 7.0)) * self.MODULAR_SCALE

            nb_mag = (p_grad_total + ttt_grad_mag) / d_nb
            np.add.at(grad, idx_i, nb_mag[:, np.newaxis] * diff_nb)
            np.add.at(grad, idx_j, -nb_mag[:, np.newaxis] * diff_nb)

            # 4. Hydrophobic collapse
            h_i = self.h_pos[idx_i]
            h_j = self.h_pos[idx_j]
            h_prod = h_i * h_j
            if np.any(h_prod > 0):
                exp_term = np.exp(-0.5 * (d_nb - 4.5)**2)
                hydro_e = self.weights["hydro"] * np.sum(-h_prod * exp_term)
                total_e += hydro_e

                hydro_mag = self.weights["hydro"] * h_prod * (d_nb - 4.5) * exp_term / d_nb
                np.add.at(grad, idx_i, hydro_mag[:, np.newaxis] * diff_nb)
                np.add.at(grad, idx_j, -hydro_mag[:, np.newaxis] * diff_nb)

            # 5. Coulomb electrostatics
            q_i = self.charges[idx_i]
            q_j = self.charges[idx_j]
            q_prod = q_i * q_j
            if np.any(q_prod != 0):
                elec_e = self.weights["elec"] * np.sum(q_prod / d_nb**2)
                total_e += elec_e

                elec_mag = -2 * self.weights["elec"] * q_prod / d_nb**4
                np.add.at(grad, idx_i, elec_mag[:, np.newaxis] * diff_nb)
                np.add.at(grad, idx_j, -elec_mag[:, np.newaxis] * diff_nb)

            # 6. Hydrogen bonding sheet (long range attraction with Gaussian localization)
            p_beta_prod = self.p_beta[idx_i] * self.p_beta[idx_j]
            if np.any(p_beta_prod > 0):
                diff_target = d_nb - 4.8
                exp_term = np.exp(-0.25 * diff_target**2)
                sheet_e = self.weights["sheet"] * np.sum(p_beta_prod * diff_target**2 * exp_term)
                total_e += sheet_e

                f_prime = diff_target * exp_term * (2.0 - 0.5 * diff_target**2)
                sheet_mag = self.weights["sheet"] * p_beta_prod * f_prime / d_nb
                np.add.at(grad, idx_i, sheet_mag[:, np.newaxis] * diff_nb)
                np.add.at(grad, idx_j, -sheet_mag[:, np.newaxis] * diff_nb)

        # 7. Local pseudo-torsions (distances i to i+2 and i to i+3)
        if N > 2:
            idx_i = np.arange(N - 2)
            idx_k = idx_i + 2
            diff_2 = coords[idx_i] - coords[idx_k]
            d_2 = np.linalg.norm(diff_2, axis=1) + 1e-9

            p_alpha_local = (self.p_alpha[idx_i] + self.p_alpha[idx_i+1] + self.p_alpha[idx_k]) / 3.0
            p_beta_local = (self.p_beta[idx_i] + self.p_beta[idx_i+1] + self.p_beta[idx_k]) / 3.0

            term_alpha = p_alpha_local * (d_2 - 5.4)**2
            term_beta = p_beta_local * (d_2 - 6.6)**2
            torsion_2_e = self.weights["torsion"] * np.sum(term_alpha + term_beta)
            total_e += torsion_2_e

            mag_2 = 2 * self.weights["torsion"] * (p_alpha_local * (d_2 - 5.4) + p_beta_local * (d_2 - 6.6)) / d_2
            np.add.at(grad, idx_i, mag_2[:, np.newaxis] * diff_2)
            np.add.at(grad, idx_k, -mag_2[:, np.newaxis] * diff_2)

        if N > 3:
            idx_i = np.arange(N - 3)
            idx_k = idx_i + 3
            diff_3 = coords[idx_i] - coords[idx_k]
            d_3 = np.linalg.norm(diff_3, axis=1) + 1e-9

            p_alpha_local = (self.p_alpha[idx_i] + self.p_alpha[idx_i+1] + self.p_alpha[idx_i+2] + self.p_alpha[idx_k]) / 4.0
            p_beta_local = (self.p_beta[idx_i] + self.p_beta[idx_i+1] + self.p_beta[idx_i+2] + self.p_beta[idx_k]) / 4.0

            term_alpha = p_alpha_local * (d_3 - 5.1)**2
            term_beta = p_beta_local * (d_3 - 9.8)**2
            torsion_3_e = self.weights["torsion"] * np.sum(term_alpha + term_beta)
            total_e += torsion_3_e

            mag_3 = 2 * self.weights["torsion"] * (p_alpha_local * (d_3 - 5.1) + p_beta_local * (d_3 - 9.8)) / d_3
            np.add.at(grad, idx_i, mag_3[:, np.newaxis] * diff_3)
            np.add.at(grad, idx_k, -mag_3[:, np.newaxis] * diff_3)

        # 8. Helix i to i+4 Hydrogen bonding
        if N > 4:
            idx_i = np.arange(N - 4)
            idx_k = idx_i + 4
            diff_4 = coords[idx_i] - coords[idx_k]
            d_4 = np.linalg.norm(diff_4, axis=1) + 1e-9

            p_alpha_prod = self.p_alpha[idx_i] * self.p_alpha[idx_k]
            helix_e = self.weights["helix"] * np.sum(p_alpha_prod * (d_4 - 6.2)**2)
            total_e += helix_e

            mag_4 = 2 * self.weights["helix"] * p_alpha_prod * (d_4 - 6.2) / d_4
            np.add.at(grad, idx_i, mag_4[:, np.newaxis] * diff_4)
            np.add.at(grad, idx_k, -mag_4[:, np.newaxis] * diff_4)

        # 9. Radius of Gyration
        mean_coords = np.mean(coords, axis=0)
        rel_coords = coords - mean_coords
        rg = np.sqrt(np.mean(np.sum(rel_coords**2, axis=1)) + 1e-9)
        conf_e = self.weights["rg"] * (rg - self.RG_TARGET)**2
        total_e += conf_e
        grad += (2.0 * self.weights["rg"] * (rg - self.RG_TARGET) / (rg * N + 1e-9)) * rel_coords

        return total_e, grad.flatten()

    def optimize(self, max_iter: int = 500) -> np.ndarray:
        """Run L-BFGS-B minimization."""
        res = minimize(
            self.energy_and_gradient,
            self.x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "gtol": 1e-5},
        )
        self.x0 = res.x
        return res.x.reshape(-1, 3)

    def generate_all_atom(self, ca_coords: np.ndarray) -> dict:
        """Flesh out CA coordinates to full atom representation."""
        ca_coords = ca_coords.reshape(-1, 3)
        all_coords = []
        atom_types = []
        res_indices = []
        res_names = []

        for i, aa in enumerate(self.sequence):
            if i > 0 and i < self.N_res - 1:
                v_prev = ca_coords[i] - ca_coords[i - 1]
                v_next = ca_coords[i + 1] - ca_coords[i]
                t = (v_prev + v_next) / (np.linalg.norm(v_prev + v_next) + 1e-9)
                n = np.cross(v_prev, v_next)
                n /= np.linalg.norm(n) + 1e-9
                b = np.cross(t, n)
                rot = np.column_stack((t, n, b))
                phi = np.arctan2(n[1], n[0])
                psi = np.arctan2(b[2], b[1])
            else:
                rot = np.eye(3)
                phi, psi = 0.0, 0.0

            res_dict = self.atom_lib.get_full_residue(
                aa, ca_coords[i], rotation_matrix=rot, phi=phi, psi=psi
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
