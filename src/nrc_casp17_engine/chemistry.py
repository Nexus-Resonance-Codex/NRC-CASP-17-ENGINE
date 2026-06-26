"""
NRC Chemistry — AMBER ff99SB-Inspired Residue Parameters
========================================================

Maps amino acids to Lennard-Jones parameters (sigma, epsilon)
and partial charges, modulated by the TTT-7 φ-resonance field.
"""

import numpy as np


class NRCChemistry:
    """
    NRC Chemistry Manifold: Maps amino acids to physical properties
    anchored by AMBER ff99SB and modulated by TTT-7 resonance.
    """

    def __init__(self):
        # Base AMBER-like parameters (Sigma in Å, Epsilon in kcal/mol)
        self.residue_data = {
            "A": {"name": "ALA", "sigma": 3.40, "epsilon": 0.170, "charge": 0.0},
            "R": {"name": "ARG", "sigma": 3.50, "epsilon": 0.200, "charge": 1.0},
            "N": {"name": "ASN", "sigma": 3.30, "epsilon": 0.170, "charge": 0.0},
            "D": {"name": "ASP", "sigma": 3.30, "epsilon": 0.170, "charge": -1.0},
            "C": {"name": "CYS", "sigma": 3.50, "epsilon": 0.250, "charge": 0.0},
            "Q": {"name": "GLN", "sigma": 3.40, "epsilon": 0.200, "charge": 0.0},
            "E": {"name": "GLU", "sigma": 3.40, "epsilon": 0.200, "charge": -1.0},
            "G": {"name": "GLY", "sigma": 2.50, "epsilon": 0.050, "charge": 0.0},
            "H": {"name": "HIS", "sigma": 3.40, "epsilon": 0.170, "charge": 0.1},
            "I": {"name": "ILE", "sigma": 3.70, "epsilon": 0.250, "charge": 0.0},
            "L": {"name": "LEU", "sigma": 3.70, "epsilon": 0.250, "charge": 0.0},
            "K": {"name": "LYS", "sigma": 3.50, "epsilon": 0.200, "charge": 1.0},
            "M": {"name": "MET", "sigma": 3.50, "epsilon": 0.200, "charge": 0.0},
            "F": {"name": "PHE", "sigma": 3.70, "epsilon": 0.250, "charge": 0.0},
            "P": {"name": "PRO", "sigma": 3.40, "epsilon": 0.200, "charge": 0.0},
            "S": {"name": "SER", "sigma": 3.30, "epsilon": 0.170, "charge": 0.0},
            "T": {"name": "THR", "sigma": 3.40, "epsilon": 0.170, "charge": 0.0},
            "W": {"name": "TRP", "sigma": 3.70, "epsilon": 0.300, "charge": 0.0},
            "Y": {"name": "TYR", "sigma": 3.70, "epsilon": 0.250, "charge": 0.0},
            "V": {"name": "VAL", "sigma": 3.50, "epsilon": 0.200, "charge": 0.0},
        }

    def get_params(self, sequence: str) -> tuple:
        """
        Returns vectorized LJ parameters for a sequence.

        Parameters
        ----------
        sequence : str
            Amino acid sequence (one-letter codes).

        Returns
        -------
        tuple of (np.ndarray, np.ndarray, np.ndarray)
            (sigmas, epsilons, charges) arrays.
        """
        sigmas = []
        epsilons = []
        charges = []

        for aa in sequence:
            data = self.residue_data.get(aa, self.residue_data["G"])
            idx = len(sigmas)
            res_factor = 1.0 + 0.01 * np.sin(2 * np.pi * idx / 1.618)

            sigmas.append(data["sigma"] * res_factor)
            epsilons.append(data["epsilon"])
            charges.append(data["charge"])

        return np.array(sigmas), np.array(epsilons), np.array(charges)

    def get_three_letter(self, aa: str) -> str:
        """Convert single-letter amino acid code to three-letter code."""
        data = self.residue_data.get(aa)
        return data["name"] if data else "UNK"
