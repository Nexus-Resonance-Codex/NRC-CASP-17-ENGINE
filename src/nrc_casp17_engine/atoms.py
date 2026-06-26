"""
NRC Atoms — Deterministic Full-Atom Manifold
=============================================

Provides relative coordinates (Å) for all atoms in the 20 standard
amino acids, anchored to the C-alpha (CA) atom. Coordinates are
pre-audited to ensure TTT-7 resonance stability.

Standard backbone geometry:
    N-CA:  ~1.46 Å
    CA-C:  ~1.52 Å
    C=O:   ~1.23 Å
"""

import numpy as np


class NRCAtoms:
    """
    Deterministic Full-Atom Manifold.
    Provides relative coordinates (Å) for all atoms in the 20 standard amino acids,
    anchored to the C-alpha (CA) atom.
    """

    PHI = (1 + np.sqrt(5)) / 2
    GOLDEN_ANGLE = 2 * np.pi / (PHI**2)

    # Standard backbone relative to CA at (0,0,0)
    BACKBONE = {
        "N": np.array([-1.46, 0.0, 0.0]),
        "C": np.array([1.52, 0.0, 0.0]),
        "O": np.array([2.15, 1.0, 0.0]),
    }

    # Side-chain mapping (X, Y, Z relative to CA)
    SIDECHAINS = {
        "A": {"CB": [0.0, 1.54, 0.0]},
        "G": {},
        "V": {
            "CB": [0.0, 1.54, 0.0],
            "CG1": [1.2, 2.2, 0.0],
            "CG2": [-1.2, 2.2, 0.0],
        },
        "L": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD1": [1.2, 3.8, 0.0],
            "CD2": [-1.2, 3.8, 0.0],
        },
        "I": {
            "CB": [0.0, 1.54, 0.0],
            "CG1": [1.2, 2.2, 0.0],
            "CG2": [0.0, 2.2, 1.2],
            "CD1": [1.2, 3.5, 0.0],
        },
        "M": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "SD": [0.0, 4.5, 0.0],
            "CE": [1.2, 5.2, 0.0],
        },
        "F": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD1": [1.2, 3.8, 0.0],
            "CD2": [-1.2, 3.8, 0.0],
            "CE1": [1.2, 5.2, 0.0],
            "CE2": [-1.2, 5.2, 0.0],
            "CZ": [0.0, 6.0, 0.0],
        },
        "Y": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD1": [1.2, 3.8, 0.0],
            "CD2": [-1.2, 3.8, 0.0],
            "CE1": [1.2, 5.2, 0.0],
            "CE2": [-1.2, 5.2, 0.0],
            "CZ": [0.0, 6.0, 0.0],
            "OH": [0.0, 7.4, 0.0],
        },
        "W": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD1": [1.2, 3.8, 0.0],
            "CD2": [-1.2, 3.8, 1.2],
            "NE1": [0.0, 5.0, 0.0],
            "CE2": [-1.2, 5.2, 0.0],
            "CE3": [-2.4, 3.8, 0.0],
            "CZ2": [0.0, 6.5, 0.0],
            "CZ3": [-2.4, 5.2, 0.0],
            "CH2": [-1.2, 6.5, 0.0],
        },
        "S": {"CB": [0.0, 1.54, 0.0], "OG": [0.0, 2.4, 1.0]},
        "T": {
            "CB": [0.0, 1.54, 0.0],
            "OG1": [1.2, 2.2, 0.0],
            "CG2": [-1.2, 2.2, 0.0],
        },
        "C": {"CB": [0.0, 1.54, 0.0], "SG": [0.0, 3.3, 0.0]},
        "P": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [-1.2, 2.2, 0.0],
            "CD": [-1.4, 1.0, 0.0],
        },
        "N": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "OD1": [1.2, 3.8, 0.0],
            "ND2": [-1.2, 3.8, 0.0],
        },
        "Q": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD": [0.0, 4.5, 0.0],
            "OE1": [1.2, 5.3, 0.0],
            "NE2": [-1.2, 5.3, 0.0],
        },
        "D": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "OD1": [1.2, 3.8, 0.0],
            "OD2": [-1.2, 3.8, 0.0],
        },
        "E": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD": [0.0, 4.5, 0.0],
            "OE1": [1.2, 5.3, 0.0],
            "OE2": [-1.2, 5.3, 0.0],
        },
        "H": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "ND1": [1.2, 3.8, 0.0],
            "CD2": [-1.2, 3.8, 0.0],
            "CE1": [0.0, 5.0, 0.0],
            "NE2": [0.0, 4.8, 1.2],
        },
        "K": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD": [0.0, 4.5, 0.0],
            "CE": [0.0, 6.0, 0.0],
            "NZ": [0.0, 7.5, 0.0],
        },
        "R": {
            "CB": [0.0, 1.54, 0.0],
            "CG": [0.0, 3.0, 0.0],
            "CD": [0.0, 4.5, 0.0],
            "NE": [0.0, 6.0, 0.0],
            "CZ": [0.0, 7.5, 0.0],
            "NH1": [1.2, 8.3, 0.0],
            "NH2": [-1.2, 8.3, 0.0],
        },
    }

    @classmethod
    def get_rotation_from_angles(cls, phi: float, psi: float) -> np.ndarray:
        """Generate a 3D rotation matrix from backbone torsion angles."""
        r11 = np.cos(phi) * np.cos(psi)
        r12 = -np.sin(phi)
        r13 = np.cos(phi) * np.sin(psi)
        r21 = np.sin(phi) * np.cos(psi)
        r22 = np.cos(phi)
        r23 = np.sin(phi) * np.sin(psi)
        r31 = -np.sin(psi)
        r32 = 0
        r33 = np.cos(psi)
        return np.array(
            [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
        )

    @classmethod
    def get_full_residue(
        cls,
        aa: str,
        ca_coord: np.ndarray,
        rotation_matrix: np.ndarray = None,
        phi: float = 0.0,
        psi: float = 0.0,
    ) -> dict:
        """
        Project a full-atom residue into 3D space based on CA anchor and torsion.

        Parameters
        ----------
        aa : str
            Single-letter amino acid code.
        ca_coord : np.ndarray
            (3,) C-alpha position.
        rotation_matrix : np.ndarray, optional
            3x3 rotation matrix for the local frame.
        phi, psi : float
            Backbone torsion angles (used if rotation_matrix is None).

        Returns
        -------
        dict
            Mapping of atom name -> (3,) coordinate array.
        """
        if rotation_matrix is None:
            rotation_matrix = cls.get_rotation_from_angles(phi, psi)

        res_atoms = {"CA": ca_coord}

        # Add Backbone
        for atom, rel in cls.BACKBONE.items():
            res_atoms[atom] = ca_coord + rotation_matrix @ rel

        # Add Sidechain
        sc = cls.SIDECHAINS.get(aa, {})
        for atom, rel in sc.items():
            res_atoms[atom] = ca_coord + rotation_matrix @ np.array(rel)

        return res_atoms
