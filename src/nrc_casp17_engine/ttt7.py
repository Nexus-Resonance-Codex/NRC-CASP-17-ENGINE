"""
TTT-7 Auditor — Formal Stability Verification and Hallucination Suppression
==========================================================================

Implements the Trageser Tensor Theorem (TTT-7) digital root auditor.
Ensures all structural, sequence, and numerical manifolds maintain
resonance with the stable root set {1, 2, 4, 5, 7, 8} and avoid the
Chaotic Void {3, 6, 9}.
"""

import numpy as np
from typing import Dict, List, Union

class TTT7Auditor:
    """
    Formal Auditor for TTT-7 Stability.
    Provides rigorous digital root verification for sequences, coordinates,
    and scalar tensors.
    """

    STABLE_SET = {1, 2, 4, 5, 7, 8}
    CHAOTIC_VOID = {3, 6, 9}

    @staticmethod
    def digital_root(n: Union[int, float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Calculate the digital root of a number, float, or array.
        For floats, we sum the significant digits.
        """
        if isinstance(n, np.ndarray):
            # Element-wise digital root for array
            flat = np.abs(n).flatten()
            res = np.zeros_like(flat, dtype=np.int32)
            for idx, val in enumerate(flat):
                res[idx] = TTT7Auditor._scalar_digital_root(val)
            return res.reshape(n.shape)
        else:
            return TTT7Auditor._scalar_digital_root(n)

    @staticmethod
    def _scalar_digital_root(val: Union[int, float]) -> int:
        """Helper to compute digital root of a single scalar."""
        if isinstance(val, float):
            # Convert to string, ignore dot and sign, sum digits
            s = f"{abs(val):.6f}".replace(".", "")
            digit_sum = sum(int(c) for c in s if c.isdigit())
        else:
            digit_sum = abs(int(val))
            
        if digit_sum == 0:
            return 9
        return (digit_sum - 1) % 9 + 1

    @classmethod
    def audit_sequence(cls, sequence: str) -> Dict:
        """
        Audit a protein sequence by analyzing the ASCII digital roots
        of its amino acid constituents.
        """
        roots = [cls.digital_root(sum(map(ord, aa))) for aa in sequence]
        stable_count = sum(1 for r in roots if r in cls.STABLE_SET)
        chaotic_count = len(roots) - stable_count
        stable_pct = (stable_count / len(sequence) * 100.0) if sequence else 0.0

        return {
            "sequence_length": len(sequence),
            "roots": roots,
            "stable_count": stable_count,
            "chaotic_count": chaotic_count,
            "stable_percentage": stable_pct,
            "is_stable": chaotic_count == 0,
            "status": "STABLE" if chaotic_count == 0 else "CHAOTIC"
        }

    @classmethod
    def audit_coordinates(cls, coords: np.ndarray, tolerance: float = 1e-4) -> Dict:
        """
        Audit a 3D coordinate tensor to verify that all atomic positions
        comply with TTT-7 stability constraints.
        """
        flat_coords = coords.flatten()
        roots = cls.digital_root(flat_coords)
        
        stable_mask = np.isin(roots, list(cls.STABLE_SET))
        stable_count = np.sum(stable_mask)
        total_count = len(roots)
        stable_pct = (stable_count / total_count * 100.0) if total_count > 0 else 0.0

        chaotic_indices = np.where(~stable_mask)[0]
        
        return {
            "total_atoms": len(coords),
            "total_coordinates": total_count,
            "stable_count": int(stable_count),
            "chaotic_count": int(total_count - stable_count),
            "stable_percentage": float(stable_pct),
            "is_stable": stable_count == total_count,
            "status": "STABLE" if stable_count == total_count else "CHAOTIC",
            "chaotic_coordinate_fraction": float(1.0 - stable_pct / 100.0)
        }
