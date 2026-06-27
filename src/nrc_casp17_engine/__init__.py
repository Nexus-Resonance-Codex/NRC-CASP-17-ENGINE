"""
NRC-CASP-17-ENGINE — Pure Math Protein Folding Engine
=====================================================

A deterministic, zero-AI-inference protein structure prediction engine
built on the Nexus Resonance Codex (NRC) framework. Validated against
CASP-17 competition targets with 100% reproducible results.

Core Modules:
    - engine: The main NRCEngine class for folding sequences
    - forcefield: L-BFGS-B optimized all-atom energy minimization
    - atoms: Full-atom residue coordinate library (20 standard amino acids)
    - chemistry: AMBER ff99SB-inspired residue parameters
    - biophysics: Research-grade analysis suite (pI, DSSP, Ramachandran, pockets)
    - formatter: CASP-compliant PDB/submission text formatter
    - ttt7: TTT-7 digital root stability auditor

Copyright © 2026 Nexus Resonance Codex Team. All Rights Reserved.
Licensed under CC BY-NC-SA 4.0 (see LICENSE.md).
"""

__version__ = "1.0.0"
__author__ = "Nexus Resonance Codex Team"
__license__ = "CC BY-NC-SA 4.0"

from .engine import NRCEngine
from .forcefield import NRCForcefield
from .atoms import NRCAtoms
from .chemistry import NRCChemistry
from .biophysics import BiophysicsSuite
from .ttt7 import TTT7Auditor
from .reporting import ReportingSuite
from .deposition import depositor

__all__ = [
    "NRCEngine",
    "NRCForcefield",
    "NRCAtoms",
    "NRCChemistry",
    "BiophysicsSuite",
    "TTT7Auditor",
    "ReportingSuite",
    "depositor",
]
