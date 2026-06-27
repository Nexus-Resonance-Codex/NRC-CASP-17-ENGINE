"""
validation.py — Automated benchmarking against native PDB structures
"""

import numpy as np
from .engine import NRCEngine
from .biophysics import BiophysicsSuite

def benchmark_folding(
    sequence: str, 
    native_pdb_id: str, 
    contacts: list = None, 
    use_annealing: bool = False
) -> dict:
    """
    Run ab initio folding and calculate CA RMSD to the native PDB structure.
    """
    engine = NRCEngine()
    
    # Run the folding generator to the final step
    steps = 40
    fold_gen = engine.fold_sequence(
        sequence, 
        k_guide=0.0, 
        steps=steps, 
        contacts=contacts
    )
    
    final_result = None
    for step in fold_gen:
        if step["final"]:
            final_result = step
            
    if final_result is None:
        raise ValueError("Folding failed to generate coordinates.")

    # Extract predicted CA coordinates
    pred_coords = []
    for coord, atom_type in zip(final_result["coords"], final_result["atom_types"]):
        if atom_type == "CA":
            pred_coords.append(coord)
            
    pred_coords = np.array(pred_coords)
    
    # Compare with native PDB
    comparison = BiophysicsSuite.compare_to_native(native_pdb_id, pred_coords)
    return comparison
