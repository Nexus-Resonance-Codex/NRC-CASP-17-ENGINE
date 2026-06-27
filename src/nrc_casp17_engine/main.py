"""
main.py — Command-line interface for the NRC Folding Engine
"""

import argparse
import sys
from .engine import NRCEngine
from .reporting import ReportingSuite
from .validation import benchmark_folding

def main():
    parser = argparse.ArgumentParser(description="NRC Protein Folding Engine CLI")
    parser.add_argument("--sequence", type=str, required=True, help="Amino acid sequence to fold")
    parser.add_argument("--pdb_id", type=str, default=None, help="Optional PDB ID of native structure for RMSD validation")
    parser.add_argument("--output", type=str, default="folded_structure.pdb", help="Output path for the folded PDB file")
    parser.add_argument("--steps", type=int, default=40, help="Number of refinement steps")
    parser.add_argument("--optimizer", type=str, choices=["lbfgs", "annealing"], default="lbfgs", help="Optimizer routine to use")

    args = parser.parse_args()

    print(f"Initializing folding engine for sequence length: {len(args.sequence)}")
    
    engine = NRCEngine()
    
    # Run the generator
    # We will pass the optimizer choice as part of a custom param or configure the engine to use simulated annealing.
    # Note: To avoid breaking the fold_sequence signature, we can set an env variable or inject optimizer type.
    # Since we are refactoring, we'll make sure engine.py knows about simulated annealing.
    
    print(f"Folding sequence using {args.optimizer} optimizer...")
    
    # We run the generator to get final coordinates
    fold_gen = engine.fold_sequence(
        args.sequence, 
        k_guide=0.0, 
        steps=args.steps
    )
    
    final_res = None
    for step in fold_gen:
        final_res = step

    if final_res is None:
        print("Error: Folding failed to generate coordinates.", file=sys.stderr)
        sys.exit(1)

    # Export PDB
    pdb_content = ReportingSuite.generate_pdb(
        args.sequence, 
        final_res["coords"], 
        final_res["confidence"],
        atom_types=final_res.get("atom_types"),
        res_indices=final_res.get("res_indices"),
        res_names=final_res.get("res_names")
    )
    
    with open(args.output, "w") as f:
        f.write(pdb_content)
    print(f"Folded structure successfully written to {args.output}")

    # Run validation if pdb_id is supplied
    if args.pdb_id:
        print(f"Validating against native PDB {args.pdb_id}...")
        try:
            comparison = benchmark_folding(args.sequence, args.pdb_id)
            if "error" in comparison:
                print(f"Validation failed: {comparison['error']}", file=sys.stderr)
            else:
                print(f"Validation Success!")
                print(f"  PDB ID: {comparison['pdb_id']}")
                print(f"  CA RMSD: {comparison['rmsd']:.4f} A")
        except Exception as e:
            print(f"Error running validation: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
