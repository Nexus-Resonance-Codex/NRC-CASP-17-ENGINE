#!/usr/bin/env python3
"""
Example Script: Programmatic Protein Folding using NRC-CASP-17-ENGINE
===================================================================

This script demonstrates how to use the pure math polymer physics engine
to fold a target sequence, perform biophysical analysis, and audit the
resulting 3D structure for TTT-7 stability.
"""

import os
import sys

# Add project root to path for local execution without installation
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.nrc_casp17_engine import NRCEngine, BiophysicsSuite, TTT7Auditor

def main():
    # 1. Define sequence (Vasopressin: 9 residues)
    sequence = "CYFQNCPRG"
    print(f"🧬 Targeting Sequence: {sequence}")
    
    # 2. Initialize Engine
    print("🏛 Initializing NRC Polymer Physics Engine...")
    engine = NRCEngine()
    
    # 3. Fold Sequence (Pure Math Relaxation)
    print("⏳ Relaxing sequence along φ-spiral lattice...")
    trajectory = engine.fold_sequence(sequence, steps=40)
    
    final_frame = None
    for frame in trajectory:
        final_frame = frame
        
    coords = final_frame["coords"]
    confidence = final_frame["confidence"]
    print(f"✅ Folding completed! Generated {len(coords)} atoms.")
    
    # 4. Biophysical Characterization
    print("\n⚡ Running Biophysics Analysis Suite...")
    analysis = BiophysicsSuite.analyze_sequence(sequence, coords, confidence)
    print(f" - Isoelectric Point (pI): {analysis['pI']:.2f}")
    print(f" - Secondary Structure (DSSP): {''.join(analysis['dssp'])}")
    print(f" - Resonance Error: {analysis['resonance_error']:.4f}")
    
    # 5. TTT-7 Stability Audit
    print("\n🛡 Running TTT-7 Stability Audit...")
    audit = TTT7Auditor.audit_coordinates(coords)
    print(f" - TTT-7 Parity Status: {audit['status']}")
    print(f" - Stable Coordinate Ratio: {audit['stable_percentage']:.2f}%")
    
    if audit["is_stable"]:
        print("🎉 Structure resides entirely in the stable mathematical manifold!")
    else:
        print("⚠️ Minor chaotic fluctuations detected in high-dimensional coordinates.")

if __name__ == "__main__":
    main()
