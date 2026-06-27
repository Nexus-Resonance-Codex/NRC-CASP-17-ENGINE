"""
Test Contact Restraints (Option B)
==================================

Verifies that adding a contact list forces distant residues to collapse 
together to satisfy co-evolutionary constraints.
"""

import numpy as np
from nrc_casp17_engine import NRCEngine

def test_contacts():
    # A simple 20-residue sequence
    sequence = "MTEYKLVVVGAGGVGKSALT"
    
    # Define contact constraints with extremely strong weights to force collapse
    contacts = [
        (0, 19, 6.0, 100.0),
        (4, 15, 6.0, 100.0)
    ]
    
    engine = NRCEngine()
    
    # 1. Fold WITHOUT contacts (control)
    from nrc_casp17_engine.forcefield import NRCForcefield
    
    ff_no = NRCForcefield(sequence)
    ca_no = ff_no.optimize(max_iter=500)
    dist_0_19_no = np.linalg.norm(ca_no[0] - ca_no[19])
    dist_4_15_no = np.linalg.norm(ca_no[4] - ca_no[15])
    
    # 2. Fold WITH contacts (scale the forcefield contact weight to 1000.0 to ensure strong attraction)
    ff_yes = NRCForcefield(sequence, weights={"contact": 1000.0}, contacts=contacts)
    ca_yes = ff_yes.optimize(max_iter=500)
    dist_0_19_yes = np.linalg.norm(ca_yes[0] - ca_yes[19])
    dist_4_15_yes = np.linalg.norm(ca_yes[4] - ca_yes[15])
    
    print(f"Without contacts: dist(0, 19) = {dist_0_19_no:.2f} A, dist(4, 15) = {dist_4_15_no:.2f} A")
    print(f"With contacts:    dist(0, 19) = {dist_0_19_yes:.2f} A, dist(4, 15) = {dist_4_15_yes:.2f} A")
    
    # Verify that the contacts are significantly closer in the constrained run
    assert abs(dist_0_19_yes - 6.0) < 1.0, "Residues 0 and 19 should satisfy target restraint of 6.0"
    assert abs(dist_4_15_yes - 6.0) < 1.0, "Residues 4 and 15 should satisfy target restraint of 6.0"
    assert dist_0_19_yes < 8.0, "Residues 0 and 19 should satisfy target restraint (< 8.0 A)"
    assert dist_4_15_yes < 8.0, "Residues 4 and 15 should satisfy target restraint (< 8.0 A)"
    
    print("Test passed: Contact restraints successfully guided folding collapse.")

if __name__ == "__main__":
    test_contacts()
