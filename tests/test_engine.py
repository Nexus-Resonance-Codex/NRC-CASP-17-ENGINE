import numpy as np
from src.nrc_casp17_engine import NRCEngine, BiophysicsSuite, TTT7Auditor

def test_engine_folding():
    # Initialize the engine
    engine = NRCEngine()
    
    # Use a short sequence (Oxytocin: 9 residues)
    sequence = "CYIQNCPLG"
    
    # Run the folding generator
    trajectory = list(engine.fold_sequence(sequence, steps=10))
    
    # Assert steps were run
    assert len(trajectory) == 10
    
    # Check final frame
    final_frame = trajectory[-1]
    assert final_frame["step"] == 10
    assert final_frame["final"] is True
    
    # Check coordinates shape and type
    coords = final_frame["coords"]
    assert isinstance(coords, np.ndarray)
    assert coords.ndim == 2
    assert coords.shape[1] == 3
    
    # Validate biophysical analysis
    analysis = BiophysicsSuite.analyze_sequence(sequence, coords, final_frame["confidence"])
    assert "pI" in analysis
    assert "hydropathy" in analysis
    assert "dssp" in analysis
    
    # Validate TTT-7 stability auditing
    audit = TTT7Auditor.audit_coordinates(coords)
    assert "is_stable" in audit
    assert "status" in audit
    print(f"Test completed. TTT-7 Status: {audit['status']}")
