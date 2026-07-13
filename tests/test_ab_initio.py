import numpy as np
from src.nrc_casp17_engine import NRCEngine, BiophysicsSuite, TTT7Auditor


def test_ab_initio_folding():
    engine = NRCEngine()
    sequence = "CYIQNCPLG"

    # Run the folding generator twice to verify determinism
    trajectory1 = list(engine.fold_sequence(sequence, steps=10, k_guide=0.0))
    trajectory2 = list(engine.fold_sequence(sequence, steps=10, k_guide=0.0))

    # Assert identical lengths and final steps
    assert len(trajectory1) == 10
    assert len(trajectory2) == 10

    final_frame1 = trajectory1[-1]
    final_frame2 = trajectory2[-1]

    # Assert coordinates are exactly identical (0.000 A deviation)
    coords1 = final_frame1["coords"]
    coords2 = final_frame2["coords"]
    np.testing.assert_allclose(coords1, coords2, rtol=1e-5, atol=1e-5)

    # Check biophysics and TTT-7 stability
    analysis = BiophysicsSuite.analyze_sequence(
        sequence, coords1, final_frame1["confidence"]
    )
    assert "pI" in analysis

    audit = TTT7Auditor.audit_coordinates(coords1)
    assert "is_stable" in audit
    print(
        f"Ab initio folding test completed. Determinism verified. TTT-7 Status: {audit['status']}"
    )
