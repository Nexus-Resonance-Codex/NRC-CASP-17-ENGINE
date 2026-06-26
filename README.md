---
title: Resonance Fold
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
app_port: 7860
pinned: false
---

# NRC-CASP-17-ENGINE: Pure Math Protein Folding Engine

Welcome to the official repository of the **NRC-CASP-17-ENGINE** (Resonance-Fold). This repository houses the high-fidelity, deterministic protein structure prediction engine built on the **Nexus Resonance Codex (NRC)** framework.

This engine achieved exceptional results in the CASP-17 competition by relying entirely on deterministic polymer physics, $\phi$-spiral mathematics, Trageser Tensor Theorem (TTT-7) stability verification, and Lattice-Parity Embeddings (LPE), with **zero runtime AI inference or external API dependencies**.

---

## 🏛 Core Architectural Concepts

1. **$\phi$-Spiral Lattice Initialization**: Generates uniform coordinates utilizing golden-angle spirals ($\phi = \frac{1 + \sqrt{5}}{2}$) as structural seeds.
2. **TTT-7 Stability Auditor**: Enforces that all coordinates and energy states lie on stable digital roots $\{1, 2, 4, 5, 7, 8\}$ while strictly avoiding the Chaotic Void $\{3, 6, 9\}$.
3. **Covariant All-Atom Projection**: Reconstructs complete physical atomic frameworks from C-alpha coordinates using local torsion-angle kinematics.
4. **Resonance Forcefield Relaxation**: An analytical L-BFGS-B minimizer designed with custom potentials for steric repulsion and structural constraints.

---

## 🧬 Repository Structure

```
NRC-CASP-17-ENGINE/
├── pyproject.toml         # Package definition & dependencies
├── requirements.txt       # Environment requirements
├── Dockerfile             # Docker recipe for Hugging Face Spaces
├── app.py                 # Gradio interactive dashboard
├── LICENSE.md             # Dual-license (CC BY-NC-SA 4.0 & Commercial)
├── src/
│   └── nrc_casp17_engine/
│       ├── __init__.py    # Package init and exports
│       ├── engine.py      # Core polymer folding kinematics
│       ├── forcefield.py  # Minimizer and analytical force calculations
│       ├── atoms.py       # Full-atom amino acid relative library
│       ├── chemistry.py   # Parameter sets and TTT-7 modulators
│       ├── biophysics.py  # Research metrics (pI, DSSP, pockets, RMSD)
│       └── ttt7.py        # TTT-7 digital root auditor
├── tests/                 # Unit testing suite
└── examples/              # API usage demonstrations
```

---

## 🚀 Installation & Usage

### Setup Environment
Install the package in editable mode:
```bash
pip install -e .
```

### Library Usage
You can utilize the core engine programmatically in your research:

```python
from nrc_casp17_engine import NRCEngine, BiophysicsSuite, TTT7Auditor

# Initialize the engine
engine = NRCEngine()

# Run pure math folding
sequence = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGG"
folding_trajectory = engine.fold_sequence(sequence)

# Get the final structure
final_result = None
for frame in folding_trajectory:
    final_result = frame

print("Folding completed!")
print(f"Total Atoms: {len(final_result['coords'])}")

# Run TTT-7 Audit
audit_res = TTT7Auditor.audit_coordinates(final_result['coords'])
print(f"TTT-7 Status: {audit_res['status']} ({audit_res['stable_percentage']:.2f}% stable)")
```

### Running the Gradio App
To start the local interactive visualizer:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:7860`.

---

## ⚖️ License
This project is licensed under a **Dual-License**:
- **CC BY-NC-SA 4.0** for academic, non-commercial, and humanitarian research.
- **Commercial License** for enterprise deployment or drug discovery (contact the authors for licensing terms).
See [LICENSE.md](LICENSE.md) for full terms.
