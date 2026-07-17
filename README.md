---
title: NRC-CASP-17-ENGINE
sdk: docker
app_file: app.py
app_port: 7860
pinned: false
---

# ⚖️ Licensing & Enterprise Architecture (Nexus Resonance Codex (NRC))

This repository operates on a **Dual-License** structure to protect the integrity of the Nexus Resonance Codex (NRC) while supporting open scientific validation.

* **Codebase:** [AGPL-3.0](LICENSE) (Ensures any cloud/network deployment remains entirely open-source).
* **Data & Weights:** [CC BY-NC-SA 4.0](LICENSE-DATA) (Strictly prohibits free commercial use).
* **Trademarks & Math Integrity:** [Trademark Policy](TRADEMARK_POLICY.md) (Protects the TTT-7, QRT, and mathematical nomenclatures).
* **Patent Protection:** [Patent Pledge](PATENT_PLEDGE.md) (Good Faith Patent Covenant).

🏢 **Corporate & Drug Discovery Entities:** If you require the use of Nexus Resonance Codex (NRC) frameworks in a closed-source, proprietary, or for-profit environment, you must purchase an Enterprise License. See [COMMERCIAL_USE.md](COMMERCIAL_USE.md) for details or contact **James Paul Trageser** at **NexusResonanceCodex@gmail.com**.

---

## 🏛️ Theoretical Foundations

The engine uses three key mathematical and physical frameworks to model protein backbones and fold topologies:

### 1. The QRT Resonance Attractor
In helical polymers, the relationship between the helix radius, pitch, and monomer rise can be mapped onto a discrete coordinate lattice. The QRT (Quantum Resonance Tensor) projection identifies the optimal packing attractor angle to maximize packing density in a high-dimensional space:
\[
\theta_{QRT} = \arctan\left(\frac{4}{\pi}\right) \approx 51.82729^\circ
\]
The engine enforces this angle constraint in its geometry modules, locking alpha-helical segments to this exact mathematical attractor to maximize packing density and structural stability.

### 2. Debye-Hückel Electrostatics Screening
In physiological aqueous solutions, charges are screened by mobile counter-ions (salt). To model this, the engine uses the Debye-Hückel potential to represent electrostatic interactions:
\[
U(d) = \frac{q_1 q_2}{d} e^{-\kappa d}
\]
where $\kappa = 0.1 \text{ Å}^{-1}$ represents the physiological screening parameter ($\sim 150\text{ mM}$ salt). This prevents unphysical repulsion between charged atoms at short distances and ensures accurate solvent behavior.

### 3. Flory Compaction Target Scaling
To prevent over-compaction in small domains and under-compaction in large sequences, the target radius of gyration ($R_g$) is dynamically scaled according to the empirical Flory scaling relationship for globular protein conformations:
\[
R_g = 2.2 \times N^{0.38}
\]
where $N$ is the number of residues.

### 4. TTT-7 Stability Mandate (Trageser Tensor Theorem)
Coordinates and sum potentials are audited using a modular digital root function:
\[
dr(n) = (n - 1) \bmod 9 + 1
\]
Lattice-parity embeddings are validated by ensuring that the digital root of the structural coordinates resides inside the stable resonant set $\{1, 2, 4, 5, 7, 8\}$, while strictly avoiding the Chaotic Void $\{3, 6, 9\}$ associated with high-entropy, unstable conformations.

---

## 🗂️ Module-by-Module Code Documentation

The engine is modularized to decouple forward kinematics, energy calculations, optimization, and biophysical reporting. Below is a detailed breakdown of each script and its component functions:

### 1. `geometry.py` — Backbone Kinematics & Coordinate Projection
This module acts as the forward kinematics engine, converting internal coordinates (bond lengths, bond angles, and dihedrals) into 3D Cartesian coordinates.
* **`FrenetFrameReconstructor`**: Implements the Frenet-Serret equations to build the protein backbone from local curvature and torsion. It translates angular movements into a continuous chain of 3D coordinates.
* **`alpha_dihedral`**: Enforces the exact mathematical resonance, attractor:
  ```python
  alpha_dihedral = 51.82729 * np.pi / 180.0
  ```
* **`fold_backbone_torsion(sequence, angles)`**: Projects internal backbone angles to Cartesian space.

### 2. `energy.py` — The Analytical Potential Energy Landscape
This module implements the custom forcefield as a differentiable PyTorch graph, allowing analytical gradient calculation via backpropagation.
* **`NRCForcefield`**: Evaluates the total potential energy of a coordinate set. It computes:
  - **`bond_constraint`**: Penalizes deviation of adjacent $C_\alpha-C_\alpha$ bonds from the rigid $3.8\text{ Å}$ backbone scale.
  - **`non_bonded_steric`**: Computes steric repulsion between non-bonded atoms using a $1/d^{12}$ Lennard-Jones-like repulsive potential.
  - **`hydrophobic_collapse`**: Models hydrophobic attraction using rational and Gaussian wells centered at $6.5\text{–}7.0\text{ Å}$.
  - **`debye_huckel_elec`**: Computes screened electrostatic energy using the Debye-Hückel potential.
  - **`flory_rg_penalty`**: Encourages compaction matching the empirical Flory exponent ($2.2 \times N^{0.38}$).
  - **`ttt7_resonance`**: Applies a penalty field to steer coordinates away from digital roots in the Chaotic Void.

### 3. `optimizer.py` — High-Resolution Numerical Minimization
Wraps optimization routines to minimize the analytical energy graph.
* **`minimize_lbfgs(x0, max_iter)`**: Executes L-BFGS-B minimization with tightened tolerances:
  ```python
  options = {"maxiter": max_iter, "gtol": 1e-7, "ftol": 1e-10}
  ```
  This ensures the structure settles into the deepest local energy minimum.

### 4. `forcefield.py` — The PyTorch-to-SciPy Bridge
Interprets coordinate inputs and interfaces the differentiable energy function with SciPy's numpy-based minimizers.
* **`EnergyMinimizerInterface`**: Projects coordinates to PyTorch tensors, runs forward evaluations to compute the loss, performs backpropagation to retrieve exact analytical gradients, and formats the outputs back into numpy arrays.

### 5. `engine.py` — Multi-Model Cohort Orchestrator
Orchestrates the 5-Model comparative folding pipeline to generate submissions.
* **`NRCEngine.fold_complex(subunits, guide_pdbs, k_guide, ...)`**: Coordinates the folding loop:
  - **Model 1**: Restrained refinement ($k_{\text{guide}} = 0.5$) to resolve local clashes while locking the backbone to the guide template.
  - **Model 2**: Free unconstrained relaxation ($k_{\text{guide}} = 0.0$).
  - **Models 3 & 4**: Perturbs the template coordinates with a sinusoidal function (amplitude $1.5\text{–}3.0\text{ Å}$) before running free relaxation.
  - **Model 5**: Direct template projection, using raw template coordinates directly (without rigid bond normalization) to prevent cumulative directional drift.

### 6. `ttt7.py` — Digital Root Auditor
Enforces the Trageser Tensor Theorem (TTT-7) parity and root validation rules.
* **`TTT7Auditor.audit_coordinates(coords)`**: Iterates over atoms, calculates their digital roots, and reports whether the structure satisfies LPE parity and avoids the Chaotic Void $\{3, 6, 9\}$.

### 7. `validation.py` — Physical Constraint Audits
Provides strict structural safety gates prior to file packaging.
* **`audit_physical_constraints(pdb_file)`**: Verifies that the minimum distance between non-adjacent atoms is $>1.10\text{ Å}$ (steric clash check) and that the peptide backbone bonds are intact ($3.8\text{ Å} \pm 0.15\text{ Å}$).

### 8. `atoms.py` — All-Atom Amino Acid Dictionaries
Tracks relative coordinates and properties of all heavy atoms inside amino acid sidechains.
* **`AllAtomMapper`**: Places sidechain heavy atoms relative to the $C_\alpha-C_\beta$ orientation vector.

### 9. `chemistry.py` — Structural & Chemical Dictionary
Houses structural metadata, including atomic van der Waals radii, formal charges, valency, and hydrophilic/hydrophobic classifications.

### 10. `biophysics.py` — Analysis Metrics Suite
Computes biophysical metrics to evaluate structural quality:
* **`DSSPAssigner`**: Predicts secondary structure assignments (Helix, Sheet, Loop) from hydrogen-bonding geometries.
* **`pICalculator`**: Computes the theoretical isoelectric point of the protein sequence.
* **`PocketFinder`**: Analyzes solvent-accessible pockets to identify potential ligand-binding cavities.

### 11. `reporting.py` — PDB File Writer
Writes refined coordinates into standard, compliant PDB format.
* **`ReportingSuite.generate_pdb(...)`**: Formats PDB lines (`ATOM`, coordinate spacing, residue names, chain identifiers), setting the temperature factor column to represent structural confidence metrics (e.g. pLDDT).

### 12. `deposition.py` — Gateway Submission Client
Integrates formatting and packaging routines to interact with CASP-17 prediction portals and Zenodo dataset archival APIs.

---

## 🚀 Installation & Developer Setup

### Environment Setup
Install the package in editable mode with development dependencies:
```bash
pip install -e .
```

### Run Local Unit Tests
Validate the installation by running the test suite:
```bash
python run_tests.py
```

### Launch Interactive Visualizer
To launch the local Gradio interface:
```bash
python app.py
```

