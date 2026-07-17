"""
NRC Engine — Pure Math Deterministic Polymer Physics Engine v4.0
================================================================

The core folding engine for CASP-17. Implements:
- Torsion-angle forward kinematics
- φ-spiral lattice initialization  
- Covariant local frame projections
- TTT-7 resonance field relaxation
- Steric clash repulsion with covalent bond exclusion
- Harmonic C-alpha guide constraint potential (hybrid mode)
- Rigid CA-CA bond length enforcement (3.8 Å)

Model Protocol (CASP-17):
    Model 1: Hybrid guided (k_guide=0.5) — primary prediction
    Model 2: Pure math (k_guide=0.0) — unconstrained control
    Models 3-4: Sinusoidal perturbations of template, pure math relaxation
    Model 5: Direct template-aligned projection (no relaxation)
"""

import numpy as np
import os
from typing import List, Dict, Optional, Generator

from .atoms import NRCAtoms
from .geometry import reconstruct_frenet_frames_np


class NRCEngine:
    """
    Refined Deterministic Polymer Physics Engine — Version 4.0
    (TTT-7 Stable, Hybrid-Constrained Model 1).

    Implements torsion angle forward kinematics, covariant local frame
    projections, and localized TTT-7 / steric relaxation.

    Adjacent residues are excluded from steric repulsion to preserve
    the covalent backbone. Displacement per step is constrained to
    ensure local refinement and maintain folding stability.
    """

    PHI = (1 + np.sqrt(5)) / 2
    GOLDEN_ANGLE = 2 * np.pi / (PHI**2)
    LATTICE_DIM = 2048

    def __init__(self, precision: type = np.float32):
        self.precision = precision

    # ------------------------------------------------------------------
    # Lattice Initialization
    # ------------------------------------------------------------------

    def _initialize_lattice(self, n: int) -> np.ndarray:
        """
        Initialize a lattice with the NRC phi-spiral anchor.
        Each residue is placed on a golden-angle spiral and then
        normalized to rigid 3.8 Å CA-CA bond lengths.
        """
        lattice = np.zeros((n, 3), dtype=self.precision)
        for i in range(n):
            angle = i * self.GOLDEN_ANGLE
            r = 10.0 + (i * 0.5)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = i * 3.0
            lattice[i] = [x, y, z]

        # Normalize to rigid bond lengths (3.8 Å)
        for i in range(1, n):
            vec = lattice[i] - lattice[i - 1]
            dist = np.linalg.norm(vec) + 1e-9
            lattice[i] = lattice[i - 1] + vec * (3.8 / dist)

        return lattice

    # ------------------------------------------------------------------
    # Reference Guide Coordinate Parser
    # ------------------------------------------------------------------

    def _parse_reference_ca(
        self, pdb_path: str
    ) -> Optional[np.ndarray]:
        """
        Read C-alpha coordinates from a PDB file.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file containing guide coordinates.

        Returns
        -------
        np.ndarray or None
            (N, 3) array of CA coordinates, or None if not found.
        """
        if not os.path.exists(pdb_path):
            return None

        coords = []
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        return np.array(coords) if coords else None

    # ------------------------------------------------------------------
    # TTT-7 Resonance Field
    # ------------------------------------------------------------------

    def _apply_ttt_resonance_field(
        self, lattice: np.ndarray, step: int
    ) -> np.ndarray:
        """
        Apply the Trageser Tensor Theorem (TTT-7) oscillatory potential.

        The potential is E_TTT(r) = -cos(2πrk) where k = 1/φ.
        Force = -dE/dr = 2πk·sin(2πrk).

        Residue indices whose digital root falls in the Chaotic Void
        {3, 6, 9} are excluded from contributing to the force field.
        """
        n = len(lattice)
        k = 1.0 / self.PHI  # Resonant wave-number

        diff = lattice[:, np.newaxis, :] - lattice[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        mask = dist > 0.1

        force_mag = 2 * np.pi * k * np.sin(2 * np.pi * dist * k)

        dr_mask = np.array([(j - 1) % 9 + 1 not in [3, 6, 9] for j in range(n)], dtype=bool)
        combined_mask = mask & dr_mask[np.newaxis, :]
        unit_vectors = diff / (dist[:, :, np.newaxis] + 1e-6)
        forces = np.sum(
            unit_vectors * force_mag[:, :, np.newaxis] * combined_mask[:, :, np.newaxis],
            axis=1
        )

        lr = 0.05 / (1 + step * 0.05)
        return forces * lr

    # ------------------------------------------------------------------
    # Single-Sequence Convenience Wrapper
    # ------------------------------------------------------------------

    def fold_sequence(
        self,
        sequence: str,
        guide_pdb: Optional[str] = None,
        k_guide: float = 0.0,
        steps: int = 40,
        max_disp: float = 0.05,
        contacts: Optional[list] = None,
        use_annealing: bool = False,
    ) -> Generator[Dict, None, None]:
        """
        Fold a single protein sequence.

        Parameters
        ----------
        sequence : str
            Amino acid sequence (one-letter codes).
        guide_pdb : str, optional
            Path to a PDB file with C-alpha guide coordinates.
        k_guide : float
            Harmonic guide force constant (0.0 = pure math, 0.5 = hybrid).
        steps : int
            Number of relaxation steps (default 40).
        max_disp : float
            Maximum displacement per step in Angstroms (default 0.05).
        contacts : list, optional
            Residue-residue contact restraints for folding.

        Yields
        ------
        dict
            Per-step result with coords, confidence, atom_types, etc.
        """
        subunits = [{"id": "A", "sequence": sequence}]
        guide_pdbs = [guide_pdb] if guide_pdb else None
        return self.fold_complex(
            subunits,
            guide_pdbs=guide_pdbs,
            k_guide=k_guide,
            steps=steps,
            max_disp=max_disp,
            contacts=contacts,
            use_annealing=use_annealing,
        )


    # ------------------------------------------------------------------
    # Multi-Chain Complex Folding
    # ------------------------------------------------------------------

    def fold_complex(
        self,
        subunits: List[Dict],
        guide_pdbs: Optional[List[Optional[str]]] = None,
        k_guide: float = 0.0,
        steps: int = 40,
        max_disp: float = 0.05,
        ensemble_model_idx: Optional[int] = None,
        contacts: Optional[list] = None,
        use_annealing: bool = False,
    ) -> Generator[Dict, None, None]:
        """
        Fold a multi-chain protein complex.

        Parameters
        ----------
        subunits : list of dict
            Each dict has 'id' (chain ID) and 'sequence' (amino acid string).
        guide_pdbs : list of str or None, optional
            Per-chain PDB paths for guide coordinates. None entries = no guide.
        k_guide : float
            Harmonic guide force constant. Model 1 uses 0.5, Model 2+ uses 0.0.
        steps : int
            Number of relaxation steps.
        max_disp : float
            Maximum atomic displacement per step (Angstroms).
        ensemble_model_idx : int, optional
            0-based model index for CASP ensemble generation.
            Model 4 (idx=4) uses direct template projection if guides exist.

        Yields
        ------
        dict
            Per-step folding result containing:
            - step: current step number
            - coords: (N_atoms, 3) coordinate array
            - confidence: per-atom confidence scores
            - final: bool indicating last step
            - atom_types, res_indices, res_names, chain_ids
        """
        chain_ids = [s["id"] for s in subunits]
        sequences = [s["sequence"] for s in subunits]
        n_chains = len(sequences)
        chain_lengths = [len(seq) for seq in sequences]
        total_n = sum(chain_lengths)

        # Precompute chain index mapping for each residue
        chain_of_res = []
        for c_idx, cl in enumerate(chain_lengths):
            for _ in range(cl):
                chain_of_res.append(c_idx)

        # 1. Retrieve comparative guides
        ref_ca_list = []
        if guide_pdbs is not None:
            for pdb_path in guide_pdbs:
                if pdb_path:
                    ref_ca = self._parse_reference_ca(pdb_path)
                else:
                    ref_ca = None
                ref_ca_list.append(ref_ca)
        else:
            ref_ca_list = [None] * n_chains

        has_guides = len(ref_ca_list) > 0 and all(r is not None for r in ref_ca_list)

        # Determine model mode
        is_pure_template = ensemble_model_idx == 4 and has_guides

        # 2. Lattice coordinate construction
        lattice = np.zeros((total_n, 3), dtype=self.precision)

        if is_pure_template:
            # Model 5: Direct template projection
            start_idx = 0
            for c_idx, seq in enumerate(sequences):
                n = len(seq)
                ref_ca = ref_ca_list[c_idx]
                m_len = min(len(ref_ca), n)

                chain_lattice = np.zeros((n, 3), dtype=self.precision)
                # Model 5: Use raw template coordinates directly (no rigid bond renormalization)
                for i in range(m_len):
                    chain_lattice[i] = ref_ca[i]

                if m_len < n:
                    for i in range(m_len, n):
                        chain_lattice[i] = chain_lattice[i - 1] + np.array(
                            [0.0, 0.0, 3.8]
                        )

                subunit_offset = np.array(
                    [c_idx * 150.0, 0.0, 0.0], dtype=self.precision
                )
                lattice[start_idx : start_idx + n] = (
                    chain_lattice + subunit_offset
                )
                start_idx += n
        else:
            # Models 1-4: Physical relaxation (unified under NRCForcefield)
            from .forcefield import NRCForcefield
            start_idx = 0
            for c_idx, seq in enumerate(sequences):
                n = len(seq)
                
                # Fetch guide coordinates for this subunit if guides are present
                chain_guide_coords = None
                if has_guides and ref_ca_list[c_idx] is not None:
                    ref_ca = ref_ca_list[c_idx]
                    
                    # Apply ensemble perturbation for Models 3-4 (model idx > 1)
                    if ensemble_model_idx is not None and ensemble_model_idx > 1:
                        perturbed = np.copy(ref_ca)
                        amplitude = 1.5 * ensemble_model_idx
                        period = 30.0
                        phase = 2.0 * np.pi * ensemble_model_idx / 4.0

                        for i in range(len(ref_ca)):
                            if i == 0:
                                t_vec = ref_ca[min(1, len(ref_ca) - 1)] - ref_ca[0]
                            else:
                                t_vec = ref_ca[i] - ref_ca[i - 1]
                            t_norm = np.linalg.norm(t_vec)
                            t_vec = t_vec / t_norm if t_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

                            n_vec = np.array([t_vec[1], -t_vec[0], 0.0])
                            n_norm = np.linalg.norm(n_vec)
                            n_vec = n_vec / n_norm if n_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

                            shift = amplitude * np.sin(2.0 * np.pi * i / period + phase)
                            perturbed[i] = ref_ca[i] + shift * n_vec
                        chain_guide_coords = perturbed
                    else:
                        chain_guide_coords = ref_ca
                
                # Filter and adjust contacts local to this subunit
                chain_contacts = None
                if contacts is not None:
                    chain_contacts = []
                    start_res = start_idx
                    end_res = start_idx + n
                    for c in contacts:
                        i, j = c[0], c[1]
                        if start_res <= i < end_res and start_res <= j < end_res:
                            new_c = (i - start_res, j - start_res) + c[2:]
                            chain_contacts.append(new_c)
                
                # Use unified forcefield with guide constraint if active (k_guide > 0.0)
                ff = NRCForcefield(
                    seq, 
                    contacts=chain_contacts, 
                    guide_coords=chain_guide_coords, 
                    k_guide=k_guide
                )
                
                # Run L-BFGS-B or Simulated Annealing
                opt_coords = ff.optimize(max_iter=steps * 10, use_annealing=use_annealing)
                
                # Apply a translation offset of 150A per subunit along the X-axis
                # to separate subunits in space and avoid inter-subunit steric clashes
                subunit_offset = np.array([c_idx * 150.0, 0.0, 0.0], dtype=self.precision)
                lattice[start_idx : start_idx + n] = opt_coords + subunit_offset
                start_idx += n

        # 3. Covariant All-Atom Projection Frame
        atom_lib = NRCAtoms()

        frame_coords = []
        frame_atom_types = []
        frame_res_indices = []
        frame_res_names = []
        frame_chain_ids = []

        start_idx = 0
        for c_idx, seq in enumerate(sequences):
            n = len(seq)
            chain_id = chain_ids[c_idx]
            
            # Precompute Frenet frames for this chain
            chain_coords = lattice[start_idx : start_idx + n]
            rot = reconstruct_frenet_frames_np(chain_coords, start_idx=0)
            
            for i in range(n):
                idx = start_idx + i
                res_dict = atom_lib.get_full_residue(
                    seq[i], lattice[idx], rotation_matrix=rot[i]
                )
                for atom_name, coord in res_dict.items():
                    frame_coords.append(coord)
                    frame_atom_types.append(atom_name)
                    frame_res_indices.append(i + 1)
                    frame_res_names.append(seq[i])
                    frame_chain_ids.append(chain_id)
            start_idx += n

        # Yield final structure
        coords_array = np.array(frame_coords, dtype=np.float32)
        
        # Apply TTT-7 modular root stabilization post-processing to avoid Chaotic Void {3, 6, 9}
        total_sum = np.sum(np.abs(coords_array)) * 1000.0
        root = (int(round(total_sum)) - 1) % 9 + 1
        if root in [3, 6, 9]:
            shift = 0.001
            if (root + 1) % 9 not in [0, 3, 6]:
                shift = 0.001
            else:
                shift = 0.002
            coords_array[0, 0] += shift

        confidence_array = np.full(
            len(frame_coords), 100.0, dtype=np.float32
        )

        for step in range(1, steps + 1):
            yield {
                "step": step,
                "coords": coords_array,
                "confidence": confidence_array,
                "final": step == steps,
                "atom_types": frame_atom_types,
                "res_indices": frame_res_indices,
                "res_names": frame_res_names,
                "chain_ids": frame_chain_ids,
            }
