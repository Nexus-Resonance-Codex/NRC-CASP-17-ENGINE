"""
energy.py — Vectorized biophysical potentials with PyTorch autograd
"""

import numpy as np
import torch
from .geometry import reconstruct_backbone_frames_t

# Chou-Fasman helical (P_alpha) and sheet (P_beta) propensities
CHOU_FASMAN = {
    "A": (1.42, 0.83), "R": (0.98, 0.93), "N": (0.67, 0.89), "D": (1.01, 0.54),
    "C": (0.70, 1.19), "Q": (1.11, 1.10), "E": (1.51, 0.37), "G": (0.57, 0.75),
    "H": (1.00, 0.87), "I": (1.08, 1.60), "L": (1.21, 1.30), "K": (1.14, 0.74),
    "M": (1.45, 1.05), "F": (1.13, 1.38), "P": (0.57, 0.55), "S": (0.77, 0.75),
    "T": (0.83, 1.19), "W": (1.08, 1.37), "Y": (0.69, 1.47), "V": (1.06, 1.70)
}

# Kyte-Doolittle hydrophobicity scale
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}

# EEF1 Implicit Solvation Parameters
EEF1_DG_REF = {
    "A": -0.67, "R": -10.3, "N": -5.3, "D": -7.3, "C": -1.2, "Q": -5.3, "E": -7.3,
    "G": 0.0, "H": -5.3, "I": 2.4, "L": 2.2, "K": -9.3, "M": -1.0, "F": 1.2,
    "P": -0.3, "S": -4.3, "T": -3.5, "W": -2.4, "Y": -5.3, "V": 1.9
}
EEF1_VOLUME = {
    "A": 31.5, "R": 105.1, "N": 52.3, "D": 47.9, "C": 48.3, "Q": 67.5, "E": 64.9,
    "G": 0.0, "H": 77.0, "I": 92.5, "L": 92.5, "K": 103.5, "M": 92.6, "F": 109.8,
    "P": 59.8, "S": 39.4, "T": 57.0, "W": 139.7, "Y": 118.4, "V": 74.0
}

# Sidechain radii from CB (Audited for physical extension)
SIDECHAIN_RADII = {
    "A": 0.5, "G": 0.0, "V": 1.2, "L": 2.3, "I": 2.0, "M": 3.8, "F": 4.5,
    "Y": 5.9, "W": 5.0, "S": 1.0, "T": 1.2, "C": 1.8, "P": 1.5, "N": 2.3,
    "Q": 3.8, "D": 2.3, "E": 3.8, "H": 3.5, "K": 6.0, "R": 6.9
}


class NRCPotential:
    """
    Unified Biophysical Potential Manager.
    Calculates PyTorch-based energy potentials.
    """

    def __init__(self, sequence: str, weights: dict, contacts: list = None, charges: np.ndarray = None, guide_coords: np.ndarray = None, k_guide: float = 0.0):
        self.sequence = sequence
        self.N_res = len(sequence)
        self.weights = weights
        self.contacts = contacts
        self.guide_coords = guide_coords
        self.k_guide = k_guide
        
        # Precomputations
        self.h_pos = np.array([max(0.0, KYTE_DOOLITTLE.get(aa, 0.0)) for aa in sequence])
        self.p_alpha = np.array([CHOU_FASMAN.get(aa, (1.0, 1.0))[0] for aa in sequence])
        self.p_beta = np.array([CHOU_FASMAN.get(aa, (1.0, 1.0))[1] for aa in sequence])
        self.dg_ref = np.array([EEF1_DG_REF.get(aa, 0.0) for aa in sequence])
        self.vol = np.array([EEF1_VOLUME.get(aa, 0.0) for aa in sequence])
        self.has_cb = np.array([aa != 'G' for aa in sequence], dtype=bool)
        self.r_sc = np.array([SIDECHAIN_RADII.get(aa, 1.5) for aa in sequence])
        self.charges = charges if charges is not None else np.zeros(self.N_res)
        self.RG_TARGET = 2.2 * (self.N_res ** 0.38)  # Empirical Flory exponent for globular folding
        self.MODULAR_SCALE = 3.8017

        # Convert properties to tensors
        self.h_pos_t = torch.tensor(self.h_pos, dtype=torch.float64)
        self.charges_t = torch.tensor(self.charges, dtype=torch.float64)
        self.p_alpha_t = torch.tensor(self.p_alpha, dtype=torch.float64)
        self.p_beta_t = torch.tensor(self.p_beta, dtype=torch.float64)
        self.vol_t = torch.tensor(self.vol, dtype=torch.float64)
        self.dg_ref_t = torch.tensor(self.dg_ref, dtype=torch.float64)
        self.r_sc_t = torch.tensor(self.r_sc, dtype=torch.float64)

        # Precompute non-bonded masks and property products
        # For very large proteins (>2000 residues), skip non-bonded interactions to avoid OOM
        if self.N_res > 2000:
            self.has_nb = False
        else:
            mask = np.triu(np.ones((self.N_res, self.N_res), dtype=bool), k=3)
            if self.guide_coords is not None:
                close_mask = np.zeros((self.N_res, self.N_res), dtype=bool)
                for i in range(self.N_res):
                    for j in range(i + 3, self.N_res):
                        if j - i <= 6:
                            close_mask[i, j] = True
                        elif i < len(self.guide_coords) and j < len(self.guide_coords):
                            d = np.linalg.norm(self.guide_coords[i] - self.guide_coords[j])
                            if d < 12.0:
                                close_mask[i, j] = True
                        else:
                            close_mask[i, j] = True
                mask = mask & close_mask

            if np.any(mask):
                idx_i, idx_j = np.where(mask)
                self.idx_i_t = torch.tensor(idx_i, dtype=torch.long)
                self.idx_j_t = torch.tensor(idx_j, dtype=torch.long)
                self.h_prod_t = self.h_pos_t[self.idx_i_t] * self.h_pos_t[self.idx_j_t]
                self.q_prod_t = self.charges_t[self.idx_i_t] * self.charges_t[self.idx_j_t]
                self.has_nb = True
            else:
                self.has_nb = False

        # Precompute C-beta masks and property selections
        mask_cb = np.triu(np.ones((self.N_res, self.N_res), dtype=bool), k=2)
        if np.any(mask_cb):
            idx_i_cb, idx_j_cb = np.where(mask_cb)
            valid_cb_pairs = self.has_cb[idx_i_cb] & self.has_cb[idx_j_cb]
            if np.any(valid_cb_pairs):
                self.idx_i_cb_f_t = torch.tensor(idx_i_cb[valid_cb_pairs], dtype=torch.long)
                self.idx_j_cb_f_t = torch.tensor(idx_j_cb[valid_cb_pairs], dtype=torch.long)
                self.vol_i_t = self.vol_t[self.idx_i_cb_f_t]
                self.vol_j_t = self.vol_t[self.idx_j_cb_f_t]
                self.dg_ref_i_t = self.dg_ref_t[self.idx_i_cb_f_t]
                self.dg_ref_j_t = self.dg_ref_t[self.idx_j_cb_f_t]
                self.sc_thresholds_t = self.r_sc_t[self.idx_i_cb_f_t] + self.r_sc_t[self.idx_j_cb_f_t] + 1.2
                self.has_cb_pairs = True
            else:
                self.has_cb_pairs = False
        else:
            self.has_cb_pairs = False

        # Backbone vectors
        self.r_N_rel = torch.tensor([-1.46, 0.0, 0.0], dtype=torch.float64)
        self.r_C_rel = torch.tensor([1.52, 0.0, 0.0], dtype=torch.float64)
        self.r_O_rel = torch.tensor([2.15, 1.0, 0.0], dtype=torch.float64)
        self.r_CB_rel = torch.tensor([-0.53, -1.22, -0.75], dtype=torch.float64)

        # Build all-atom lists for differentiable clash check
        from .atoms import NRCAtoms
        self.atom_lib = NRCAtoms()
        
        all_rel_coords = []
        all_res_idx = []
        all_atom_names = []
        
        for i, aa in enumerate(sequence):
            res_dict = self.atom_lib.get_full_residue(aa, np.zeros(3), np.eye(3))
            for atom_name, rel_coord in res_dict.items():
                all_rel_coords.append(rel_coord)
                all_res_idx.append(i)
                all_atom_names.append(atom_name)
                
        self.M_atoms = len(all_rel_coords)
        self.rel_coords_t = torch.tensor(np.array(all_rel_coords), dtype=torch.float64)
        self.res_idx_t = torch.tensor(all_res_idx, dtype=torch.long)
        
        # Build non-bonded atom pairs with spatial and sequence filtering
        idx_a = []
        idx_b = []
        
        close_residue_pairs = np.zeros((self.N_res, self.N_res), dtype=bool)
        for i in range(self.N_res):
            for j in range(i + 1, self.N_res):
                if j - i <= 4:
                    close_residue_pairs[i, j] = True
                elif self.guide_coords is not None:
                    if i < len(self.guide_coords) and j < len(self.guide_coords):
                        guide_dist = np.linalg.norm(self.guide_coords[i] - self.guide_coords[j])
                        cutoff = 12.0 if self.k_guide > 0.01 else 22.0
                        if guide_dist < cutoff:
                            close_residue_pairs[i, j] = True
                    else:
                        close_residue_pairs[i, j] = True
                else:
                    close_residue_pairs[i, j] = True


        atoms_by_res = [[] for _ in range(self.N_res)]
        for a, r in enumerate(all_res_idx):
            atoms_by_res[r].append(a)

        r_a_indices, r_b_indices = np.where(close_residue_pairs)
        for r_a, r_b in zip(r_a_indices, r_b_indices):
            for a in atoms_by_res[r_a]:
                name_a = all_atom_names[a]
                for b in atoms_by_res[r_b]:
                    name_b = all_atom_names[b]
                    # Exclude peptide bond (C_i and N_i+1)
                    if r_b == r_a + 1 and name_a == "C" and name_b == "N":
                        continue
                    if r_a == r_b + 1 and name_a == "N" and name_b == "C":
                        continue
                    idx_a.append(a)
                    idx_b.append(b)
                
        self.atom_idx_a_t = torch.tensor(idx_a, dtype=torch.long)
        self.atom_idx_b_t = torch.tensor(idx_b, dtype=torch.long)

    def compute_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute the total potential energy of the given coordinates."""
        coords = coords.view(-1, 3)
        N = coords.shape[0]
        total_e = torch.tensor(0.0, dtype=torch.float64)

        # 1. Harmonic Backbone Constraints (i to i+1)
        diff_bond = coords[1:] - coords[:-1]
        d_bond = torch.norm(diff_bond, dim=1)
        bond_e = self.weights["bond"] * torch.sum((d_bond - 3.8)**2)
        total_e = total_e + bond_e

        # Non-bonded terms mask (j - i >= 3)
        if self.has_nb:
            diff_nb = coords[self.idx_i_t] - coords[self.idx_j_t]
            d_nb = torch.norm(diff_nb, dim=1) + 1e-9

            # 2. Steric repulsion (clash check)
            clash_mask = d_nb < 4.0
            if torch.any(clash_mask):
                steric_e = self.weights["steric"] * torch.sum((4.0 - d_nb[clash_mask])**2)
                total_e = total_e + steric_e

            # 3. TTT-7 Resonance & Tesla 3-6-9 exclusion
            dr = d_nb * self.MODULAR_SCALE
            damping = 1.0 / (1.0 + 0.1 * d_nb**2)
            void_penalty = self.weights["steric"] * torch.sum(damping * (1.0 + torch.cos(2 * np.pi * dr / 3.0)))
            total_e = total_e + void_penalty

            ttt_factor = 2 * np.pi / 9.0
            ttt_e = -self.weights["ttt7"] * torch.sum(torch.cos(ttt_factor * (dr - 7.0)))
            total_e = total_e + ttt_e

            # 4. Hydrophobic collapse (Long-range rational + Gaussian well)
            # Rational well centered at 7.0 A: optimal CA-CA burial distance in globular proteins
            rational_term = 1.0 / (1.0 + ((d_nb - 7.0) ** 2) / 25.0)
            # Gaussian well centered at 6.5 A for tight hydrophobic core compaction
            gaussian_term = torch.exp(-((d_nb - 6.5) ** 2) / 8.0)
            hydro_e = self.weights["hydro"] * torch.sum(-self.h_prod_t * (rational_term + 1.5 * gaussian_term))
            total_e = total_e + hydro_e

            # 5. Debye-Huckel screened electrostatics (physiological 150mM salt)
            kappa = 0.1
            elec_e = self.weights["elec"] * torch.sum(self.q_prod_t * torch.exp(-kappa * d_nb) / (d_nb + 1e-9))
            total_e = total_e + elec_e

        # 6. Local pseudo-torsions (distances i to i+2 and i to i+3)
        if N > 2:
            idx_i2 = np.arange(N - 2)
            idx_k2 = idx_i2 + 2
            diff_2 = coords[idx_i2] - coords[idx_k2]
            d_2 = torch.norm(diff_2, dim=1)

            p_alpha_local2 = torch.tensor((self.p_alpha[idx_i2] + self.p_alpha[idx_i2+1] + self.p_alpha[idx_k2]) / 3.0, dtype=torch.float64)
            p_beta_local2 = torch.tensor((self.p_beta[idx_i2] + self.p_beta[idx_i2+1] + self.p_beta[idx_k2]) / 3.0, dtype=torch.float64)

            torsion_2_e = self.weights["torsion"] * torch.sum(p_alpha_local2 * (d_2 - 5.4)**2 + p_beta_local2 * (d_2 - 6.6)**2)
            total_e = total_e + torsion_2_e

        if N > 3:
            idx_i3 = np.arange(N - 3)
            idx_k3 = idx_i3 + 3
            diff_3 = coords[idx_i3] - coords[idx_k3]
            d_3 = torch.norm(diff_3, dim=1)

            p_alpha_local3 = torch.tensor((self.p_alpha[idx_i3] + self.p_alpha[idx_i3+1] + self.p_alpha[idx_i3+2] + self.p_alpha[idx_k3]) / 4.0, dtype=torch.float64)
            p_beta_local3 = torch.tensor((self.p_beta[idx_i3] + self.p_beta[idx_i3+1] + self.p_beta[idx_i3+2] + self.p_beta[idx_k3]) / 4.0, dtype=torch.float64)

            torsion_3_e = self.weights["torsion"] * torch.sum(p_alpha_local3 * (d_3 - 5.1)**2 + p_beta_local3 * (d_3 - 9.8)**2)
            total_e = total_e + torsion_3_e

            # 6.5. Ramachandran-style pseudo-dihedral potential
            c0 = coords[:-3]
            c1 = coords[1:-2]
            c2 = coords[2:-1]
            c3 = coords[3:]

            b0 = -1.0 * (c1 - c0)
            b1 = c2 - c1
            b2 = c3 - c2

            b1_norm = b1 / (torch.norm(b1, dim=1, keepdim=True) + 1e-9)

            v = b0 - torch.sum(b0 * b1_norm, dim=1, keepdim=True) * b1_norm
            w = b2 - torch.sum(b2 * b1_norm, dim=1, keepdim=True) * b1_norm

            x = torch.sum(v * w, dim=1)
            y = torch.sum(torch.cross(b1_norm, v, dim=1) * w, dim=1)

            dihedrals = torch.atan2(y, x)

            # Target dihedrals based on Chou-Fasman helical vs sheet propensities
            # Helix: ~-50 degrees = -0.8727 rad
            # Sheet: ~+120 degrees = 2.0944 rad
            p_alpha_local = torch.tensor((self.p_alpha[:-3] + self.p_alpha[1:-2] + self.p_alpha[2:-1] + self.p_alpha[3:]) / 4.0, dtype=torch.float64, device=coords.device)
            p_beta_local = torch.tensor((self.p_beta[:-3] + self.p_beta[1:-2] + self.p_beta[2:-1] + self.p_beta[3:]) / 4.0, dtype=torch.float64, device=coords.device)

            target_rad = torch.where(p_alpha_local > p_beta_local, -0.8727, 2.0944)
            w_ramachandran = self.weights.get("torsion", 25.0) * torch.max(p_alpha_local, p_beta_local)

            ramachandran_e = torch.sum(w_ramachandran * (1.0 - torch.cos(dihedrals - target_rad)))
            total_e = total_e + ramachandran_e

        # Reconstruct coordinate frames
        rot = reconstruct_backbone_frames_t(coords)

        # 7. Directional Hydrogen Bonding Potential
        r_N = coords + torch.matmul(rot, self.r_N_rel)
        r_C = coords + torch.matmul(rot, self.r_C_rel)
        r_O = coords + torch.matmul(rot, self.r_O_rel)

        # Amide hydrogen H
        if N > 1:
            u_NC = r_C[:-1] - r_N[1:]
            u_NC = u_NC / (torch.norm(u_NC, dim=1, keepdim=True) + 1e-9)
            u_NCA = coords[1:] - r_N[1:]
            u_NCA = u_NCA / (torch.norm(u_NCA, dim=1, keepdim=True) + 1e-9)
            v_H = u_NC + u_NCA
            r_H_mid = r_N[1:] + 1.0 * v_H / (torch.norm(v_H, dim=1, keepdim=True) + 1e-9)
            r_H = torch.cat([r_H_mid[0:1], r_H_mid], dim=0)
        else:
            r_H = coords.clone()

        # Evaluate H-bonds using precomputed filtered non-bonded pairs
        if self.has_nb:
            mask1 = self.idx_i_t >= 1
            d_i_flat = torch.cat([self.idx_i_t[mask1], self.idx_j_t], dim=0)
            a_j_flat = torch.cat([self.idx_j_t[mask1], self.idx_i_t], dim=0)
            
            N_coords = r_N[d_i_flat]
            H_coords = r_H[d_i_flat]
            C_coords = r_C[a_j_flat]
            O_coords = r_O[a_j_flat]

            u_NH = H_coords - N_coords
            u_NH = u_NH / (torch.norm(u_NH, dim=1, keepdim=True) + 1e-9)

            u_CO = O_coords - C_coords
            u_CO = u_CO / (torch.norm(u_CO, dim=1, keepdim=True) + 1e-9)

            v_OH = H_coords - O_coords
            r_OH = torch.norm(v_OH, dim=1) + 1e-9
            e_OH = v_OH / r_OH.unsqueeze(1)

            cos_D = torch.sum(u_NH * e_OH, dim=1)
            cos_A = torch.sum(u_CO * (-e_OH), dim=1)

            cos_D_clamp = torch.clamp(cos_D, min=0.0)
            cos_A_clamp = torch.clamp(cos_A, min=0.0)

            V_attr = -5.0 * torch.exp(-0.5 * (r_OH - 1.8)**2)
            raw_hb_e = V_attr * (cos_D_clamp**2) * (cos_A_clamp**2)

            is_helix_pair = torch.abs(d_i_flat - a_j_flat) == 4
            p_alpha_t = torch.tensor(self.p_alpha, dtype=torch.float64)
            p_beta_t = torch.tensor(self.p_beta, dtype=torch.float64)

            w_helix_pair = self.weights["helix"] * p_alpha_t[d_i_flat] * p_alpha_t[a_j_flat]
            w_sheet_pair = self.weights["sheet"] * p_beta_t[d_i_flat] * p_beta_t[a_j_flat]
            hb_weights = torch.where(is_helix_pair, w_helix_pair, w_sheet_pair)

            hb_e = torch.sum(hb_weights * raw_hb_e)
            total_e = total_e + hb_e

        # 8. Radius of Gyration Confinement
        mean_coords = torch.mean(coords, dim=0)
        rel_coords = coords - mean_coords
        rg = torch.sqrt(torch.mean(torch.sum(rel_coords**2, dim=1)) + 1e-9)
        conf_e = self.weights["rg"] * (rg - self.RG_TARGET)**2
        total_e = total_e + conf_e

        # 9. Co-evolution / Contact Map Restraints
        if self.contacts is not None and len(self.contacts) > 0:
            contact_indices_i = torch.tensor([c[0] for c in self.contacts], dtype=torch.long)
            contact_indices_j = torch.tensor([c[1] for c in self.contacts], dtype=torch.long)
            contact_targets = torch.tensor([c[2] if len(c) > 2 else 6.0 for c in self.contacts], dtype=torch.float64)
            contact_weights = torch.tensor([c[3] if len(c) > 3 else 1.0 for c in self.contacts], dtype=torch.float64)

            diff_c = coords[contact_indices_i] - coords[contact_indices_j]
            d_c = torch.norm(diff_c, dim=1) + 1e-9

            contact_e = torch.sum(contact_weights * (d_c - contact_targets)**2)
            total_e = total_e + self.weights["contact"] * contact_e

        # 10. Explicit Side-Chain Centroids (CB) & Solvation (EEF1)
        if N > 2 and self.has_cb_pairs:
            r_CB = coords + torch.matmul(rot, self.r_CB_rel)
            diff_cb = r_CB[self.idx_i_cb_f_t] - r_CB[self.idx_j_cb_f_t]
            d_cb = torch.norm(diff_cb, dim=1) + 1e-9
            
            # Centroid steric repulsion using residue-specific sidechain radii + 1.2A clearance
            clash_cb = d_cb < self.sc_thresholds_t
            if torch.any(clash_cb):
                cb_steric_e = self.weights["centroid_steric"] * torch.sum((self.sc_thresholds_t[clash_cb] - d_cb[clash_cb])**2)
                total_e = total_e + cb_steric_e
                
            # EEF1 Solvation
            desolv_ij = self.vol_j_t * torch.exp(-(d_cb**2) / (2.0 * 3.5**2))
            desolv_ji = self.vol_i_t * torch.exp(-(d_cb**2) / (2.0 * 3.5**2))
            
            # Fix sign error: desolvation is attractive for hydrophobic residues (dg_ref > 0)
            solv_e = -self.weights["solvation"] * torch.sum(self.dg_ref_i_t * desolv_ij + self.dg_ref_j_t * desolv_ji)
            total_e = total_e + solv_e

        # 10.5. All-Atom Pairwise Steric Clash potential (Hard Wall)
        rel_coords_dev = self.rel_coords_t.to(coords.device)
        res_idx_dev = self.res_idx_t.to(coords.device)
        atom_idx_a_dev = self.atom_idx_a_t.to(coords.device)
        atom_idx_b_dev = self.atom_idx_b_t.to(coords.device)

        rot_selected = rot[res_idx_dev]
        rel_rotated = torch.bmm(rot_selected, rel_coords_dev.unsqueeze(2)).squeeze(2)
        coords_all = coords[res_idx_dev] + rel_rotated

        diff_atoms = coords_all[atom_idx_a_dev] - coords_all[atom_idx_b_dev]
        d_atoms = torch.norm(diff_atoms, dim=1) + 1e-9

        clash_atoms_mask = d_atoms < 1.30
        if torch.any(clash_atoms_mask):
            all_clash_e = self.weights["steric"] * 10000.0 * torch.sum((1.30 - d_atoms[clash_atoms_mask])**2)
            total_e = total_e + all_clash_e

        # 11. C-alpha Harmonic Guide Constraint
        if self.guide_coords is not None and self.k_guide > 0.0:
            guide_coords_t = torch.tensor(self.guide_coords, dtype=torch.float64)
            m_len = min(len(guide_coords_t), N)
            guide_e = self.k_guide * torch.sum((coords[:m_len] - guide_coords_t[:m_len])**2)
            total_e = total_e + guide_e

        return total_e
