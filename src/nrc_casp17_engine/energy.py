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


class NRCPotential:
    """
    Unified Biophysical Potential Manager.
    Calculates PyTorch-based energy potentials.
    """

    def __init__(self, sequence: str, weights: dict, contacts: list = None, charges: np.ndarray = None):
        self.sequence = sequence
        self.N_res = len(sequence)
        self.weights = weights
        self.contacts = contacts
        
        # Precomputations
        self.h_pos = np.array([max(0.0, KYTE_DOOLITTLE.get(aa, 0.0)) for aa in sequence])
        self.p_alpha = np.array([CHOU_FASMAN.get(aa, (1.0, 1.0))[0] for aa in sequence])
        self.p_beta = np.array([CHOU_FASMAN.get(aa, (1.0, 1.0))[1] for aa in sequence])
        self.dg_ref = np.array([EEF1_DG_REF.get(aa, 0.0) for aa in sequence])
        self.vol = np.array([EEF1_VOLUME.get(aa, 0.0) for aa in sequence])
        self.has_cb = np.array([aa != 'G' for aa in sequence], dtype=bool)

        self.charges = charges if charges is not None else np.zeros(self.N_res)
        self.RG_TARGET = 3.0 * (self.N_res**0.33)
        self.MODULAR_SCALE = 3.8017

        # Backbone vectors
        self.r_N_rel = torch.tensor([-1.46, 0.0, 0.0], dtype=torch.float64)
        self.r_C_rel = torch.tensor([1.52, 0.0, 0.0], dtype=torch.float64)
        self.r_O_rel = torch.tensor([2.15, 1.0, 0.0], dtype=torch.float64)
        self.r_CB_rel = torch.tensor([-0.53, -1.22, -0.75], dtype=torch.float64)

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
        mask = np.triu(np.ones((N, N), dtype=bool), k=3)
        if np.any(mask):
            idx_i, idx_j = np.where(mask)
            diff_nb = coords[idx_i] - coords[idx_j]
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

            # 4. Hydrophobic collapse
            h_prod = torch.tensor(self.h_pos[idx_i] * self.h_pos[idx_j], dtype=torch.float64)
            exp_term = torch.exp(-0.5 * (d_nb - 4.5)**2)
            hydro_e = self.weights["hydro"] * torch.sum(-h_prod * exp_term)
            total_e = total_e + hydro_e

            # 5. Coulomb electrostatics
            q_prod = torch.tensor(self.charges[idx_i] * self.charges[idx_j], dtype=torch.float64)
            elec_e = self.weights["elec"] * torch.sum(q_prod / d_nb**2)
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

        # Evaluate H-bonds
        donor_idx = torch.arange(1, N)
        acceptor_idx = torch.arange(0, N)
        d_i, a_j = torch.meshgrid(donor_idx, acceptor_idx, indexing="ij")
        
        pair_mask = torch.abs(d_i - a_j) >= 3
        if torch.any(pair_mask):
            d_i_flat = d_i[pair_mask]
            a_j_flat = a_j[pair_mask]

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
        if N > 2:
            r_CB = coords + torch.matmul(rot, self.r_CB_rel)
            
            mask_cb = np.triu(np.ones((N, N), dtype=bool), k=2)
            if np.any(mask_cb):
                idx_i_cb, idx_j_cb = np.where(mask_cb)
                
                valid_cb_pairs = torch.tensor(self.has_cb[idx_i_cb] & self.has_cb[idx_j_cb], dtype=torch.bool)
                if torch.any(valid_cb_pairs):
                    idx_i_cb_f = idx_i_cb[valid_cb_pairs]
                    idx_j_cb_f = idx_j_cb[valid_cb_pairs]
                    
                    diff_cb = r_CB[idx_i_cb_f] - r_CB[idx_j_cb_f]
                    d_cb = torch.norm(diff_cb, dim=1) + 1e-9
                    
                    # Centroid steric repulsion
                    clash_cb = d_cb < 3.5
                    if torch.any(clash_cb):
                        cb_steric_e = self.weights["centroid_steric"] * torch.sum((3.5 - d_cb[clash_cb])**2)
                        total_e = total_e + cb_steric_e
                        
                    # EEF1 Solvation
                    vol_i = torch.tensor(self.vol[idx_i_cb_f], dtype=torch.float64)
                    vol_j = torch.tensor(self.vol[idx_j_cb_f], dtype=torch.float64)
                    dg_ref_i = torch.tensor(self.dg_ref[idx_i_cb_f], dtype=torch.float64)
                    dg_ref_j = torch.tensor(self.dg_ref[idx_j_cb_f], dtype=torch.float64)
                    
                    desolv_ij = vol_j * torch.exp(-(d_cb**2) / (2.0 * 3.5**2))
                    desolv_ji = vol_i * torch.exp(-(d_cb**2) / (2.0 * 3.5**2))
                    
                    solv_e = self.weights["solvation"] * torch.sum(dg_ref_i * desolv_ij + dg_ref_j * desolv_ji)
                    total_e = total_e + solv_e

        return total_e
