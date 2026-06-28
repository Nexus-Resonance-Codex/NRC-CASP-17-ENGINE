"""
geometry.py — Coordinate projection and secondary structure initialization
"""

import numpy as np
import torch
from .atoms import NRCAtoms

PHI = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * np.pi / (PHI**2)

def fragment_based_initialization(sequence: str, p_alpha: np.ndarray, p_beta: np.ndarray) -> np.ndarray:
    """
    Assemble sequence fragments using Chou-Fasman propensities to generate
    idealized CA local geometries (alpha helix vs beta strand).
    """
    N = len(sequence)
    coords = np.zeros((N, 3), dtype=np.float64)
    
    # Ideal parameters
    bond_len = 3.8
    
    # Helix CA parameters
    alpha_angle = 90.0 * np.pi / 180.0
    alpha_dihedral = 51.853 * np.pi / 180.0  # Special resonance angle
    
    # Beta CA parameters
    beta_angle = 120.0 * np.pi / 180.0
    beta_dihedral = 170.0 * np.pi / 180.0

    coords[0] = [0.0, 0.0, 0.0]
    if N > 1:
        coords[1] = [bond_len, 0.0, 0.0]
    if N > 2:
        coords[2] = [bond_len + bond_len * np.cos(np.pi - alpha_angle), 
                     bond_len * np.sin(np.pi - alpha_angle), 
                     0.0]
                     
    for i in range(3, N):
        p_a = p_alpha[i]
        p_b = p_beta[i]
        
        if p_a > p_b and p_a > 1.0:
            ang = alpha_angle
            dih = alpha_dihedral
        else:
            ang = beta_angle
            dih = beta_dihedral
            
        v1 = coords[i-1] - coords[i-2]
        v2 = coords[i-2] - coords[i-3]
        
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-9)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-9)
        
        n = np.cross(v2_norm, v1_norm)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-3:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / n_norm
            
        b = np.cross(v1_norm, n)
        
        vec = bond_len * (np.cos(np.pi - ang) * v1_norm + 
                          np.sin(np.pi - ang) * np.cos(dih) * b + 
                          np.sin(np.pi - ang) * np.sin(dih) * n)
                          
        coords[i] = coords[i-1] + vec

    # Center on origin
    coords -= np.mean(coords, axis=0)
    return coords.flatten()


def reconstruct_backbone_frames_t(coords: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct local N x 3 x 3 coordinate frames for H-bonding and side-chains.
    R_i = [x_i, y_i, u_i] where Column 3 is the bond vector, Column 1 is orthogonal.
    """
    N = coords.shape[0]
    if N > 1:
        u_i_raw = coords[1:] - coords[:-1]
        u_i = torch.cat([u_i_raw[0:1], u_i_raw], dim=0)
        u_i = u_i / (torch.norm(u_i, dim=1, keepdim=True) + 1e-9)

        u_prev_init = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64, device=coords.device)
        if N > 2:
            u_prev = torch.cat([u_prev_init, u_i[1:-1]], dim=0)
        else:
            u_prev = u_prev_init[:N]

        x_vec = torch.cross(u_i, u_prev, dim=1)
        x_norm = torch.norm(x_vec, dim=1, keepdim=True)

        fallback_x = torch.where(
            torch.abs(u_i[:, 2:3]) < 0.9,
            torch.stack([u_i[:, 1], -u_i[:, 0], torch.zeros_like(u_i[:, 0])], dim=1),
            torch.stack([torch.zeros_like(u_i[:, 0]), u_i[:, 2], -u_i[:, 1]], dim=1)
        )
        fallback_x = fallback_x / (torch.norm(fallback_x, dim=1, keepdim=True) + 1e-9)
        
        x_norm_safe = torch.where(x_norm < 1e-3, torch.ones_like(x_norm), x_norm)
        x_vec_normalized = x_vec / x_norm_safe
        x_vec = torch.where(x_norm < 1e-3, fallback_x, x_vec_normalized)

        y_vec = torch.cross(u_i, x_vec, dim=1)
        y_vec = y_vec / (torch.norm(y_vec, dim=1, keepdim=True) + 1e-9)

        rot = torch.stack([x_vec, y_vec, u_i], dim=2)
    else:
        rot = torch.eye(3, dtype=torch.float64, device=coords.device).expand(N, 3, 3)
    return rot


def reconstruct_backbone_frames_np(coords: np.ndarray) -> np.ndarray:
    """
    Reconstruct local N x 3 x 3 coordinate frames using NumPy.
    """
    return reconstruct_frenet_frames_np(coords, start_idx=0)

def reconstruct_frenet_frames_np(coords: np.ndarray, start_idx: int = 0) -> np.ndarray:
    """
    Reconstruct covariant local Frenet-Serret coordinate frames for each CA atom.
    """
    N = coords.shape[0]
    rot = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(N):
        idx = start_idx + i
        if i == 0:
            if N > 1:
                u_i = coords[1] - coords[0]
                u_i /= np.linalg.norm(u_i) + 1e-9
            else:
                u_i = np.array([0.0, 0.0, 1.0])
            u_prev = np.array([1.0, 0.0, 0.0])
        else:
            u_i = coords[i] - coords[i - 1]
            u_i /= np.linalg.norm(u_i) + 1e-9
            u_prev = coords[i - 1] - (
                coords[i - 2]
                if i - 2 >= 0
                else coords[i - 1] - np.array([1.0, 0.0, 0.0])
            )
            u_prev /= np.linalg.norm(u_prev) + 1e-9

        x_i = np.cross(u_i, u_prev)
        x_norm = np.linalg.norm(x_i)
        if x_norm < 1e-3:
            x_i = (
                np.array([u_i[1], -u_i[0], 0.0])
                if abs(u_i[2]) < 0.9
                else np.array([0.0, u_i[2], -u_i[1]])
            )
            x_i /= np.linalg.norm(x_i)
        else:
            x_i /= x_norm

        y_i = np.cross(u_i, x_i)
        y_i /= np.linalg.norm(y_i)

        rot[i] = np.column_stack((x_i, y_i, u_i))
    return rot
