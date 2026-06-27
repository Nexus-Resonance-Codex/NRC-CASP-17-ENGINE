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
    """
    N = coords.shape[0]
    if N > 2:
        v_prev = coords[1:] - coords[:-1]
        d_prev = torch.norm(v_prev, dim=1, keepdim=True) + 1e-9
        u_prev = v_prev / d_prev

        t = u_prev[:-1] + u_prev[1:]
        t = t / (torch.norm(t, dim=1, keepdim=True) + 1e-9)

        n = torch.cross(u_prev[:-1], u_prev[1:], dim=1)
        n = n / (torch.norm(n, dim=1, keepdim=True) + 1e-9)

        b = torch.cross(t, n, dim=1)
        rot_mid = torch.stack([t, n, b], dim=2)
        
        # Pad boundaries
        rot = torch.cat([rot_mid[0:1], rot_mid, rot_mid[-1:]], dim=0)
    else:
        rot = torch.eye(3, dtype=torch.float64).expand(N, 3, 3)
    return rot


def reconstruct_backbone_frames_np(coords: np.ndarray) -> np.ndarray:
    """
    Reconstruct local N x 3 x 3 coordinate frames using NumPy.
    """
    N = coords.shape[0]
    if N > 2:
        v_prev = coords[1:] - coords[:-1]
        d_prev = np.linalg.norm(v_prev, axis=1, keepdims=True) + 1e-9
        u_prev = v_prev / d_prev

        t = u_prev[:-1] + u_prev[1:]
        t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)

        n = np.cross(u_prev[:-1], u_prev[1:], axis=1)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

        b = np.cross(t, n, axis=1)
        rot_mid = np.stack([t, n, b], axis=2)
        
        rot = np.concatenate([rot_mid[0:1], rot_mid, rot_mid[-1:]], axis=0)
    else:
        rot = np.repeat(np.eye(3)[np.newaxis, :, :], N, axis=0)
    return rot

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
