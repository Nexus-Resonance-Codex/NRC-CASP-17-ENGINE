import os
import numpy as np
import sys


def parse_pdb(filepath):
    lines = []
    coords = []
    atom_info = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                atom_info.append(line)
            else:
                lines.append(line)
    return np.array(coords), atom_info, lines


def write_pdb(filepath, coords, atom_info, other_lines):
    with open(filepath, "w") as f:
        # This is a bit tricky because other_lines contains REMARKs at the top
        # and TER/END at the bottom.
        # We need to maintain the order.
        # Let's just reconstruct the whole file.

        # We'll assume ATOM lines are contiguous or we can just replace them in place in atom_info
        for i, line in enumerate(atom_info):
            x, y, z = coords[i]
            new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            f.write(new_line)

        # Add non-ATOM lines (this is a bit hacky, better to keep all lines in order)
        # But for our PDBs, ATOM lines are mostly together.
        # Let's try a better approach:
        pass


import torch


def refine_pdb_in_place(filepath, min_dist=1.2, iterations=50, step_size=0.1):
    """
    Highly optimized PyTorch-accelerated steric clash resolution with GPU support and chemistry-aware bond topology.
    """
    print(f"Refining {os.path.basename(filepath)} (optimized PyTorch engine)...")

    with open(filepath, "r") as f:
        all_lines = f.readlines()

    coords = []
    atom_indices = []
    atom_names = []
    res_indices = []
    chain_ids = []

    for i, line in enumerate(all_lines):
        if line.startswith("ATOM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
            atom_indices.append(i)
            atom_names.append(line[12:16].strip())
            chain_ids.append(line[21:22].strip())
            res_indices.append(int(line[22:26]))

    n_atoms = len(coords)
    if n_atoms == 0:
        return

    # Chemistry-aware topology builder (detect bonds and angles from initial coordinates and chemical rules)
    coords_np = np.array(coords)
    bonds = []
    bonded_neighbors = {i: set() for i in range(n_atoms)}
    
    # Find covalent bonds based on chemical rules
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = np.linalg.norm(coords_np[i] - coords_np[j])
            
            is_bond = False
            # Rule 1: Same residue, distance in [0.8, 1.65] A
            if (res_indices[i] == res_indices[j] and 
                chain_ids[i] == chain_ids[j] and 
                0.8 < d < 1.65):
                is_bond = True
            # Rule 2: Peptide bond between consecutive residues
            elif (chain_ids[i] == chain_ids[j] and 
                  abs(res_indices[i] - res_indices[j]) == 1 and 
                  ((atom_names[i] == "C" and atom_names[j] == "N") or 
                   (atom_names[i] == "N" and atom_names[j] == "C")) and 
                  1.0 < d < 1.65):
                is_bond = True
            # Rule 3: Disulfide bond
            elif (atom_names[i] == "SG" and atom_names[j] == "SG" and d < 2.2):
                is_bond = True
                
            if is_bond:
                # Enforce physical bond length (1.45 A) if the starting bond was severely compressed
                target_d = d if d >= 1.30 else 1.45
                bonds.append((i, j, target_d))
                bonded_neighbors[i].add(j)
                bonded_neighbors[j].add(i)

    # Exclude 1-2 (bonded) and 1-3 (angle) pairs from steric repulsion
    excluded_pairs = set()
    for i in range(n_atoms):
        excluded_pairs.add((i, i))
        # 1-2 neighbors
        for j in bonded_neighbors[i]:
            excluded_pairs.add((min(i, j), max(i, j)))
            # 1-3 neighbors
            for k in bonded_neighbors[j]:
                excluded_pairs.add((min(i, k), max(i, k)))

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Load coordinates and construct excluded mask tensor
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    orig_coords = coords_tensor.clone()
    
    excluded_mask = torch.zeros((n_atoms, n_atoms), dtype=torch.bool, device=device)
    for i, j in excluded_pairs:
        excluded_mask[i, j] = True
        excluded_mask[j, i] = True

    # Pre-parse bonds to tensors
    if len(bonds) > 0:
        bonds_i = torch.tensor([b[0] for b in bonds], dtype=torch.long, device=device)
        bonds_j = torch.tensor([b[1] for b in bonds], dtype=torch.long, device=device)
        bonds_td = torch.tensor(
            [b[2] for b in bonds], dtype=torch.float32, device=device
        )
    else:
        bonds_i, bonds_j, bonds_td = None, None, None

    for iteration in range(iterations):
        displacements = torch.zeros_like(coords_tensor)

        # A. Steric Repulsion (Chunked to conserve GPU memory)
        chunk_size = 2000
        for start_idx in range(0, n_atoms, chunk_size):
            end_idx = min(start_idx + chunk_size, n_atoms)
            chunk_coords = coords_tensor[start_idx:end_idx]

            # Pairwise diffs: shape (chunk_len, n_atoms, 3)
            diffs = chunk_coords.unsqueeze(1) - coords_tensor.unsqueeze(0)
            dists = torch.norm(diffs, dim=-1)  # shape (chunk_len, n_atoms)

            # Exclude chemical bonds/angles
            clash_mask = dists < min_dist
            clash_mask &= ~excluded_mask[start_idx:end_idx]

            # Only push one way to avoid double counting (i < j globally)
            r_indices = torch.arange(start_idx, end_idx, device=device).unsqueeze(1)
            c_indices = torch.arange(n_atoms, device=device).unsqueeze(0)
            clash_mask &= r_indices < c_indices

            if torch.any(clash_mask):
                chunk_i, global_j = torch.where(clash_mask)
                global_i = start_idx + chunk_i

                vecs = diffs[chunk_i, global_j]  # shape (M, 3)
                d = dists[chunk_i, global_j].unsqueeze(-1) + 1e-6

                push = (min_dist - d) * step_size
                displacement = (vecs / d) * push

                displacements.index_add_(0, global_i, displacement)
                displacements.index_add_(0, global_j, -displacement)

        coords_tensor += displacements

        # B. Bond Restoration
        if bonds_i is not None:
            bond_displacements = torch.zeros_like(coords_tensor)
            diff = coords_tensor[bonds_i] - coords_tensor[bonds_j]
            curr_d = torch.norm(diff, dim=-1).unsqueeze(-1) + 1e-6
            correction = (bonds_td.unsqueeze(-1) - curr_d) * 0.5
            move = (diff / curr_d) * correction

            bond_displacements.index_add_(0, bonds_i, move)
            bond_displacements.index_add_(0, bonds_j, -move)
            coords_tensor += bond_displacements

        # C. Anchor
        coords_tensor += (orig_coords - coords_tensor) * 0.01

    # Copy back to CPU numpy
    refined_coords = coords_tensor.cpu().numpy()

    # Write back to PDB lines
    for i, idx in enumerate(atom_indices):
        line = all_lines[idx]
        x, y, z = refined_coords[i]
        new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
        all_lines[idx] = new_line

    with open(filepath, "w") as f:
        f.writelines(all_lines)


def main():
    pdb_dir = "/mnt/2TBext/FOLD-TEMP/CASP-17/PDB_SUBMISSIONS/"
    if len(sys.argv) > 1:
        pdb_dir = sys.argv[1]

    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith(".pdb"):
            refine_pdb_in_place(os.path.join(pdb_dir, pdb_file))


if __name__ == "__main__":
    main()
