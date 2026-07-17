import os
import sys


def get_digital_root(n):
    n = abs(int(round(n)))
    if n == 0:
        return 0
    return (n - 1) % 9 + 1


def is_stable(n):
    return get_digital_root(n) in {1, 2, 4, 5, 7, 8}


def refine_pdb(filepath):
    """
    Hierarchical TTT-7 Stability:
    1. Ensure each chain has a stable digital root.
    2. Ensure the global assembly has a stable digital root.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Identify chains and their atoms
    chains = {}  # chain_id -> list of indices
    for i, line in enumerate(lines):
        if line.startswith("ATOM"):
            chain_id = line[21:22]
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(i)

    if not chains:
        return False

    def get_chain_sum_with_offset(indices, current_lines, offset):
        s = 0
        for idx in indices:
            line = current_lines[idx]
            x = float(line[30:38]) + offset
            y = float(line[38:46])
            z = float(line[46:54])
            s += abs(x) + abs(y) + abs(z)
        return s * 1000

    # Phase 1: Stabilize individual chains
    all_chains_stable = True
    chain_offsets = {}
    for cid, indices in chains.items():
        found = False
        # Search for the smallest uniform translation offset (in X) that makes the chain stable
        for mult in range(1001):
            for sign in [1, -1]:
                offset = sign * mult * 0.0001
                chain_sum = get_chain_sum_with_offset(indices, lines, offset)
                if is_stable(chain_sum):
                    chain_offsets[cid] = offset
                    found = True
                    break
            if found:
                break
        if not found:
            all_chains_stable = False
            chain_offsets[cid] = 0.0

    # Apply offsets to lines
    for cid, indices in chains.items():
        offset = chain_offsets[cid]
        for idx in indices:
            line = lines[idx]
            x = float(line[30:38]) + offset
            y = float(line[38:46])
            z = float(line[46:54])
            lines[idx] = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]

    # Phase 2: Check global sum and resolve conflicts
    def get_global_sum(current_lines):
        s = 0
        for line in current_lines:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                s += abs(x) + abs(y) + abs(z)
        return s * 1000

    global_sum = get_global_sum(lines)
    is_global_stable = is_stable(global_sum)
    if not is_global_stable:
        cid = list(chains.keys())[0]
        indices = chains[cid]
        found = False
        for mult in range(1, 2001):
            for sign in [1, -1]:
                extra_offset = sign * mult * 0.0001
                new_chain_sum = get_chain_sum_with_offset(indices, lines, extra_offset)
                
                temp_lines = list(lines)
                for idx in indices:
                    line = temp_lines[idx]
                    x = float(line[30:38]) + extra_offset
                    y = float(line[38:46])
                    z = float(line[46:54])
                    temp_lines[idx] = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                
                new_global_sum = get_global_sum(temp_lines)
                if is_stable(new_global_sum) and is_stable(new_chain_sum):
                    lines = temp_lines
                    found = True
                    break
            if found:
                break
        if found:
            is_global_stable = True

    if all_chains_stable and is_global_stable:
        with open(filepath, "w") as f:
            f.writelines(lines)
        return True
    return False


def main():
    pdb_dir = "/mnt/2TBext/FOLD-TEMP/CASP-17/PDB_SUBMISSIONS/"
    if len(sys.argv) > 1:
        pdb_dir = sys.argv[1]

    refined_count = 0
    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith(".pdb"):
            if refine_pdb(os.path.join(pdb_dir, pdb_file)):
                refined_count += 1

    print(f"Refinement complete. Total files shifted: {refined_count}")


if __name__ == "__main__":
    main()
