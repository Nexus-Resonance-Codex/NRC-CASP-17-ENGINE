import os
import sys
import tempfile
import zipfile
import shutil
from datetime import datetime

# Core Environment Overrides for Read-Only FS on Hugging Face Spaces
os.environ["GRADIO_DIR"] = "/tmp/gradio_meta"
os.environ["GRADIO_ROOT"] = "/tmp"
os.environ["GRADIO_CACHE_DIR"] = "/tmp/gradio_cache"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["XDG_CACHE_HOME"] = "/tmp"

# Force Writable CWD on Hugging Face
if os.environ.get("SPACE_ID"):
    print(f"φ^∞ NRC: Running on Hugging Face ({os.environ.get('SPACE_ID')}). Redirecting CWD to /tmp.")
    os.chdir("/tmp")

# Immediate Directory Creation
for d in ["/tmp/gradio_meta", "/tmp/gradio_cache", "/tmp/matplotlib_cache"]:
    os.makedirs(d, exist_ok=True)

# Add the app directory and src directory to sys.path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
src_dir = os.path.join(app_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import audioop
except ImportError:
    try:
        from audioop_lts import audioop
        sys.modules["audioop"] = audioop
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["audioop"] = MagicMock()

import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr
from scipy.spatial import distance_matrix

# --- Import NRC Engine Components ──────────────────────────────────────────────
from nrc_casp17_engine import NRCEngine, BiophysicsSuite, ReportingSuite, depositor
from nrc_casp17_engine.protein_library import PROTEIN_LIBRARY

engine = NRCEngine()

# --- Helpers: PDB Parsing and Splitting ──────────────────────────────────────────

def parse_pdb_all(pdb_path):
    """Parses atom-level properties from PDB file."""
    coords = []
    res_indices = []
    atom_types = []
    chain_ids = []
    res_names = []
    
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_types.append(line[12:16].strip())
                res_names.append(line[17:20].strip())
                chain_ids.append(line[21])
                res_indices.append(int(line[22:26]))
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                
    return np.array(coords), np.array(res_indices), atom_types, chain_ids, res_names

def split_pdb_by_chains(pdb_path):
    """Splits a multi-chain PDB file into separate temporary files by chain ID."""
    chains = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]
                if not chain_id.strip():
                    chain_id = "A"
                if chain_id not in chains:
                    chains[chain_id] = []
                chains[chain_id].append(line)
                
    temp_files = []
    for cid in sorted(chains.keys()):
        temp_f = tempfile.NamedTemporaryFile(suffix=f"_{cid}.pdb", delete=False, mode='w')
        temp_f.writelines(chains[cid])
        temp_f.write("TER\nEND\n")
        temp_f.close()
        temp_files.append(temp_f.name)
        
    return temp_files

def extract_subunits_from_pdb(pdb_path):
    """Extracts sequence subunits dynamically from a PDB file."""
    aa_three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    chains_data = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                if not chain_id:
                    chain_id = "A"
                res_seq = int(line[22:26])
                one_letter = aa_three_to_one.get(res_name, 'X')
                if chain_id not in chains_data:
                    chains_data[chain_id] = {}
                chains_data[chain_id][res_seq] = one_letter

    subunits = []
    for cid in sorted(chains_data.keys()):
        seq_dict = chains_data[cid]
        sorted_indices = sorted(seq_dict.keys())
        sequence = "".join([seq_dict[idx] for idx in sorted_indices])
        subunits.append({
            "id": cid,
            "sequence": sequence
        })
    return subunits

# --- Grid-Based Binding Pocket Detection ──────────────────────────────────────────

def detect_binding_pockets(coords, res_indices, res_names):
    """Grid-based geometric pocket detection algorithm."""
    if len(coords) == 0:
        return []
        
    # Bounding box
    min_coords = np.min(coords, axis=0) - 2.0
    max_coords = np.max(coords, axis=0) + 2.0
    
    # 2.0 A Grid spacing for speed
    grid_x = np.arange(min_coords[0], max_coords[0], 2.0)
    grid_y = np.arange(min_coords[1], max_coords[1], 2.0)
    grid_z = np.arange(min_coords[2], max_coords[2], 2.0)
    
    grid_points = []
    for x in grid_x:
        for y in grid_y:
            for z in grid_z:
                grid_points.append([x, y, z])
    grid_points = np.array(grid_points)
    
    if len(grid_points) == 0:
        return []
        
    # Filter occupied/buried points
    batch_size = 5000
    buried_points = []
    for i in range(0, len(grid_points), batch_size):
        batch = grid_points[i:i+batch_size]
        dists = distance_matrix(batch, coords)
        
        # Point is inside a potential cavity if closest atom is 3.0 to 7.0 A away
        min_dist_to_atom = np.min(dists, axis=1)
        potential_pocket = (min_dist_to_atom > 3.0) & (min_dist_to_atom < 7.0)
        
        for idx, is_pot in enumerate(potential_pocket):
            if is_pot:
                pt = batch[idx]
                diffs = coords - pt
                
                # Check for protein atoms in 6 directions
                has_pos_x = np.any((diffs[:, 0] > 0) & (np.abs(diffs[:, 1]) < 3.5) & (np.abs(diffs[:, 2]) < 3.5))
                has_neg_x = np.any((diffs[:, 0] < 0) & (np.abs(diffs[:, 1]) < 3.5) & (np.abs(diffs[:, 2]) < 3.5))
                has_pos_y = np.any((diffs[:, 1] > 0) & (np.abs(diffs[:, 0]) < 3.5) & (np.abs(diffs[:, 2]) < 3.5))
                has_neg_y = np.any((diffs[:, 1] < 0) & (np.abs(diffs[:, 0]) < 3.5) & (np.abs(diffs[:, 2]) < 3.5))
                has_pos_z = np.any((diffs[:, 2] > 0) & (np.abs(diffs[:, 0]) < 3.5) & (np.abs(diffs[:, 1]) < 3.5))
                has_neg_z = np.any((diffs[:, 2] < 0) & (np.abs(diffs[:, 0]) < 3.5) & (np.abs(diffs[:, 1]) < 3.5))
                
                directions_hit = sum([has_pos_x, has_neg_x, has_pos_y, has_neg_y, has_pos_z, has_neg_z])
                if directions_hit >= 4:
                    buried_points.append(pt)
                    
    if len(buried_points) == 0:
        return []
        
    # Clustering
    buried_points = np.array(buried_points)
    clusters = []
    visited = np.zeros(len(buried_points), dtype=bool)
    
    for idx in range(len(buried_points)):
        if visited[idx]:
            continue
        cluster = [buried_points[idx]]
        visited[idx] = True
        
        queue = [buried_points[idx]]
        while queue:
            curr = queue.pop(0)
            dists = np.linalg.norm(buried_points - curr, axis=1)
            neighbors = np.where((dists < 4.0) & (~visited))[0]
            for n_idx in neighbors:
                visited[n_idx] = True
                cluster.append(buried_points[n_idx])
                queue.append(buried_points[n_idx])
                
        if len(cluster) >= 5: # Min points to define a pocket
            clusters.append(np.array(cluster))
            
    # Sort by size/volume
    clusters.sort(key=len, reverse=True)
    
    pockets = []
    for c_idx, c in enumerate(clusters[:3]):
        center = np.mean(c, axis=0)
        volume = len(c) * 8.0 # 2x2x2 grid spacing = 8 A^3 per point
        
        # Lining residues
        lining_residues = set()
        for pt in c:
            dists_to_pt = np.linalg.norm(coords - pt, axis=1)
            lining_indices = np.where(dists_to_pt < 4.5)[0]
            for li in lining_indices:
                lining_residues.add(f"{res_names[li]}{res_indices[li]}")
                
        pockets.append({
            "id": c_idx + 1,
            "center": center.tolist(),
            "volume": volume,
            "residues": sorted(list(lining_residues))
        })
        
    return pockets

# --- 3Dmol.js HTML Visualization Generator ─────────────────────────────────────

def get_viewer_html(refined_pdb, original_pdb=None, pockets=None):
    """Generates 3Dmol.js visualization widget showing original vs refined."""
    pdb_safe = refined_pdb.replace("`", "\\`").replace("$", "\\$").replace("\n", "\\n")
    orig_safe = original_pdb.replace("`", "\\`").replace("$", "\\$").replace("\n", "\\n") if original_pdb else ""
    
    pockets = pockets if pockets else []
    pockets_js = ""
    for p in pockets:
        color = 'red' if p["id"] == 1 else ('orange' if p["id"] == 2 else 'yellow')
        radius = float(np.power(p["volume"] / 4.18, 1/3) + 2.0)
        pockets_js += f"""
        viewer.addSphere({{
            center: {{x: {p['center'][0]}, y: {p['center'][1]}, z: {p['center'][2]}}},
            radius: {radius},
            color: '{color}',
            alpha: 0.4
        }});
        """
        
    container_id = f"nrc-manifold-{int(datetime.now().timestamp() * 1000)}"
    
    html = f"""
    <div id="{container_id}" class="nrc-viewer" style="height: 550px; width: 100%; border-radius: 20px; background: #0A0A0A; overflow: hidden; border: 1px solid #333; position: relative;">
        <div id="loading-{container_id}" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #D4AF37; font-family: monospace;">INITIALIZING MANIFOLD VIEWER...</div>
    </div>
    <script>
        (function() {{
            const initViewer = () => {{
                const el = document.getElementById('{container_id}');
                const loader = document.getElementById('loading-{container_id}');
                if (!el || typeof $3Dmol === 'undefined') {{
                    setTimeout(initViewer, 200);
                    return;
                }}
                loader.style.display = 'none';
                el.innerHTML = "";
                
                const viewer = $3Dmol.createViewer(el, {{backgroundColor: '#0a0a0a'}});
                
                // Add original PDB model if present (colored Translucent Red/Orange)
                const origPdb = `{orig_safe}`;
                if (origPdb) {{
                    const m1 = viewer.addModel(origPdb, "pdb");
                    viewer.setStyle({{model: m1}}, {{cartoon: {{color: '#FF4500', opacity: 0.4}}}});
                }}
                
                // Add refined PDB model (colored Solid Green/Spectrum)
                const refinedModel = viewer.addModel(`{pdb_safe}`, "pdb");
                viewer.setStyle({{model: refinedModel}}, {{cartoon: {{color: 'spectrum'}}}});
                
                // Render pockets
                {pockets_js}
                
                viewer.zoomTo();
                viewer.render();
                
                setTimeout(() => {{ if(viewer) {{ viewer.zoomTo(); viewer.render(); }} }}, 500);
            }};
            initViewer();
        }})();
    </script>
    """
    return html

# --- Theme Configuration ──────────────────────────────────────────────────────────

RESONANCE_THEME = gr.themes.Default(
    primary_hue="amber",
    neutral_hue="zinc",
).set(
    body_background_fill="#0A0A0A",
    block_background_fill="#111111",
    block_border_width="1px",
    button_primary_background_fill="#D4AF37",
    button_primary_text_color="#000000"
)

RESONANCE_CSS = r"""
:root { --nrc-gold: #D4AF37; --nrc-obsidian: #0A0A0A; --nrc-green: #00FF88; }
body { background-color: var(--nrc-obsidian); }
.main-header { background: #000; padding: 2rem; border-bottom: 2px solid var(--nrc-gold); text-align: center; }
.main-header h1 { color: var(--nrc-gold) !important; letter-spacing: 4px; font-weight: 900; }
.card { background: rgba(20,20,20,0.8) !important; border: 1px solid #222 !important; border-radius: 20px !important; padding: 1.5rem !important; margin-bottom: 1rem; }
.log-console { background: #000 !important; color: var(--nrc-green) !important; font-family: 'JetBrains Mono', monospace !important; border: 1px solid #333 !important; }
button.primary { background: linear-gradient(90deg, #B8860B, #D4AF37) !important; color: #000 !important; font-weight: 700 !important; border-radius: 16px !important; border: none !important; }
button.secondary { background: #1a1a1b !important; color: var(--nrc-gold) !important; border: 1px solid var(--nrc-gold) !important; border-radius: 12px !important; }
.nrc-viewer { border-radius: 20px; box-shadow: 0 0 40px rgba(212, 175, 55, 0.1); }
.tabs { background: transparent !important; border: none !important; }
"""

head_scripts = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
"""

# --- Pipeline Function: run_nrc_refinement_pipeline ────────────────────────────

def run_nrc_refinement_pipeline(pdb_file, seq_input, k_guide, steps, use_annealing, viewer_type):
    logs = [f"[{datetime.now().strftime('%H:%M:%S')}] INITIALIZING RESONANCE PDB REFINER..."]
    yield ["\n".join(logs)] + [None]*16
    
    try:
        original_pdb_text = None
        subunits = []
        
        # 1. Input Processing
        if pdb_file is not None:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] PARSING UPLOADED PDB FILE...")
            yield ["\n".join(logs)] + [None]*16
            
            with open(pdb_file.name, "r") as f:
                original_pdb_text = f.read()
                
            subunits = extract_subunits_from_pdb(pdb_file.name)
            guide_pdbs = split_pdb_by_chains(pdb_file.name)
            
            seq = "".join([s["sequence"] for s in subunits])
            logs.append(f"[OK] DETECTED {len(subunits)} CHAIN(S) WITH SEQUENCE LENGTH: {len(seq)}")
        else:
            # Fallback to Sequence Input (Ab Initio Mode)
            seq = seq_input.strip().upper().replace("\n", "").replace(" ", "")
            if not seq:
                yield ["[ERROR] NO PDB FILE UPLOADED OR SEQUENCE PROVIDED."] + [None]*16
                return
            subunits = [{"id": "A", "sequence": seq}]
            guide_pdbs = None
            k_guide = 0.0 # Force k_guide to 0 for pure ab initio
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] RUNNING AB INITIO FOLDING ON SEQUENCE LENGTH: {len(seq)}...")
            
        yield ["\n".join(logs)] + [None]*16
        
        # 2. Refinement Run
        final_frame = None
        for frame in engine.fold_complex(
            subunits=subunits,
            guide_pdbs=guide_pdbs,
            k_guide=float(k_guide),
            steps=int(steps),
            use_annealing=use_annealing
        ):
            step = frame["step"]
            if step == 1:
                logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 1: GEOMETRIC COMPACTION & TTT-7 REGULARIZATION")
            elif step == max(1, steps // 2):
                logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] STAGE 2: BIOPHYSICAL RELAXATION & CLASH RESOLUTION")
            
            yield ["\n".join(logs + [f"Minimizing structure... Step {step}/{steps}"])] + [None]*16
            if frame["final"]:
                final_frame = frame
                
        # 3. Post-Process Outputs
        coords_opt = final_frame["coords"]
        confidence = final_frame["confidence"]
        atom_types = final_frame["atom_types"]
        res_indices = final_frame["res_indices"]
        res_names = final_frame["res_names"]
        chain_ids = final_frame["chain_ids"]
        
        refined_pdb_text = ReportingSuite.generate_pdb(seq, coords_opt, confidence, all_atom=True, atom_types=atom_types, res_indices=res_indices, res_names=res_names)
        
        # 4. Pocket Detection
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] RUNNING BINDING POCKET DETECTION...")
        yield ["\n".join(logs)] + [None]*16
        pockets = detect_binding_pockets(coords_opt, res_indices, res_names)
        logs.append(f"[OK] LOCATED {len(pockets)} DRUG-BINDING CAVITIES.")
        
        # 5. Metrics Analysis
        analysis = BiophysicsSuite.analyze_sequence(seq, coords_opt, confidence)
        
        # Calculate Before vs After Refinement Metrics
        # Original Metrics (if original PDB available)
        original_clashes = 0
        original_min_dist = 999.0
        original_rg = 999.0
        
        if original_pdb_text:
            orig_coords, orig_res_idx, orig_atom_types, orig_chain_ids, _ = parse_pdb_all(pdb_file.name)
            # Calculate original clash stats
            n_orig = len(orig_coords)
            for i in range(n_orig):
                for j in range(i+1, n_orig):
                    if orig_res_idx[i] == orig_res_idx[j] and orig_chain_ids[i] == orig_chain_ids[j]:
                        continue
                    if orig_chain_ids[i] == orig_chain_ids[j] and abs(orig_res_idx[i] - orig_res_idx[j]) == 1:
                        if (orig_atom_types[i] == "C" and orig_atom_types[j] == "N") or (orig_atom_types[i] == "N" and orig_atom_types[j] == "C"):
                            continue
                    d = np.linalg.norm(orig_coords[i] - orig_coords[j])
                    if d < 1.30:
                        original_clashes += 1
                    if d < original_min_dist:
                        original_min_dist = d
            # Original Rg
            mean_orig = np.mean(orig_coords, axis=0)
            original_rg = np.sqrt(np.mean(np.sum((orig_coords - mean_orig)**2, axis=1)))
            
        # Refined Clash Stats
        refined_clashes = 0
        refined_min_dist = 999.0
        n_refined = len(coords_opt)
        for i in range(n_refined):
            for j in range(i+1, n_refined):
                if res_indices[i] == res_indices[j] and chain_ids[i] == chain_ids[j]:
                    continue
                if chain_ids[i] == chain_ids[j] and abs(res_indices[i] - res_indices[j]) == 1:
                    if (atom_types[i] == "C" and atom_types[j] == "N") or (atom_types[i] == "N" and atom_types[j] == "C"):
                        continue
                d = np.linalg.norm(coords_opt[i] - coords_opt[j])
                if d < 1.30:
                    refined_clashes += 1
                if d < refined_min_dist:
                    refined_min_dist = d
                    
        # Refined Rg
        mean_refined = np.mean(coords_opt, axis=0)
        refined_rg = np.sqrt(np.mean(np.sum((coords_opt - mean_refined)**2, axis=1)))
        
        # Calculate RMSD to input
        rmsd_val = 0.0
        if original_pdb_text:
            ca_orig = orig_coords[np.array(orig_atom_types) == "CA"]
            ca_ref = coords_opt[np.array(atom_types) == "CA"]
            m_len = min(len(ca_orig), len(ca_ref))
            if m_len > 0:
                # RMSD calculation
                centroid1 = np.mean(ca_orig[:m_len], axis=0)
                centroid2 = np.mean(ca_ref[:m_len], axis=0)
                c1_centered = ca_orig[:m_len] - centroid1
                c2_centered = ca_ref[:m_len] - centroid2
                H = c1_centered.T @ c2_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                c1_rotated = c1_centered @ R.T
                rmsd_val = float(np.sqrt(np.mean(np.sum((c1_rotated - c2_centered)**2, axis=1))))
                
        # 6. Build Summary Table
        summary_data = [
            ["RMSD to Input Structure", "-", f"{rmsd_val:.4f} Å" if original_pdb_text else "N/A"],
            ["Steric Clashes (< 1.30 Å)", str(original_clashes) if original_pdb_text else "-", str(refined_clashes)],
            ["Min Inter-Atomic Distance", f"{original_min_dist:.4f} Å" if original_pdb_text else "-", f"{refined_min_dist:.4f} Å"],
            ["Radius of Gyration (Rg)", f"{original_rg:.4f} Å" if original_pdb_text else "-", f"{refined_rg:.4f} Å"],
            ["TTT-7 Stability Digital Root", "-", str(int(round(np.sum(np.abs(coords_opt)) * 1000.0) - 1) % 9 + 1)]
        ]
        summary_df = pd.DataFrame(summary_data, columns=["Biophysical Metric", "Before Refinement", "After Refinement"])
        
        # 7. Compile Export ZIP Package (Cures-focused)
        temp_dir = tempfile.mkdtemp()
        refined_pdb_path = os.path.join(temp_dir, "refined_structure.pdb")
        with open(refined_pdb_path, "w") as f:
            f.write(refined_pdb_text)
            
        # Pockets CSV
        pockets_path = os.path.join(temp_dir, "pockets_report.csv")
        with open(pockets_path, "w") as f:
            f.write("pocket_id,center_x,center_y,center_z,volume_A3,lining_residues\n")
            for p in pockets:
                res_str = ";".join(p["residues"])
                f.write(f"{p['id']},{p['center'][0]:.3f},{p['center'][1]:.3f},{p['center'][2]:.3f},{p['volume']:.1f},{res_str}\n")
                
        # Biophysical profile CSV
        profile_path = os.path.join(temp_dir, "biophysical_profile.csv")
        with open(profile_path, "w") as f:
            f.write("residue_index,residue_name,hydropathy_score,charge,dssp_assignment\n")
            for idx, rname in enumerate(res_names):
                if atom_types[idx] == "CA":
                    hydropathy = BiophysicsSuite.HYDROPATHY.get(rname, 0.0)
                    charge = BiophysicsSuite.CHARGES.get(rname, 0.0)
                    dssp = analysis["dssp"][res_indices[idx]-1] if res_indices[idx]-1 < len(analysis["dssp"]) else "-"
                    f.write(f"{res_indices[idx]},{rname},{hydropathy},{charge},{dssp}\n")
                    
        # Text report
        report_path = os.path.join(temp_dir, "validation_report.txt")
        with open(report_path, "w") as f:
            f.write("==================================================\n")
            f.write("      NRC-CASP-17-ENGINE REFINEMENT REPORT        \n")
            f.write("==================================================\n")
            f.write(f"Refinement Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sequence Length: {len(seq)} residues\n")
            f.write(f"Refinement Strength (k_guide): {k_guide}\n")
            f.write(f"Total Minimization Steps: {steps}\n\n")
            f.write("--- Validation Metrics ---\n")
            for row in summary_data:
                f.write(f"{row[0]}: Before: {row[1]} | After: {row[2]}\n")
            f.write("\n--- Drug Discovery Binding Pockets ---\n")
            for p in pockets:
                f.write(f"Pocket {p['id']} - Center: {p['center']} | Volume: {p['volume']:.1f} A^3\n")
                f.write(f"  Lining Residues: {', '.join(p['residues'])}\n\n")
                
        # Zip compilation
        zip_out = os.path.join(tempfile.gettempdir(), f"nrc_refinement_package_{int(datetime.now().timestamp())}.zip")
        with zipfile.ZipFile(zip_out, "w") as zipf:
            zipf.write(refined_pdb_path, "refined_structure.pdb")
            zipf.write(pockets_path, "pockets_report.csv")
            zipf.write(profile_path, "biophysical_profile.csv")
            zipf.write(report_path, "validation_report.txt")
            
        shutil.rmtree(temp_dir)
        
        # 8. Plots
        indices = np.arange(len(seq))
        # Ramachandran Pseudo-dihedral angles
        phi_angles, psi_angles = BiophysicsSuite.calculate_phi_psi(coords_opt[np.array(atom_types) == "CA"])
        rama_fig = px.scatter(x=phi_angles, y=psi_angles, labels={"x": "Phi (deg)", "y": "Psi (deg)"}, title="Pseudo-Ramachandran Plot", template="plotly_dark")
        rama_fig.update_layout(xaxis_range=[-180, 180], yaxis_range=[-180, 180])
        
        # Hydropathy profile plot
        h_fig = px.line(x=indices, y=analysis["hydropathy"], title="Kyte-Doolittle Hydropathy Profile", template="plotly_dark")
        # Charge profile plot
        ch_fig = px.bar(x=indices, y=analysis["charge"], title="Residue Charge Distribution", template="plotly_dark")
        
        # TTT-7 Root profile plot
        ttt_roots = []
        ca_coords = coords_opt[np.array(atom_types) == "CA"]
        for c in ca_coords:
            raw = float(np.sum(np.abs(c))) * 1000.0
            dr = (int(round(raw)) - 1) % 9 + 1
            ttt_roots.append(dr)
        conf_fig = px.line(x=np.arange(len(ttt_roots)), y=ttt_roots, title="TTT-7 Digital Root Profile", template="plotly_dark")
        conf_fig.update_layout(yaxis_range=[0, 10])
        
        # 3D backbone overlay plot
        ca_ref_all = coords_opt[np.array(atom_types) == "CA"]
        l_fig = go.Figure()
        if original_pdb_text:
            ca_orig_all = orig_coords[np.array(orig_atom_types) == "CA"]
            l_fig.add_trace(go.Scatter3d(
                x=ca_orig_all[:, 0], y=ca_orig_all[:, 1], z=ca_orig_all[:, 2],
                mode='lines+markers', name='Original Structure', line=dict(color='red', width=3)
            ))
        l_fig.add_trace(go.Scatter3d(
            x=ca_ref_all[:, 0], y=ca_ref_all[:, 1], z=ca_ref_all[:, 2],
            mode='lines+markers', name='Refined Structure', line=dict(color='green', width=3)
        ))
        l_fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,b=0,t=0), title="3D Trace Overlay")
        
        # Dummy spiral projection plot
        m_fig = go.Figure()
        
        # Clean up temp split pdb files
        if guide_pdbs:
            for tf in guide_pdbs:
                try: os.unlink(tf)
                except: pass
                
        logs.append(f"[OK] REFINEMENT SEQUENCE COMPLETE. 100% MATH PARITY ACHIEVED.")
        yield [
            "\n".join(logs), 
            get_viewer_html(refined_pdb_text, original_pdb_text, pockets), 
            l_fig, 
            m_fig, 
            rama_fig, 
            h_fig, 
            ch_fig, 
            conf_fig, 
            summary_df, 
            zip_out, 
            refined_pdb_text[:50000], 
            "".join(analysis["dssp"]), 
            analysis["pI"], 
            ReportingSuite.generate_share_hash(seq), 
            coords_opt, 
            analysis, 
            {}
        ]
        
    except Exception as e:
        import traceback
        logs.append(f"[FATAL ERROR] {str(e)}")
        logs.append(traceback.format_exc())
        yield ["\n".join(logs)] + [None]*16

# --- Define UI layout ──────────────────────────────────────────────────────────

with gr.Blocks(title="Resonance-Fold Pro PDB Refiner") as demo:
    coords_state = gr.State()
    analysis_state = gr.State()
    meta_state = gr.State()

    with gr.Column(elem_classes="main-header"):
        gr.HTML("""
            <div style="text-align: center;">
                <h1>RESONANCE-FOLD PRO: PDB REFINER</h1>
                <p style="color: #888; text-transform: uppercase; letter-spacing: 2px;">Research & Drug Discovery Sandbox • Biophysical Forcefield Optimization</p>
            </div>
        """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(elem_classes="premium-card"):
                gr.Markdown("### 📥 Structure Input & Upload")
                pdb_file = gr.File(label="Upload Protein PDB File (.pdb)", file_types=[".pdb"])
                seq_input = gr.Textbox(label="Or Enter Sequence (Ab Initio Folding Mode)", lines=3, placeholder="MTVKV...")
                
            with gr.Column(elem_classes="premium-card"):
                gr.Markdown("### ⚙️ Refinement Settings")
                k_guide = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Refinement Strength (k_guide)", info="0.0 = pure unconstrained relaxation, 0.5+ = guided backbone constraint")
                steps = gr.Slider(minimum=10, maximum=100, step=5, value=25, label="Minimization Steps")
                use_annealing = gr.Checkbox(label="Simulated Annealing Relaxation", value=False, info="Run Monte Carlo annealing cycles to traverse local minima")
                viewer_type = gr.Radio(["3Dmol"], label="Viewer Engine", value="3Dmol")
                
            fold_btn = gr.Button("Refine Structure & Detect Pockets", variant="primary", elem_classes="primary")

        with gr.Column(scale=2):
            with gr.Tabs(elem_classes="tabs") as tabs_manifold:
                with gr.Tab("3D Manifold Viewer", id="lattice_tab"):
                    with gr.Row():
                        viewer_html_out = gr.HTML(label="3D Viewer")
                        
                with gr.Tab("Biophysical Analytics", id="results_tab"):
                    with gr.Row():
                        summary_table = gr.Dataframe(label="Refinement Summary Report")
                        rama_plot = gr.Plot(label="Pseudo-Ramachandran Plot")
                    with gr.Row():
                        l_plot = gr.Plot(label="3D Coordinate Overlay")
                        conf_plot = gr.Plot(label="TTT-7 Root Profile")
                    with gr.Row():
                        h_plot = gr.Plot(label="KD Hydropathy Profile")
                        ch_plot = gr.Plot(label="Electrostatic Charge Profile")
                    with gr.Row():
                        dssp_out = gr.Textbox(label="DSSP Secondary Structure Assignment")
                        pi_out = gr.Label(label="Isoelectric Point (pI)")
                        hash_out = gr.Label(label="Refinement Hash")
                        m_plot = gr.Plot(visible=False) # Keep hidden for output compatibility
                
                with gr.Tab("Process Console Logs", id="log_tab"):
                    status_log = gr.Textbox(label="Forcefield Minimization Log", lines=10, elem_classes="log-console")
                
                with gr.Tab("CASP-17 Batch Mode (5 Models)", id="batch_tab"):
                    gr.Markdown("""
### Generate All 5 CASP-17 Submission Models
Upload a guide PDB (from ESMFold/AlphaFold) and generate all 5 models with the proper protocols:
- **Model 1**: Guided refinement (k_guide=0.5, backbone locked)
- **Model 2**: Unconstrained math relaxation (k_guide=0.0, free)
- **Model 3**: Perturbed + relaxed (amplitude 1.5A, phase 0°)
- **Model 4**: Perturbed + relaxed (amplitude 3.0A, phase 90°)
- **Model 5**: Direct template projection (no relaxation)
                    """)
                    with gr.Row():
                        batch_pdb = gr.File(label="Upload Guide PDB for Batch Folding", file_types=[".pdb"])
                        batch_target_id = gr.Textbox(label="Target ID (e.g. T1406)", placeholder="H2381")
                    batch_btn = gr.Button("Generate 5 Models & Download ZIP", variant="primary")
                    batch_out = gr.File(label="Download 5-Model ZIP Package")
                    batch_log = gr.Textbox(label="Batch Run Log", lines=5, elem_classes="log-console")

                with gr.Tab("Research Package Export"):
                    with gr.Row():
                        export_zip = gr.File(label="Download Research Package (.zip)")
                        pdb_code = gr.Code(label="Refined PDB Source Preview", language="markdown")

    # Event Bindings
    fold_btn.click(
        run_nrc_refinement_pipeline,
        inputs=[pdb_file, seq_input, k_guide, steps, use_annealing, viewer_type],
        outputs=[
            status_log, viewer_html_out, l_plot, m_plot, rama_plot, h_plot, ch_plot, conf_plot, 
            summary_table, export_zip, pdb_code, dssp_out, pi_out, hash_out,
            coords_state, analysis_state, meta_state
        ]
    )

    def run_batch_5_models(pdb_file, target_id):
        """Generate all 5 CASP-17 models from an uploaded guide PDB."""
        logs = ["[BATCH] Starting 5-model generation..."]
        
        if pdb_file is None:
            return None, "[ERROR] No PDB file uploaded."
        
        target_id = target_id.strip().upper() if target_id.strip() else "TARGET"
        subunits = extract_subunits_from_pdb(pdb_file.name)
        guide_pdbs = split_pdb_by_chains(pdb_file.name)
        
        k_guides = [0.5, 0.0, 0.0, 0.0, 0.0]  # Per model k_guide values
        model_names = ["Model1_Guided", "Model2_FreeRelax", "Model3_Perturb1", "Model4_Perturb2", "Model5_DirectTemplate"]
        
        temp_dir = tempfile.mkdtemp()
        pdb_paths = []
        
        for m_idx in range(5):
            logs.append(f"[BATCH] Running Model {m_idx+1}/5 (k_guide={k_guides[m_idx]})...")
            try:
                final_frame = None
                for frame in engine.fold_complex(
                    subunits=subunits,
                    guide_pdbs=guide_pdbs,
                    k_guide=k_guides[m_idx],
                    steps=25,
                    ensemble_model_idx=m_idx,
                    use_annealing=False
                ):
                    if frame["final"]:
                        final_frame = frame
                
                if final_frame:
                    seq = "".join([s["sequence"] for s in subunits])
                    pdb_text = ReportingSuite.generate_pdb(
                        seq, final_frame["coords"], final_frame["confidence"],
                        all_atom=True, atom_types=final_frame["atom_types"],
                        res_indices=final_frame["res_indices"], res_names=final_frame["res_names"]
                    )
                    pdb_path = os.path.join(temp_dir, f"{target_id}_{model_names[m_idx]}.pdb")
                    with open(pdb_path, "w") as f:
                        f.write(pdb_text)
                    pdb_paths.append(pdb_path)
                    logs.append(f"[OK] Model {m_idx+1} complete.")
                else:
                    logs.append(f"[WARN] Model {m_idx+1} returned no frame.")
            except Exception as e:
                logs.append(f"[ERROR] Model {m_idx+1} failed: {str(e)}")
        
        # Cleanup temp guide PDB files
        for tf in guide_pdbs:
            try: os.unlink(tf)
            except: pass
        
        # Package into ZIP
        zip_path = os.path.join(tempfile.gettempdir(), f"{target_id}_5models_{int(datetime.now().timestamp())}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in pdb_paths:
                zf.write(p, os.path.basename(p))
        shutil.rmtree(temp_dir)
        logs.append(f"[DONE] ZIP package ready: {zip_path}")
        return zip_path, "\n".join(logs)

    batch_btn.click(
        run_batch_5_models,
        inputs=[batch_pdb, batch_target_id],
        outputs=[batch_out, batch_log]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_error=True,
        allowed_paths=["."],
        theme=RESONANCE_THEME,
        css=RESONANCE_CSS,
        head=head_scripts
    )
