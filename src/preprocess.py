
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdmolops
from tqdm import tqdm

# Silence RDKit spam
RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("warning")

# Dataset paths (edit if needed)
PC9_RAW_DIR = Path("data/raw/Archive/pc9/PC9_data/PC9_data/XYZ")
QM9_RAW_DIR = Path("data/raw/Archive/qm9")
PREPROCESSED_DIR = Path("data/preprocessed")
SUPPORTED_EXTS = {".sdf", ".mol", ".xyz"}


# -------------------------------------------------
# Node Feature Extraction (MolGAN style: 5 features)
# -------------------------------------------------
def atom_features(atom: Chem.Atom) -> np.ndarray:
    """
    These 5 atom features are EXACTLY the ones used in MolGAN & HQ-Cycle-MolGAN.
    """
    # Handle valence carefully - may not be calculated for XYZ-derived molecules
    try:
        # Try to get total valence (implicit + explicit)
        valence = atom.GetTotalValence()
    except (RuntimeError, Exception):
        # Fallback: use degree as approximation for valence
        valence = atom.GetDegree()
    
    return np.array(
        [
            atom.GetAtomicNum(),        # atomic number
            atom.GetDegree(),           # number of neighbors
            valence,                    # total valence (or degree as fallback)
            atom.GetFormalCharge(),     # charge
            int(atom.GetIsAromatic()),  # aromatic flag
        ],
        dtype=np.float32,
    )


# -------------------------------------------------
# Molecule → Graph conversion
# -------------------------------------------------
def mol_to_graph(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (adjacency_matrix, feature_matrix)
    A: NxN binary adjacency
    F: Nx5 node features
    """
    adjacency = rdmolops.GetAdjacencyMatrix(mol)  # NO bond types
    features = np.array([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    return adjacency, features


# -------------------------------------------------
# File scanning helper
# -------------------------------------------------
def _scan_molecule_files(base_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(base_dir):
        for file in files:
            path = Path(root) / file
            if path.suffix.lower() in SUPPORTED_EXTS:
                yield path


# -------------------------------------------------
# XYZ Loader (faithful to MolGAN)
# -------------------------------------------------
def _load_xyz_as_mol(filepath: Path) -> List[Chem.Mol]:
    """
    IMPORTANT:
    - Builds MolGAN-style molecules from XYZ
    - Uses ConnectTheDots (NO chemistry, NO sanitization)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.rstrip() for line in f.readlines() if line.strip()]
    except:
        return []

    if len(lines) < 3:
        return []

    try:
        num_atoms = int(lines[0].strip())
        atom_lines = lines[2:2 + num_atoms]
        if len(atom_lines) < num_atoms:
            return []

        xyz_block = f"{num_atoms}\n\n"
        for line in atom_lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            element = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            xyz_block += f"{element} {x} {y} {z}\n"

        # Build molecule with NO bonds
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return []

        # CRITICAL: MolGAN graph construction
        # ConnectUsing geometry only, no chemistry
        try:
            mol = rdmolops.ConnectTheDots(mol)
        except:
            pass  # Still return whatever we got

        return [mol]

    except:
        return []


# -------------------------------------------------
# Unified loader for sdf, mol, xyz
# -------------------------------------------------
def _load_molecules_from_file(filepath: Path) -> List[Chem.Mol]:
    ext = filepath.suffix.lower()

    if ext == ".sdf":
        supplier = Chem.SDMolSupplier(str(filepath), removeHs=False)
        return [mol for mol in supplier if mol is not None]

    if ext == ".mol":
        mol = Chem.MolFromMolFile(str(filepath), sanitize=False, removeHs=False)
        return [mol] if mol is not None else []

    if ext == ".xyz":
        return _load_xyz_as_mol(filepath)

    return []


# -------------------------------------------------
# Save pickle
# -------------------------------------------------
def _serialize_graphs(graphs: List[Tuple[np.ndarray, np.ndarray]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        pickle.dump(graphs, f)


# -------------------------------------------------
# Main Preprocessing
# -------------------------------------------------
def preprocess_dataset():
    datasets = {
        "PC9": PC9_RAW_DIR,
        "QM9": QM9_RAW_DIR,
    }

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, object]] = {}

    for dataset_name, base_dir in datasets.items():
        if not base_dir.exists():
            print(f"[{dataset_name}] Skipped (missing directory)")
            continue

        molecule_files = list(_scan_molecule_files(base_dir))
        if not molecule_files:
            print(f"[{dataset_name}] No files found.")
            continue

        graphs = []
        sources: set[str] = set()

        for file_path in tqdm(molecule_files, desc=f"Processing {dataset_name}", unit="file"):
            molecules = _load_molecules_from_file(file_path)
            for mol in molecules:
                A, F = mol_to_graph(mol)
                graphs.append((A, F))

            sources.add(str(file_path))

        if not graphs:
            print(f"[{dataset_name}] No valid molecules.")
            continue

        out_dir = PREPROCESSED_DIR / dataset_name
        out_file = out_dir / "graphs.pkl"
        _serialize_graphs(graphs, out_file)

        manifest[dataset_name] = {
            "graph_count": len(graphs),
            "source_files": len(sources),
            "example_sources": sorted(sources)[:5],
            "output_pickle": str(out_file),
        }

        print(f"[{dataset_name}] Processed {len(graphs)} graphs → {out_file}")

    with open(PREPROCESSED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Manifest written.")


if __name__ == "__main__":
    preprocess_dataset()