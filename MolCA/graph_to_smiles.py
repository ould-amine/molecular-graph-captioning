"""
Graph to SMILES Conversion Module

Converts PyTorch Geometric molecular graphs to SMILES strings using RDKit.
Includes validation to ensure conversion accuracy.
"""

import argparse
import os
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch_geometric.data import Data
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdchem
    from rdkit import RDLogger

    # Suppress RDKit warnings for cleaner output
    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")

# =========================================================
# Feature Maps (from data_utils.py)
# =========================================================
x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER',
        'CHI_TETRAHEDRAL', 'CHI_ALLENE', 'CHI_SQUAREPLANAR', 'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'QUINTUPLE', 'HEXTUPLE',
        'ONEANDAHALF', 'TWOANDAHALF', 'THREEANDAHALF', 'FOURANDAHALF', 'FIVEANDAHALF',
        'AROMATIC', 'IONIC', 'HYDROGEN', 'THREECENTER', 'DATIVEONE', 'DATIVE', 'DATIVEL',
        'DATIVER', 'OTHER', 'ZERO',
    ],
    'stereo': [
        'STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

# =========================================================
# RDKit Mapping Dictionaries
# =========================================================

# Chirality mapping
CHIRALITY_MAP = {
    'CHI_UNSPECIFIED': rdchem.ChiralType.CHI_UNSPECIFIED,
    'CHI_TETRAHEDRAL_CW': rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    'CHI_TETRAHEDRAL_CCW': rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    'CHI_OTHER': rdchem.ChiralType.CHI_OTHER,
    'CHI_TETRAHEDRAL': rdchem.ChiralType.CHI_TETRAHEDRAL,
    'CHI_ALLENE': rdchem.ChiralType.CHI_ALLENE,
    'CHI_SQUAREPLANAR': rdchem.ChiralType.CHI_SQUAREPLANAR,
    'CHI_TRIGONALBIPYRAMIDAL': rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    'CHI_OCTAHEDRAL': rdchem.ChiralType.CHI_OCTAHEDRAL,
}

# Hybridization mapping
HYBRIDIZATION_MAP = {
    'UNSPECIFIED': rdchem.HybridizationType.UNSPECIFIED,
    'S': rdchem.HybridizationType.S,
    'SP': rdchem.HybridizationType.SP,
    'SP2': rdchem.HybridizationType.SP2,
    'SP3': rdchem.HybridizationType.SP3,
    'SP3D': rdchem.HybridizationType.SP3D,
    'SP3D2': rdchem.HybridizationType.SP3D2,
    'OTHER': rdchem.HybridizationType.OTHER,
}

# Bond type mapping
BOND_TYPE_MAP = {
    'UNSPECIFIED': rdchem.BondType.UNSPECIFIED,
    'SINGLE': rdchem.BondType.SINGLE,
    'DOUBLE': rdchem.BondType.DOUBLE,
    'TRIPLE': rdchem.BondType.TRIPLE,
    'QUADRUPLE': rdchem.BondType.QUADRUPLE,
    'QUINTUPLE': rdchem.BondType.QUINTUPLE,
    'HEXTUPLE': rdchem.BondType.HEXTUPLE,
    'ONEANDAHALF': rdchem.BondType.ONEANDAHALF,
    'TWOANDAHALF': rdchem.BondType.TWOANDAHALF,
    'THREEANDAHALF': rdchem.BondType.THREEANDAHALF,
    'FOURANDAHALF': rdchem.BondType.FOURANDAHALF,
    'FIVEANDAHALF': rdchem.BondType.FIVEANDAHALF,
    'AROMATIC': rdchem.BondType.AROMATIC,
    'IONIC': rdchem.BondType.IONIC,
    'HYDROGEN': rdchem.BondType.HYDROGEN,
    'THREECENTER': rdchem.BondType.THREECENTER,
    'DATIVEONE': rdchem.BondType.DATIVEONE,
    'DATIVE': rdchem.BondType.DATIVE,
    'DATIVEL': rdchem.BondType.DATIVE,
    'DATIVER': rdchem.BondType.DATIVE,
    'OTHER': rdchem.BondType.OTHER,
    'ZERO': rdchem.BondType.ZERO,
}

# Bond stereo mapping
BOND_STEREO_MAP = {
    'STEREONONE': rdchem.BondStereo.STEREONONE,
    'STEREOANY': rdchem.BondStereo.STEREOANY,
    'STEREOZ': rdchem.BondStereo.STEREOZ,
    'STEREOE': rdchem.BondStereo.STEREOE,
    'STEREOCIS': rdchem.BondStereo.STEREOCIS,
    'STEREOTRANS': rdchem.BondStereo.STEREOTRANS,
}


# =========================================================
# Conversion Result Dataclass
# =========================================================
@dataclass
class ConversionResult:
    """Result of a graph-to-SMILES conversion."""
    graph_id: str
    desc: str
    smiles: Optional[str]
    success: bool
    error_message: Optional[str] = None
    validation_passed: bool = False
    validation_details: Optional[Dict] = None


# =========================================================
# Core Conversion Functions
# =========================================================

def decode_node_features(x: torch.Tensor, node_idx: int) -> Dict[str, Any]:
    """
    Decodes node features from tensor indices to actual values.
    
    Args:
        x: Node feature tensor of shape [num_nodes, 9]
        node_idx: Index of the node to decode
        
    Returns:
        Dictionary with decoded node properties
    """
    features = x[node_idx].tolist()

    return {
        'atomic_num': x_map['atomic_num'][int(features[0])],
        'chirality': x_map['chirality'][int(features[1])],
        'degree': x_map['degree'][int(features[2])],
        'formal_charge': x_map['formal_charge'][int(features[3])],
        'num_hs': x_map['num_hs'][int(features[4])],
        'num_radical_electrons': x_map['num_radical_electrons'][int(features[5])],
        'hybridization': x_map['hybridization'][int(features[6])],
        'is_aromatic': x_map['is_aromatic'][int(features[7])],
        'is_in_ring': x_map['is_in_ring'][int(features[8])],
    }


def decode_edge_features(edge_attr: torch.Tensor, edge_idx: int) -> Dict[str, Any]:
    """
    Decodes edge features from tensor indices to actual values.
    
    Args:
        edge_attr: Edge feature tensor of shape [num_edges, 3]
        edge_idx: Index of the edge to decode
        
    Returns:
        Dictionary with decoded edge properties
    """
    features = edge_attr[edge_idx].tolist()

    return {
        'bond_type': e_map['bond_type'][int(features[0])],
        'stereo': e_map['stereo'][int(features[1])],
        'is_conjugated': e_map['is_conjugated'][int(features[2])],
    }


def graph_to_mol(graph: Data, add_explicit_hs: bool = False) -> Optional[Chem.RWMol]:
    """
    Converts a PyTorch Geometric graph to an RDKit molecule.
    
    Args:
        graph: PyTorch Geometric Data object with x, edge_index, edge_attr
        add_explicit_hs: Whether to add explicit hydrogens based on num_hs feature
        
    Returns:
        RDKit RWMol object or None if conversion fails
    """
    try:
        mol = Chem.RWMol()

        # Step 1: We add atoms with their properties
        num_nodes = graph.x.size(0)
        atom_map = {}  # Map from graph node idx to RDKit atom idx

        for i in range(num_nodes):
            node_features = decode_node_features(graph.x, i)

            # Create atom with atomic number
            atom = Chem.Atom(node_features['atomic_num'])

            # Set formal charge
            atom.SetFormalCharge(node_features['formal_charge'])

            # Set chirality
            chirality_str = node_features['chirality']
            if chirality_str in CHIRALITY_MAP:
                atom.SetChiralTag(CHIRALITY_MAP[chirality_str])

            # Set radical electrons
            atom.SetNumRadicalElectrons(node_features['num_radical_electrons'])

            # Set hybridization
            hybridization_str = node_features['hybridization']
            if hybridization_str in HYBRIDIZATION_MAP:
                atom.SetHybridization(HYBRIDIZATION_MAP[hybridization_str])

            # Set aromaticity flag
            atom.SetIsAromatic(node_features['is_aromatic'])

            # Set explicit number of Hs (RDKit will use this during sanitization)
            atom.SetNumExplicitHs(node_features['num_hs'])

            # Add atom to molecule
            rdkit_idx = mol.AddAtom(atom)
            atom_map[i] = rdkit_idx

        # Step 2: We add bonds (only unique bonds, i.e., i < j)
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        num_edges = edge_index.size(1)

        added_bonds = set()

        for e in range(num_edges):
            src = int(edge_index[0, e])
            dst = int(edge_index[1, e])

            # We only add each bond once (skipping if already added or if src >= dst)
            bond_key = (min(src, dst), max(src, dst))
            if bond_key in added_bonds:
                continue
            added_bonds.add(bond_key)

            edge_features = decode_edge_features(edge_attr, e)

            # Get bond type
            bond_type_str = edge_features['bond_type']
            bond_type = BOND_TYPE_MAP.get(bond_type_str, rdchem.BondType.SINGLE)

            # Add bond
            bond_idx = mol.AddBond(atom_map[src], atom_map[dst], bond_type)

            # Get the bond we just added and set its properties
            bond = mol.GetBondBetweenAtoms(atom_map[src], atom_map[dst])

            if bond is not None:
                # Set stereo
                stereo_str = edge_features['stereo']
                if stereo_str in BOND_STEREO_MAP:
                    bond.SetStereo(BOND_STEREO_MAP[stereo_str])

                # Set conjugation flag
                bond.SetIsConjugated(edge_features['is_conjugated'])

        return mol

    except Exception as e:
        warnings.warn(f"Error building molecule: {e}")
        return None


def sanitize_and_get_smiles(
        mol: Chem.RWMol,
        canonical: bool = True,
        isomeric: bool = True,
        kekulize: bool = False
) -> Optional[str]:
    """
    Sanitizes a molecule and converts to SMILES string.
    
    Args:
        mol: RDKit RWMol object
        canonical: Whether to produce canonical SMILES
        isomeric: Whether to include stereochemistry in SMILES
        kekulize: Whether to kekulize before generating SMILES
        
    Returns:
        SMILES string or None if sanitization fails
    """
    try:
        # Convert to regular Mol for sanitization
        mol = mol.GetMol()

        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # Try partial sanitization
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                                  Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except:
                return None

        # Assign stereochemistry
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except:
            pass  # Continue even if stereo assignment fails

        # Generate SMILES
        smiles = Chem.MolToSmiles(
            mol,
            canonical=canonical,
            isomericSmiles=isomeric,
            kekuleSmiles=kekulize
        )

        return smiles

    except Exception as e:
        return None


def graph_to_smiles(
        graph: Data,
        canonical: bool = True,
        isomeric: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """
    Converts a PyTorch Geometric graph to SMILES string.
    
    Args:
        graph: PyTorch Geometric Data object
        canonical: Whether to produce canonical SMILES
        isomeric: Whether to include stereochemistry
        
    Returns:
        Tuple of (smiles_string, error_message)
    """
    mol = graph_to_mol(graph)

    if mol is None:
        return None, "Failed to build molecule from graph"

    smiles = sanitize_and_get_smiles(mol, canonical=canonical, isomeric=isomeric)

    if smiles is None:
        return None, "Failed to sanitize molecule or generate SMILES"

    return smiles, None


# =========================================================
# Validation Functions
# =========================================================

def validate_smiles_roundtrip(
        original_graph: Data,
        smiles: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates SMILES by converting back to molecule and comparing properties.
    
    Args:
        original_graph: Original PyTorch Geometric graph
        smiles: Generated SMILES string
        
    Returns:
        Tuple of (validation_passed, details_dict)
    """
    details = {
        'smiles_valid': False,
        'atom_count_match': False,
        'bond_count_match': False,
        'atom_types_match': False,
        'aromatic_atoms_match': False,
        'formal_charge_match': False,
    }

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, details

    details['smiles_valid'] = True

    # Get original properties
    orig_num_atoms = original_graph.x.size(0)
    orig_num_bonds = len(set(
        (min(int(original_graph.edge_index[0, e]), int(original_graph.edge_index[1, e])),
         max(int(original_graph.edge_index[0, e]), int(original_graph.edge_index[1, e])))
        for e in range(original_graph.edge_index.size(1))
    ))

    # Get reconstructed properties
    recon_num_atoms = mol.GetNumAtoms()
    recon_num_bonds = mol.GetNumBonds()

    # Check atom count (allow for implicit H differences)
    details['atom_count_match'] = (orig_num_atoms == recon_num_atoms)

    # Check bond count
    details['bond_count_match'] = (orig_num_bonds == recon_num_bonds)

    # Check atom types (element counts)
    orig_elements = defaultdict(int)
    for i in range(orig_num_atoms):
        atomic_num = x_map['atomic_num'][int(original_graph.x[i, 0])]
        orig_elements[atomic_num] += 1

    recon_elements = defaultdict(int)
    for atom in mol.GetAtoms():
        recon_elements[atom.GetAtomicNum()] += 1

    details['atom_types_match'] = (dict(orig_elements) == dict(recon_elements))

    # Check aromatic atom count
    orig_aromatic = sum(1 for i in range(orig_num_atoms)
                        if x_map['is_aromatic'][int(original_graph.x[i, 7])])
    recon_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    details['aromatic_atoms_match'] = (orig_aromatic == recon_aromatic)

    # Check total formal charge
    orig_charge = sum(x_map['formal_charge'][int(original_graph.x[i, 3])]
                      for i in range(orig_num_atoms))
    recon_charge = Chem.GetFormalCharge(mol)
    details['formal_charge_match'] = (orig_charge == recon_charge)

    # Overall validation
    critical_checks = [
        details['smiles_valid'],
        details['atom_count_match'],
        details['atom_types_match'],
    ]

    return all(critical_checks), details


# =========================================================
# Batch Processing Functions
# =========================================================

def convert_dataset(
        graphs: List[Data],
        canonical: bool = True,
        isomeric: bool = True,
        validate: bool = True,
        verbose: bool = True
) -> List[ConversionResult]:
    """
    Converts a list of graphs to SMILES strings.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        canonical: Whether to produce canonical SMILES
        isomeric: Whether to include stereochemistry
        validate: Whether to validate conversions
        verbose: Whether to show progress bar
        
    Returns:
        List of ConversionResult objects
    """
    results = []

    iterator = tqdm(graphs, desc="Converting graphs") if verbose else graphs

    for graph in iterator:
        graph_id = getattr(graph, 'id', 'unknown')
        desc = getattr(graph, 'description', None)

        smiles, error = graph_to_smiles(graph, canonical=canonical, isomeric=isomeric)

        if smiles is None:
            results.append(ConversionResult(
                graph_id=graph_id,
                desc=desc,
                smiles=None,
                success=False,
                error_message=error,
                validation_passed=False
            ))
            continue

        if validate:
            valid, details = validate_smiles_roundtrip(graph, smiles)
            results.append(ConversionResult(
                graph_id=graph_id,
                desc=desc,
                smiles=smiles,
                success=True,
                validation_passed=valid,
                validation_details=details
            ))
        else:
            results.append(ConversionResult(
                graph_id=graph_id,
                desc=desc,
                smiles=smiles,
                success=True,
                validation_passed=True
            ))

    return results


def print_conversion_summary(results: List[ConversionResult], split_name: str = ""):
    """Prints summary statistics for conversion results."""
    total = len(results)
    successful = sum(1 for r in results if r.success)
    validated = sum(1 for r in results if r.validation_passed)

    print(f"\n{'=' * 60}")
    print(f"Conversion Summary {f'({split_name})' if split_name else ''}")
    print(f"{'=' * 60}")
    print(f"Total graphs:        {total}")
    print(f"Successful:          {successful} ({100 * successful / total:.1f}%)")
    print(f"Validation passed:   {validated} ({100 * validated / total:.1f}%)")
    print(f"Failed:              {total - successful}")

    if any(r.validation_details for r in results if r.validation_details):
        print(f"\nValidation Details:")
        checks = ['smiles_valid', 'atom_count_match', 'bond_count_match',
                  'atom_types_match', 'aromatic_atoms_match', 'formal_charge_match']
        for check in checks:
            passed = sum(1 for r in results
                         if r.validation_details and r.validation_details.get(check, False))
            print(f"  {check:25s}: {passed}/{successful} ({100 * passed / successful:.1f}%)")

    failures = [r for r in results if not r.success][:5]
    if failures:
        print(f"\nSample failures:")
        for f in failures:
            print(f"  ID: {f.graph_id}, Error: {f.error_message}")


def save_smiles_mapping(
        results: List[ConversionResult],
        output_path: str,
        include_failed: bool = False
):
    """
    Saves SMILES mapping to file.
    
    Args:
        results: List of ConversionResult objects
        output_path: Path to output pickle file
        include_failed: Whether to include failed conversions (with None SMILES)
    """
    mapping = {}
    for r in results:
        if r.success or include_failed:
            mapping[r.graph_id] = {
                'description': r.desc,
                'smiles': r.smiles,
                'success': r.success,
                'validation_passed': r.validation_passed,
            }

    with open(output_path, 'wb') as f:
        pickle.dump(mapping, f)

    print(f"Saved {len(mapping)} SMILES mappings to {output_path}")

    base, _ = os.path.splitext(output_path)
    human_readable_path = base + ".tsv"

    with open(human_readable_path, "w", encoding="utf-8") as f:
        if "test" in base:
            f.write("graph_id\tsmiles\tsuccess\tvalidation_passed\n")
        else:
            f.write("graph_id\tdescription\tsmiles\tsuccess\tvalidation_passed\n")
        for graph_id, info in mapping.items():
            desc = info["description"] if info["description"] is not None else ""
            smiles = info["smiles"] if info["smiles"] is not None else ""
            if "test" in base:
                f.write(f"{graph_id}\t{smiles}\t{int(info['success'])}\t{int(info['validation_passed'])}\n")
            else:
                f.write(f"{graph_id}\t{desc}\t{smiles}\t{int(info['success'])}\t{int(info['validation_passed'])}\n")

    print(f"Saved human-readable SMILES mappings to {human_readable_path}")


# =========================================================
# Main Script
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Convert molecular graphs to SMILES")
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing graph pickle files')
    parser.add_argument('--output_dir', type=str, default='data/smiles',
                        help='Directory to save SMILES mappings')
    parser.add_argument('--canonical', action='store_true', default=True,
                        help='Generate canonical SMILES')
    parser.add_argument('--isomeric', action='store_true', default=True,
                        help='Include stereochemistry in SMILES')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                        help='Dataset splits to process')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    for split in args.splits:
        input_path = os.path.join(args.data_dir, f"{split}_graphs.pkl")

        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping {split}")
            continue

        print(f"\nProcessing {split} split...")

        with open(input_path, 'rb') as f:
            graphs = pickle.load(f)

        print(f"Loaded {len(graphs)} graphs")

        results = convert_dataset(
            graphs,
            canonical=args.canonical,
            isomeric=args.isomeric,
            validate=not args.no_validate,
            verbose=True
        )

        print_conversion_summary(results, split)

        output_path = os.path.join(args.output_dir, f"{split}_smiles.pkl")
        save_smiles_mapping(results, output_path)

        all_results[split] = results

    print(f"\n{'=' * 60}")
    print("Overall Summary")
    print(f"{'=' * 60}")

    total_graphs = sum(len(r) for r in all_results.values())
    total_success = sum(sum(1 for x in r if x.success) for r in all_results.values())
    total_validated = sum(sum(1 for x in r if x.validation_passed) for r in all_results.values())

    print(f"Total graphs processed: {total_graphs}")
    print(f"Total successful:       {total_success} ({100 * total_success / total_graphs:.1f}%)")
    print(f"Total validated:        {total_validated} ({100 * total_validated / total_graphs:.1f}%)")


if __name__ == "__main__":
    main()
