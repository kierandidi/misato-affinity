import h5py
import torch as t
import numpy as np
import os
import pickle
import torch
import scipy.spatial as ss
from src.utils.logutils import get_logger
from src.constants import ATOM_TYPE_MAP
from torch_geometric.utils import to_undirected

# Setup logger
logger = get_logger(__name__)

# Load atom mapping constant
ATOM_MAPPING = pickle.load(open(ATOM_TYPE_MAP, "rb"))

def get_atom_class(atoms, residues):
    """
    Retrieve atom classes based on atom and residue types.

    Parameters:
    - atoms (list): List of atom types.
    - residues (list): List of residue types.

    Returns:
    - list: List of corresponding atom classes.
    """
    return [ATOM_MAPPING[(atom, residue)] for atom, residue in zip(atoms, residues)]

def calc_atom_1hot(atom_idx):
    """
    Convert atom index to one-hot encoding.

    Parameters:
    - atom_idx (int): Index of the atom.

    Returns:
    - torch.Tensor: One-hot encoded tensor for the atom.
    """
    num_classes = len(set(ATOM_MAPPING.values()))
    return t.nn.functional.one_hot(atom_idx, num_classes=num_classes + 1)

def calculate_dist(coordinates, edge_dist_cutoff):
    """
    Calculate edges and edges weights based on distance cutoff.

    Parameters:
    - coordinates (numpy.ndarray): Array of coordinates.
    - edge_dist_cutoff (float): Distance cutoff for edges.

    Returns:
    - tuple: Edges and edge weights.
    """
    kd_tree = ss.KDTree(coordinates)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()
    edges = to_undirected(edges)
    edge_weights = torch.FloatTensor(
        [1.0 / (np.linalg.norm(coordinates[i] - coordinates[j]) + 1e-5) for i, j in edges.t()]
    ).view(-1)
    return edges, edge_weights

def load_entity(entity_id, md, qm, affinities, interaction_cutoff):
    """
    Load entity data and compute associated properties.

    Parameters:
    - entity_id (str): Identifier for the entity.
    - md (h5py.File): Molecular dynamics data file.
    - qm (h5py.File): Quantum mechanics data file.
    - affinities (h5py.File): Affinities data file.
    - interaction_cutoff (float): Distance cutoff for interactions.

    Returns:
    - tuple: Processed data for the entity.
    """

    # Retrieve molecular dynamics (MD) entity data
    if entity_id not in md:
        logger.error(f"Entity {entity_id} not found in MD file.")
        raise KeyError
    MDentity = md[entity_id]

    # Retrieve quantum mechanics (QM) entity data
    if entity_id not in qm:
        logger.error(f"Entity {entity_id} not found in QM file.")
        raise KeyError
    QMentity = qm[entity_id]

    # Get affinity data
    if entity_id not in affinities:
        logger.error(f"Entity {entity_id} not found in affinity file.")
        raise KeyError
    affinity = affinities[entity_id]["affinity"][()]

    # Extract features from MD entity data
    coordinates = t.as_tensor(MDentity["atoms_coordinates_ref"][:])
    chain_index = t.as_tensor(MDentity["molecules_begin_atom_index"][:])
    atoms = MDentity["atoms_type"]
    residues = MDentity["atoms_residue"]
    classes = get_atom_class(atoms, residues)
    x = calc_atom_1hot(t.as_tensor(classes)).float()

    # Compute MISATO features
    adaptability = np.array(
        t.as_tensor(MDentity["feature_atoms_adaptability"][:]).unsqueeze(1)
    ).astype(np.float32)

    # Extract QM properties for the ligand
    noHindices_lig = np.where(QMentity["atom_properties"]["atom_names"][()] != b"1")[0]
    polarizabilities_lig = t.as_tensor(QMentity["atom_properties"]["atom_properties_values"][:, 15][noHindices_lig])
    charges_lig = t.as_tensor(QMentity["atom_properties"]["atom_properties_values"][:, 7][noHindices_lig])

    # Set QM properties for the protein
    polarizabilities_prot = t.zeros(chain_index[-1])
    charges_prot = t.zeros(chain_index[-1])

    # Combine QM properties
    polarizabilities = t.cat((polarizabilities_prot, polarizabilities_lig)).unsqueeze(1)
    charges = t.cat((charges_prot, charges_lig)).unsqueeze(1)

    # MISATO features added
    x = t.cat((x, adaptability, polarizabilities, charges), 1)

    # Calculate interaction distances
    edge_index_prot, edge_attr_prot = calculate_dist(coordinates[: chain_index[-1]], interaction_cutoff)
    edge_index_lig, edge_attr_lig = calculate_dist(coordinates[chain_index[-1]:], interaction_cutoff)
    edge_index_com, edge_attr_com = calculate_dist(coordinates, interaction_cutoff)
    edge_index_ligfull, edge_attr_ligfull = calculate_dist(coordinates[chain_index[-1]:], 50.0)

    edges_index = (edge_index_prot, edge_index_lig, edge_index_com, edge_index_ligfull)
    edges_attr = (edge_attr_prot, edge_attr_lig, edge_attr_com, edge_attr_ligfull)

    return (
        x,
        edges_index,
        edges_attr,
        coordinates.float(),
        t.as_tensor(affinity),
        chain_index[-1],
    )

def write_h5(struct, md, qm, affinities, interaction_cutoff, oF):
    """
    Write processed entity data to an HDF5 file.

    Parameters:
    - struct (str): Identifier for the structure.
    - md (h5py.File): Molecular dynamics data file.
    - qm (h5py.File): Quantum mechanics data file.
    - affinities (h5py.File): Affinities data file.
    - interaction_cutoff (float): Distance cutoff for interactions.
    - oF (h5py.File): Output file to write data to.

    Returns:
    - None
    """

    # Load processed entity data
    (
        x,
        polarizabilities,
        charges,
        adaptabilities,
        edges_index,
        edges_attr,
        coordinates,
        affinity,
        ligand_begin_index,
    ) = load_entity(struct, md, qm, affinities, interaction_cutoff)

    # Extract edge attributes and indices
    edge_attr_prot, edge_attr_lig, edge_attr_com, edge_attr_ligfull = edges_attr
    edge_index_prot, edge_index_lig, edge_index_com, edge_index_ligfull = edges_index

    # Create a subgroup for the structure in the output file
    subgroup = oF.create_group(struct)

    # Add datasets to the subgroup
    datasets = {
        "atom_1hot": (x, "gzip"),
        "polarizabilities": (polarizabilities, "gzip"),
        "charges": (charges, "gzip"),
        "edge_index_prot": (edge_index_prot, "gzip"),
        "edge_index_lig": (edge_index_lig, "gzip"),
        "edge_index_com": (edge_index_com, "gzip"),
        "edge_index_ligfull": (edge_index_ligfull, "gzip"),
        "edge_attr_prot": (edge_attr_prot, "gzip"),
        "edge_attr_lig": (edge_attr_lig, "gzip"),
        "edge_attr_com": (edge_attr_com, "gzip"),
        "edge_attr_ligfull": (edge_attr_ligfull, "gzip"),
        "adaptabilities": (adaptabilities, "gzip"),
        "coordinates": (coordinates, "gzip"),
        "affinity": (affinity, None),
        "ligand_begin_index": (ligand_begin_index, None),
    }

    for key, (data, compression) in datasets.items():
        subgroup.create_dataset(key, data=data, compression=compression)


if __name__ == "__main__":
    # Paths
    md_path = "/p/project/hai_denovo/MISATO/adaptability_MD.hdf5"
    qm_path = "/p/project/hai_denovo/MISATO/QM.hdf5"
    affinity_path = "data/affinity_data.h5"
    output_path = "data/preprocessed_separate_combined_fullLigand_graphs_invariant.h5"
    
    if os.path.isfile(output_path):
        os.remove(output_path)
    
    # Open data files
    md = h5py.File(md_path, "r")
    qm = h5py.File(qm_path, "r")
    affinity = h5py.File(affinity_path, "r")
    
    # Load structure identifiers
    structs = pickle.load(open("data/affinity_structs_no_peptides.pickle", "rb"))
    
    output = h5py.File(output_path, "a")
    
    # Process and write data for each structure
    for i, struct in enumerate(structs, 1):
        logger.info(f"Processing structure {i}/{len(structs)}: {struct}")
        write_h5(struct, md, qm, affinity, 4.5, output)