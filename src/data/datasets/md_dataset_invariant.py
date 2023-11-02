import h5py
import torch as t
from torch_geometric.data import Dataset, Data
from src.utils.logutils import get_logger
import pickle

logger = get_logger(__name__)

def normalized(feature):
    feature = feature - feature.mean()
    feature = feature / feature.std()
    return feature


def load_entity(entity_id, graph, affinity_h5, misato_features=True):
    """
    Load entity data and calculate properties.

    Parameters:
    - entity_id (str): ID of the entity to be loaded.
    - graph (h5py.File): HDF5 file containing graph data.
    - affinity_h5 (h5py.File): HDF5 file containing affinity data.
    - misato_features (bool, optional): Whether to add MISATO features.

    Returns:
    - Data: A torch_geometric.data.Data object containing the loaded data.
    """
    try:
        graph_entity = graph[entity_id]
    except KeyError:
        logger.error(f"Entity {entity_id} not found in MD file.")
        raise

    try:
        affinity_entity = affinity_h5[entity_id]
    except KeyError:
        logger.error(f"Entity {entity_id} not found in affinity file.")
        raise
    
    # Extracting features from the graph data
    coordinates = t.as_tensor(graph_entity["coordinates"][:]).to(t.float32)
    adaptabilities = t.as_tensor(graph_entity["adaptabilities"][:])
    charges = normalized(t.as_tensor(graph_entity["charges"][:]))
    edge_index = t.as_tensor(graph_entity["edge_index"][:])
    edge_attr = t.as_tensor(graph_entity["edge_attr"][:]).unsqueeze(1)
    x = t.as_tensor(graph_entity["atom_1hot"][:])
    lig_index = graph_entity["ligand_begin_index"][()]
    affinity = t.as_tensor(affinity_entity["affinity"][()]).to(t.float32)
    
    if misato_features:
        lig_len = int(charges.shape[0]) - lig_index
        nodes_numlig = t.cat((t.zeros(lig_index), t.ones(lig_len))).unsqueeze(1)
        x = t.cat((x, adaptabilities, charges, nodes_numlig), 1)
    else:
        x = t.cat((x, charges), 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=affinity, pdb_id=entity_id)


class MDDataset(Dataset):
    def __init__(self, graph_path, pair_path, affinity_path, misato_features=True):
        """
        Initialize MD dataset.

        Parameters:
        - graph_path (str): Path to the HDF5 file containing graph data.
        - pair_path (str): Path to the pickle file containing pair data.
        - affinity_path (str): Path to the HDF5 file containing affinity data.
        - misato_features (bool, optional): Whether to add MISATO features.
        """
        super().__init__(graph_path)
        self.graph = h5py.File(graph_path, "r")
        self.affinity = h5py.File(affinity_path, "r")
        self.misato_features = misato_features
        with open(pair_path, "rb") as fp:
            self.pairs = pickle.load(fp)

    def __len__(self):
        """Return length of the dataset."""
        return len(self.pairs)

    def get(self, idx):
        """
        Get item at index idx.

        Parameters:
        - idx (int): Index of the item.

        Returns:
        - tuple: Two Data objects representing the two entities in the pair.
        """
        pair = self.pairs[idx]
        data1 = load_entity(pair[0], self.graph, self.affinity, self.misato_features)
        data2 = load_entity(pair[1], self.graph, self.affinity, self.misato_features)
        
        return data1, data2

