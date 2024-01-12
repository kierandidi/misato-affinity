import h5py
import torch as t
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph
import os
from src.utils.logutils import get_logger
from src.constants import ATOM_TYPE_MAP

logger = get_logger(__name__)

# Constants
ATOM_MAPPING = pickle.load(open(ATOM_TYPE_MAP,'rb'))
# Convert atom index to one-hot encoding
def atom_1hot(atom_idx):
    """Convert atom index to one-hot encoding."""
    num_classes = len(ATOM_MAPPING)
    # had to add 1, probably due to null class...
    one_hot = t.nn.functional.one_hot(atom_idx, num_classes=num_classes+1)
    return one_hot

def get_numligand_edge_feature(edges, lig_index):
    p0_lig = (edges[0]>=lig_index).astype(int)
    p1_lig = (edges[1]>=lig_index).astype(int)
    numlig_feature = p0_lig+p1_lig
    return numlig_feature

def load_entity(entity_id, md, graph):
    """Load entity data and calculate properties."""
    try:
        MDentity = md[entity_id]
    except KeyError:
        logger.error(f'Entity {entity_id} not found in MD file.')
        raise
    try:
        graph_entity = graph[entity_id]
    except KeyError:
        logger.error(f'Entity {entity_id} not found in MD file.')
        raise

    chain_index = MDentity["molecules_begin_atom_index"][:]
    lig_index = chain_index[-1]

    edges = graph_entity["edge_index"][:]
    numlig_edges = get_numligand_edge_feature(edges, lig_index)

    return lig_index, numlig_edges


def write_h5(struct, md, graph, oF):
    ligand_begin_index, numlig_edges  = load_entity(struct, md, graph)
    #print('x',x.numpy(), type(x.numpy()))
    #print('aff', affinity, type(affinity))
    subgroup = oF.create_group(struct)
    subgroup.create_dataset('atom_1hot', data= graph[struct]['atom_1hot'], compression = "gzip")
    subgroup.create_dataset('polarizabilities', data= graph[struct]['polarizabilities'], compression = "gzip")
    subgroup.create_dataset('charges', data= graph[struct]['charges'], compression = "gzip")
    subgroup.create_dataset('edge_index', data= graph[struct]['edge_index'], compression = "gzip")
    subgroup.create_dataset('adaptabilities', data= graph[struct]['adaptabilities'], compression = "gzip")
    subgroup.create_dataset('edge_attr', data= graph[struct]['edge_attr'], compression = "gzip")
    subgroup.create_dataset('coordinates', data= graph[struct]['coordinates'], compression = "gzip")
    subgroup.create_dataset('affinity', data= graph[struct]['affinity'])
    subgroup.create_dataset('ligand_begin_index', data= ligand_begin_index)
    subgroup.create_dataset('edge_attr_numlig', data= numlig_edges, compression = "gzip")

if __name__=="__main__":
    md_path="/p/project/hai_denovo/MISATO/adaptability_MD.hdf5"
    qm_path="/p/project/hai_denovo/MISATO/QM.hdf5"
    affinity_path="data/affinity_data.h5"
    graph_path = 'data/preprocessed_graph_invariant.h5'
    output_path = 'data/preprocessed_graph_invariant_numlig.h5'

    md = h5py.File(md_path)
    graph = h5py.File(graph_path)
    #qm = h5py.File(qm_path)
    #affinity = h5py.File(affinity_path)
    structs = pickle.load(open('data/affinity_structs_no_peptides.pickle', 'rb'))
    output = h5py.File(output_path, 'a')
    i = 0
    for struct in structs:
        i+=1
        print(struct, i)
        write_h5(struct, md, graph, output)
# Test
#dataset = MDDataset(

# )
#data = dataset.get(10)
# #
#print(data)
