import h5py
import pickle
import os
from src.utils.logutils import get_logger


logger = get_logger(__name__)

def convert_affinityType(affType):
    if affType == 'Kd (nM)':
        converted_affType = 0
    if affType == 'Ki (nM)':
        converted_affType = 1
    if affType == 'IC50 (nM)':
        converted_affType = 2
    return converted_affType

def load_entity(entity_id, md):
    """Load entity data and calculate properties."""
    try:
        MDentity = md[entity_id]
    except KeyError:
        logger.error(f'Entity {entity_id} not found in MD file.')
        raise

    affinityType_str = MDentity['affinityType'][()].decode()
    converted_affType = convert_affinityType(affinityType_str)

    return converted_affType


def write_h5(struct, md, oF):
    affType = load_entity(struct, md)
    oF[struct].create_dataset('affinity_type', data= affType)



if __name__=="__main__":
    aff_path="data/affinity_data.h5"
    affinity_path="data/affinity_data.h5"
    output_path = 'data/preprocessed_graph_invariant_affTypes.h5'
    aff = h5py.File(aff_path)

    structs = pickle.load(open('data/affinity_structs.pickle', 'rb'))
    output = h5py.File(output_path, 'a')

    for struct in structs:
        write_h5(struct, aff, output)
