import h5py
import pandas as pd
import os
import numpy as np
from collections import defaultdict


def check_affinity_type(df, index):
    series = df.iloc[index, 1:4] != 0.0
    affType = series.index[series].tolist()
    # print('afftype', affType, series[affType], df.iloc[1,1:4])
    return affType[0], float(df.iloc[index, 1:4][affType][0])


def write_h5_info(outName, df):
    """
    to read:
    dat = h5py.File('affinity_data.h5')
    dat['1W8L']['affinityType'][()].decode()
    dat['1W8L']['affinity'][()]
    """
    if os.path.isfile(outName):
        os.remove(outName)

    with h5py.File(outName, "w") as oF:
        for i in range(df.shape[0]):
            subgroup = oF.create_group(df["PDBid"][i])
            affType, affValue = check_affinity_type(df, i)
            subgroup.create_dataset("affinity", data=affValue)
            subgroup.create_dataset("affinityType", data=affType)
            subgroup.create_dataset("ligandID", data=df["ligand"][i])
            subgroup.create_dataset("proteinName", data=df["Protein"][i])
            subgroup.create_dataset("type", data=df["type"][i])
            subgroup.create_dataset("Uniprot", data=df["Uniprot"][i])


d = defaultdict(list)
df = pd.read_csv("../../../data/affinity_data.csv", on_bad_lines="skip", delimiter=";")

for i in range(df.shape[0]):
    d[df["Uniprot"][i]].append(df["PDBid"][i])

print(len(d.keys))
