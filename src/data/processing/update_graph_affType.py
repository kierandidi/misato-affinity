import h5py
import pandas as pd
import os
from collections import defaultdict


def check_affinity_type(df, index):
    """
    Check the affinity type for a given row in the dataframe.

    Parameters:
    - df (pandas.DataFrame): Dataframe containing affinity data.
    - index (int): Row index to check.

    Returns:
    - tuple: Affinity type and its corresponding value.
    """
    series = df.iloc[index, 1:4] != 0.0
    affType = series.index[series].tolist()

    return affType[0], float(df.iloc[index, 1:4][affType][0])


def write_h5_info(outName, df):
    """
    Write affinity information to an HDF5 file.

    Parameters:
    - outName (str): Name of the output file.
    - df (pandas.DataFrame): Dataframe containing affinity data.

    Returns:
    - None
    """
    if os.path.isfile(outName):
        os.remove(outName)

    with h5py.File(outName, "w") as oF:
        for i in range(df.shape[0]):
            subgroup = oF.create_group(df["PDBid"][i])
            affType, affValue = check_affinity_type(df, i)
            
            datasets = {
                "affinity": affValue,
                "affinityType": affType,
                "ligandID": df["ligand"][i],
                "proteinName": df["Protein"][i],
                "type": df["type"][i],
                "Uniprot": df["Uniprot"][i]
            }

            for key, data in datasets.items():
                subgroup.create_dataset(key, data=data)


if __name__ == "__main__":
    d = defaultdict(list)
    df = pd.read_csv("../../../data/affinity_data.csv", on_bad_lines="skip", delimiter=";")

    for i in range(df.shape[0]):
        d[df["Uniprot"][i]].append(df["PDBid"][i])

    print(len(d.keys()))
