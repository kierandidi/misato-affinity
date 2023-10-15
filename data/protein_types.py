"""Mapping residue types to their respective encodings via binning."""

import pickle

pair = ["atom", "residue"]
if (pair[0] == "protein-C") or (pair[0] == "protein-CX"):
    pair[1] = "any"
if (pair[0] == "protein-N") and (pair[1] != "PRO"):
    pair[1] = "any"
if pair[0] == "protein-O":
    pair[1] = "any"
if pair[0] == "protein-N3":
    pair[1] = "any"

ProteinTypes_clean = {
    tuple(["protein-C", "any"]): 0,  # Ccarboxyl main chain
    tuple(["protein-N", "any"]): 1,  # Ncarboxyl with 1 R substituent
    tuple(["protein-N", "PRO"]): 2,  # Ncarboxyl with 2 R substituents
    tuple(["protein-O", "any"]): 3,  # Ocarboxyl main chain
    tuple(["protein-CX", "any"]): 4,  # Calpha
    tuple(["protein-CT", "PHE"]): 5,  # Cbenzyl
    tuple(["protein-CT", "TRP"]): 5,  # Cbenzyl
    tuple(["protein-CT", "HIS"]): 5,  # Cbenzyl
    tuple(["protein-CT", "TYR"]): 5,  # Cbenzyl
    tuple(["protein-2C", "CYS"]): 6,  # Caliphatic
    tuple(["protein-3C", "THR"]): 6,  # Caliphatic
    tuple(["protein-2C", "SER"]): 6,  # Caliphatic
    tuple(["protein-CT", "CYS"]): 6,  # Caliphatic
    tuple(["protein-CT", "ALA"]): 6,  # Caliphatic
    tuple(["protein-2C", "MET"]): 6,  # Caliphatic
    tuple(["protein-3C", "VAL"]): 6,  # Caliphatic
    tuple(["protein-2C", "LEU"]): 6,  # Caliphatic
    tuple(["protein-2C", "GLU"]): 6,  # Caliphatic
    tuple(["protein-3C", "ILE"]): 6,  # Caliphatic
    tuple(["protein-2C", "GLN"]): 6,  # Caliphatic
    tuple(["protein-CT", "PRO"]): 6,  # Caliphatic
    tuple(["protein-CT", "LYS"]): 6,  # Caliphatic
    tuple(["protein-C8", "LYS"]): 6,  # Caliphatic
    tuple(["protein-C8", "ARG"]): 6,  # Caliphatic
    tuple(["protein-2C", "ASP"]): 6,  # Caliphatic
    tuple(["protein-2C", "ASN"]): 6,  # Caliphatic
    tuple(["protein-CT", "VAL"]): 6,  # Caliphatic
    tuple(["protein-3C", "LEU"]): 6,  # Caliphatic
    tuple(["protein-CT", "ILE"]): 6,  # Caliphatic
    tuple(["protein-2C", "ILE"]): 6,  # Caliphatic
    tuple(["protein-CT", "THR"]): 6,  # Caliphatic
    tuple(["protein-CT", "LEU"]): 6,  # Caliphatic
    tuple(["protein-CT", "MET"]): 6,  # Caliphatic
    tuple(["protein-CO", "ASP"]): 7,  # Ccarboxyl side chain
    tuple(["protein-CO", "GLU"]): 7,  # Ccarboxyl side chain
    tuple(["protein-O2", "ASP"]): 8,  # Ocarboxyl
    tuple(["protein-O2", "GLU"]): 8,  # Ocarboxyl
    tuple(["protein-SH", "CYS"]): 9,  # Sthiol
    tuple(["protein-S", "MET"]): 10,  # Sthioether
    tuple(["protein-OH", "SER"]): 11,  # Oalcohol
    tuple(["protein-OH", "THR"]): 11,  # Oalcohol
    tuple(["protein-OH", "TYR"]): 12,  # Ophenol
    tuple(["protein-NA", "HIS"]): 13,  # Narom1
    tuple(["protein-NA", "TRP"]): 13,  # Narom1
    tuple(["protein-NB", "HIS"]): 14,  # Narom2
    tuple(["protein-CA", "PHE"]): 15,  # Caromatic simple
    tuple(["protein-C*", "TRP"]): 15,  # Caromatic simple
    tuple(["protein-CC", "HIS"]): 15,  # Caromatic simple
    tuple(["protein-CA", "TYR"]): 15,  # Caromatic para
    tuple(["protein-CW", "TRP"]): 15,  # Caromatic simple
    tuple(["protein-CB", "TRP"]): 15,  # Caromatic simple
    tuple(["protein-CW", "HIS"]): 15,  # Caromatic simple
    tuple(["protein-CV", "HIS"]): 15,  # Caromatic simple
    tuple(["protein-CA", "TRP"]): 15,  # Caromatic simple
    tuple(["protein-CR", "HIS"]): 15,  # Caromatic simple
    tuple(["protein-CA", "ARG"]): 16,  # Cguanidino
    tuple(["protein-N2", "ARG"]): 17,  # Nguanidino1
    tuple(["protein-N3", "any"]): 18  # Nammonium
    # residue HYP is missing from this collection
}

with open("protein_types.pickle", "wb") as f:
    pickle.dump(ProteinTypes_clean, f)
