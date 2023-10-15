"""Mapping atom types to their respective encodings via binning."""

import pickle

LigandTypes = {
    # hydrogen
    "hc": 10,
    "ha": 10,
    "h1": 10,
    "hn": 10,
    "ho": 10,
    "hx": 10,
    "h4": 10,
    "h2": 10,
    "h5": 10,
    "hs": 10,
    "h3": 10,
    "hp": 10,
    # boron
    "b": 50,
    # Caromatic
    "ca": 60,
    "cc": 60,
    "cd": 60,
    "cp": 60,
    "cq": 60,
    "cz": 60,
    # Cpurine (can be merged with Caromatic)
    "ce": 61,
    "cf": 61,
    # Cthiocarbonyl (can be merged with Ccarbonyl)
    "cs": 62,
    # Ccarbonyl
    "c": 63,
    # Csp
    "c1": 64,
    "cg": 64,
    "ch": 64,
    # Csp2
    "c2": 65,
    "cu": 65,
    # Csp3
    "c3": 66,
    "cx": 66,
    "cy": 66,
    # Nsp:
    "n1": 70,
    # Nsp2;
    "ne": 71,
    "nf": 71,
    "n2": 71,
    # Namide:
    "n": 72,
    "nj": 72,
    "ns": 72,
    "nt": 72,
    # Naniline:
    "nh": 73,
    "nn": 73,
    "nu": 73,
    # Npyridine:
    "nb": 74,
    "nd": 74,
    "nc": 74,
    "nv": 74,
    # Npyrrole:
    "na": 75,
    # N3:
    "n3": 76,
    "n8": 76,
    "n7": 76,
    # this might be mixture of different chemical types...
    # N+:
    "n4": 77,
    "nl": 77,
    "nk": 77,
    "nz": 77,
    "ny": 77,
    "nx": 77,
    "nq": 77,
    # Nnitro:
    "no": 78,
    # Ocarbonyl:
    "o": 80,
    # Oalcohol:
    "oh": 81,
    "os": 81,
    # Otension:
    "op": 82,
    "oq": 82,
    # Fluorine:
    "f": 90,
    # Phosphorous
    "p5": 150,
    "py": 150,
    # Sthiol:
    "ss": 160,
    "sh": 160,
    # Scarbonyl:
    "s": 161,
    # S3:
    "sx": 162,
    "s4": 162,
    # Sso2:
    "sy": 163,
    "s6": 163,
    # Chlorine:
    "cl": 170,
    # Bromine:
    "br": 350,
    # Iodine:
    "i": 530,
    # ignore
    "nm": -1,
}

with open("ligand_types.pickle", "wb") as f:
    pickle.dump(LigandTypes, f)
