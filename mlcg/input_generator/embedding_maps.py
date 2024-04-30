from typing import Dict

embedding_map_fivebead = {
    "ALA": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "NLE": 10,  # Type Norleucine as Leucine
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}


class CGEmbeddingMap(dict):
    """
    General class for defining embedding maps as Dict
    """

    def __init__(self, embedding_map_dict: Dict[str, int]):
        for k, v in embedding_map_dict.items():
            self[k] = v


class CGEmbeddingMapFiveBead(CGEmbeddingMap):
    """
    Five-bead embedding map defined by:
        - N : backbone nitrogen
        - CA : backbone alpha carbon (specialized for glycing)
        - C : backbone carbonyl carbon
        - O : backbone carbonyl oxygen
        - CB : residue-specific beta carbon
    """

    def __init__(self):
        super().__init__(embedding_map_fivebead)


all_residues = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def embedding_fivebead(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_fivebead[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_fivebead["GLY"]
        else:
            atom_type = embedding_map_fivebead[name]
    elif name == "CB":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    print('ATOM_TYPE :',atom_type)
    return atom_type










#Membrane class:
embedding_map_membrane = {
        "C" : 1,
        "N" : 2,
        "P" : 3,
        "D" : 4,
        "G" : 5,
        }

class CGEmbeddingMapMembrane(CGEmbeddingMap):
    """
    """

    def __init__(self):
        super().__init__(embedding_map_membrane)

def embedding_membrane(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    print('CABECERAS: ',atom_df.columns)
    atom_type = embedding_map_membrane[name]
    print('ATOM_TYPE :',atom_type)
    return atom_type

