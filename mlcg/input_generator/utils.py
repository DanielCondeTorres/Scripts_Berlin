import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import mdtraj as md
from functools import wraps

from aggforce import linearmap as lm
from aggforce import agg as ag
from aggforce import constfinder as cf

from .prior_gen import PriorBuilder

#Función nueva añadida para el cg_mapping:
import argparse
from pathlib import Path
from copy import deepcopy
import mdtraj as md
import numpy as np
# load the system
def create_martini_cg(pdb_file, map_file, path_to_popc_itp='popc.itp'):
    print('CREANDO MARTINI CG ...')
    print(pdb_file)
    system = md.load_pdb(pdb_file)
    system = system.remove_solvent()
    num_lipid = len(set([atom.residue.resSeq for atom in system.top.atoms]))
    with open(map_file, 'r') as file:
        lines = file.readlines()
    martini_bead_list = []
    atoms_per_bead = []
    in_martini_section = False
    in_atoms_section = False

    # Iterating over martini map lines
    for i, line in enumerate(lines):
        if line.strip() == '[ martini ]':
            in_martini_section = True
            continue

        elif in_martini_section and line.startswith('['):
            in_martini_section = False

        if line.strip() == '[ atoms ]':
            in_atoms_section = True
            continue
        elif in_atoms_section and line.startswith('['):
            in_atoms_section = False
            # process line accordint to section
        if in_martini_section and line.strip():
            martini_bead_list.extend(line.strip().split())
        elif in_atoms_section and line.strip():
            columns = line.split()
            atoms_per_bead.append(columns[1:])
    #print("Martini bead List:", martini_bead_list)
    #print('Atoms per bead: ',atoms_per_bead)

    output_dict = {item[0]: item[1:] for item in atoms_per_bead if item}
    atoms_per_lip = list(output_dict.keys())
    masses = {'H': 1,'C': 12,'N': 14,'O': 16,'S': 32,'P': 31,'M': 0, 'B': 32}

    cg_map = np.zeros((num_lipid*len(martini_bead_list),system.n_atoms))

    for c,res in enumerate(system.top.residues):
        for atom in res.atoms:
            aa_idx = atom.index
            m = masses[atom.name[0]]
            contributions = output_dict[atom.name]
            cg_indexes = [martini_bead_list.index(bead) for bead in contributions]
            cg_indexes = [idx+c*len(martini_bead_list) for idx in cg_indexes]
            cg_map[cg_indexes,aa_idx] = m
        # normalize the cg_map
    col_sum = cg_map.sum(axis=1)
    cg_map = (cg_map.T*(1/col_sum)).T
    new_coords =  cg_map @ system.xyz
    # As the CG change is not a simple slicing of the original AA topology
    # we need to create the new topology from scratch.
    new_top = md.Topology()
    new_top.add_chain()
    # mdtraj requires that you asing elements to its atoms.
    # although the correct would be to use virutal, we use 
    # this mapping for convenience in visualization
    element_map = {
        "C" : md.element.carbon,
        "N" : md.element.nitrogen,
        "P" : md.element.phosphorus,
        "D" : md.element.boron,
        "G" : md.element.virtual_site,
                    }

    # Leer el archivo popc.itp para obtener información sobre enlaces
    with open(path_to_popc_itp, 'r') as file:
        lines = file.readlines()

    # Encontrar la sección [bonds] y extraer información
    bonds_section = False
    enlaces_definidos = []

    for line in lines:
        if line.startswith('[bonds]'):
            bonds_section = True
        elif line.startswith('['):
            bonds_section = False
        elif bonds_section and line.strip():
            # Dividir la línea en elementos y obtener información relevante
            elementos = line.split()
            bead1_index, bead2_index = int(elementos[0]), int(elementos[1])
            enlaces_definidos.append((bead1_index, bead2_index))

    for i in range(num_lipid):
        new_top.add_residue(f"POP",new_top.chain(0))
        for j,elem in enumerate(martini_bead_list,start=1):
            md_elem = element_map[elem[0]]
            new_atom = new_top.add_atom(f"{elem}{j}", md_elem, new_top.residue(i))
            
        for enlace in enlaces_definidos:
            bead1_index, bead2_index = enlace
            #print('Bead 1 de enlace',new_top.atom(bead1_index - 1))
            #print('Bead 2 de enlace',new_top.atom(bead2_index - 1))
            new_top.add_bond(new_top.atom(bead1_index - 1), new_top.atom(bead2_index - 1))

    print('New top: ',new_top)
    config_map_matrix = cg_map
    force_map_matrix = cg_map*4.184
    return config_map_matrix, force_map_matrix, new_coords, new_top, system
#Fin función nueva añadida
def with_attrs(**func_attrs):
    """Set attributes in the decorated function, at definition time.
    Only accepts keyword arguments.
    """

    def attr_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        for attr, value in func_attrs.items():
            setattr(wrapper, attr, value)

        return wrapper

    return attr_decorator


def map_cg_topology(
    atom_df: pd.DataFrame,
    cg_atoms: List[str],
    embedding_function: str,
    skip_residues: Optional[Union[List, str]] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    atom_df:
        Pandas DataFrame row from mdTraj topology.
    cg_atoms:
        List of atoms needed in CG mapping.
    embedding_function:
        Function that slices coodinates, if not provided will fail.
    special_typing:
        Optional dictionary of alternative atom properties to use in assigning types instead of atom names.
    skip_residues:
        Optional list of residues to skip when assigning CG atoms (can be used to skip caps for example);
        As of now, this skips all instances of a given residue.


    Returns
    -------
    New DataFrame columns indicating atom involvement in CG mapping and type assignment.

    Example
    -------
    First obtain a Pandas DataFrame object using the built-in MDTraj function:
    >>> top_df = aa_traj.topology.to_dataframe()[0]

    For a five-bead resolution mapping without including caps:
    >>> cg_atoms = ["N", "CA", "CB", "C", "O"]
    >>> embedding_function = embedding_fivebead
    >>> skip_residues = ["ACE", "NME"]

    Apply row-wise function:
    >>> top_df = top_df.apply(map_cg_topology, axis=1, cg_atoms, embedding_dict, skip_residues)
    """
    print('Embeding funcion en utils: ',embedding_function)
    #if embedding_function == 'cg_mapping': 
    #    config_map_matrix,force_map_matrix, new_coords, cg_top, system = create_martini_cg(pdb_file, map_file, itp_file_cg)
    #    traj = md.Trajectory(xyz = new_coords, topology = new_top)
    #    atom_df = newtop.to_dataframe()
    
    #else:
    if isinstance(embedding_function, str):
        try:
            embedding_function = eval(embedding_function)
        except NameError:
            print("The specified embedding function has not been defined.")
            exit

    name, res = atom_df["name"], atom_df["resName"]
    print('CABECERAS_ ',atom_df.columns)
    if skip_residues != None and res in skip_residues:
        atom_df["mapped"] = False
        atom_df["type"] = "NA"
    else:
        if name in cg_atoms:
            atom_df["mapped"] = True
            atom_type = embedding_function(atom_df)
            atom_df["type"] = atom_type
        else:
            atom_df["mapped"] = False
            atom_df["type"] = "NA"
    print('ATOM DF; ',atom_df)
    return atom_df


def slice_coord_forces(
    coords, forces, cg_map, mapping: str = "slice_aggregate", force_stride: int = 100, pdb_file = None, map_file = None, itp_file_cg = None
) -> Tuple:
    """
    Parameters
    ----------
    coords: [n_frames, n_atoms, 3]
        Numpy array of atomistic coordinates
    forces: [n_frames, n_atoms, 3]
        Numpy array of atomistic forces
    cg_map: [n_cg_atoms, n_atomistic_atoms]
        Linear map characterizing the atomistic to CG configurational map with shape.
    mapping:
        Mapping scheme to be used, must be either 'slice_aggregate' or 'slice_optimize'.
    force_stride:
        Striding to use for force projection results

    Returns
    -------
    Coarse-grained coordinates and forces
    """
    print('SLICE COORD FORCES?: ', mapping)
    if mapping == 'slice_aggregate' or mapping == "slice_optimize":
        config_map = lm.LinearMap(cg_map)
        config_map_matrix = config_map.standard_matrix
        # taking only first 100 frames gives same results in ~1/15th of time
        constraints = cf.guess_pairwise_constraints(coords[:100], threshold=5e-3)
        print('vale, peta aqui no?')
        if mapping == "slice_aggregate":
            method = lm.constraint_aware_uni_map
            force_agg_results = ag.project_forces(
                                xyz=None,
                                forces=forces[::force_stride],
                                config_mapping=config_map,
                                constrained_inds=constraints,
                                method=method,
                                )
            force_map_matrix = force_agg_results["map"].standard_matrix #added
            new_top= None
        elif mapping == "slice_optimize":
            method = lm.qp_linear_map
            l2 = 1e3
            force_agg_results = ag.project_forces(
                                xyz=None,
                                forces=forces[::force_stride],
                                config_mapping=config_map,
                                constrained_inds=constraints,
                                method=method,
                                l2_regularization=l2,
                                )
            force_map_matrix = force_agg_results["map"].standard_matrix #added
            new_top = None

    #Añado un nuevo bloque que incluye el nuevo mapping, simplemente borrar para que quede libre
    elif mapping == "cg_mapping":
        print('Entro en mi mapping')
        config_map_matrix, force_map_matrix, new_coords, new_top, system = create_martini_cg(pdb_file, map_file, path_to_popc_itp = itp_file_cg)
    # Fin del nuevo bloque
    


    else:
        raise RuntimeError(
            f"Force mapping {mapping} is neither 'slice_aggregate' nor 'slice_optimize'."
        )

    #force_map_matrix = force_agg_results["map"].standard_matrix
    cg_coords = config_map_matrix @ coords
    cg_forces = force_map_matrix @ forces

    return cg_coords, cg_forces, new_top


def get_terminal_atoms(
    prior_builder: PriorBuilder,
    cg_dataframe: pd.DataFrame,
    N_term: Union[None, str] = None,
    C_term: Union[None, str] = None,
) -> Dict:
    """
    Parameters
    ----------
    prior_builder:

    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).
    N_term: (Optional)
        Atom used in definition of N-terminus embedding.
    C_term: (Optional)
        Atom used in definition of C-terminus embedding.
    """
    chains = cg_dataframe.chainID.unique()
    # all atoms belonging to monopeptide chains will be removed from termini list
    monopeptide_atoms = []
    for chain in chains:
        residues = cg_dataframe.loc[cg_dataframe.chainID == chain].resSeq.unique()
        if len(residues) == 1:
            monopeptide_atoms.extend(
                cg_dataframe.loc[cg_dataframe.chainID == chain].index.to_list()
            )

    first_res, last_res = cg_dataframe["resSeq"].min(), cg_dataframe["resSeq"].max()
    n_term_atoms = cg_dataframe.loc[
        (cg_dataframe["resSeq"] == first_res)
    ].index.to_list()
    c_term_atoms = cg_dataframe.loc[
        (cg_dataframe["resSeq"] == last_res)
    ].index.to_list()

    prior_builder.n_term_atoms = [a for a in n_term_atoms if a not in monopeptide_atoms]
    prior_builder.c_term_atoms = [a for a in c_term_atoms if a not in monopeptide_atoms]

    if N_term != None:
        prior_builder.n_atoms = cg_dataframe.loc[
            (cg_dataframe["resSeq"] == first_res) & (cg_dataframe["name"] == N_term)
        ].index.to_list()
    else:
        prior_builder.n_atoms = cg_dataframe.loc[
            (cg_dataframe["resSeq"] == first_res) & (cg_dataframe["name"] == "N")
        ].index.to_list()
    if N_term != None:
        prior_builder.c_atoms = cg_dataframe.loc[
            (cg_dataframe["resSeq"] == last_res) & (cg_dataframe["name"] == C_term)
        ].index.to_list()
    else:
        prior_builder.c_atoms = cg_dataframe.loc[
            (cg_dataframe["resSeq"] == first_res) & (cg_dataframe["name"] == "C")
        ].index.to_list()

    return prior_builder


def get_edges_and_orders(
    prior_builders: List[PriorBuilder],
    topology: md.Topology,
) -> List:
    """
    Parameters
    ----------
    prior_builders:
        List of PriorBuilder's to use for defining neighbour lists
    topology:
        MDTraj topology object from which atom groups defining each prior term will be created.
    cg_dataframe:
        Dataframe of CG topology (from MDTraj topology object).

    Returns
    -------
    List of edges, orders, and tag for each prior term specified in prior_dict.
    """
    all_edges_and_orders = []
    # process bond priors
    bond_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "bonds"
    ]
    all_bond_edges = []
    for prior_builder in bond_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
            all_bond_edges.extend([p[2] for p in edges_and_orders])
        else:
            all_edges_and_orders.append(edges_and_orders)
            all_bond_edges.append(edges_and_orders[2])

    # process angle priors
    angle_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "angles"
    ]
    all_angle_edges = []
    for prior_builder in angle_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
            all_angle_edges.extend([p[2] for p in edges_and_orders])
        else:
            all_edges_and_orders.append(edges_and_orders)
            all_angle_edges.append(edges_and_orders[2])

    # get nonbonded priors using bonded and angle edges
    if len(all_bond_edges) != 0:
        all_bond_edges = np.concatenate(all_bond_edges, axis=1)
    if len(all_angle_edges) != 0:
        all_angle_edges = np.concatenate(all_angle_edges, axis=1)

    nonbonded_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "non_bonded"
    ]
    for prior_builder in nonbonded_builders:
        edges_and_orders = prior_builder.build_nl(
            topology, bond_edges=all_bond_edges, angle_edges=all_angle_edges
        )
        # edges_and_orders = prior_dict[nbdict]["prior_function"](topology, all_bond_edges, all_angle_edges, **prior_dict[nbdict])
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)
    # process dihedral priors
    dihedral_builders = [
        prior_builder
        for prior_builder in prior_builders
        if prior_builder.type == "dihedrals"
    ]
    for prior_builder in dihedral_builders:
        edges_and_orders = prior_builder.build_nl(topology)
        if isinstance(edges_and_orders, list):
            all_edges_and_orders.extend(edges_and_orders)
        else:
            all_edges_and_orders.append(edges_and_orders)

    return all_edges_and_orders


def split_bulk_termini(N_term, C_term, all_edges) -> Tuple:
    """
    Parameters
    ----------
    N_term:
        List of atom indices to be split as part of the N-terminal.
    C_term:
        List of atom indices to be split as part of the C-terminal.
    all_edges:
        All atom groups forming part of prior term.

    Returns
    -------
    Separated edges for bulk and terminal groups
    """
    n_term_idx = np.where(np.isin(all_edges.T, N_term))
    n_term_edges = all_edges[:, np.unique(n_term_idx[0])]

    c_term_idx = np.where(np.isin(all_edges.T, C_term))
    c_term_edges = all_edges[:, np.unique(c_term_idx[0])]

    term_edges = np.concatenate([n_term_edges, c_term_edges], axis=1)
    bulk_edges = np.array(
        [e for e in all_edges.T if not np.all(term_edges == e[:, None], axis=0).any()]
    ).T

    return n_term_edges, c_term_edges, bulk_edges


def get_dihedral_groups(
    top: md.Topology, atoms_needed: List[str], offset: List[int], tag: Optional[str]
) -> Dict:
    """
    Parameters
    ----------
    top:
        MDTraj topology object.
    atoms_needed: [4]
        Names of atoms forming dihedrals, should correspond to existing atom name in topology.
    offset: [4]
        Residue offset of each atom in atoms_needed from starting point.
    tag:
        Dihedral prior tag.

    Returns
    -------
    Dictionary of atom groups for each residue corresponding to dihedrals.

    Example
    -------
    To obtain all phi dihedral atom groups for a backbone-preserving resolution:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["C", "N", "CA", "C"], offset=[-1.,0.,0.,0.], tag="_phi"
    >>> )

    For a one-bead-per-residue mapping with only CA atoms preserved:
    >>> dihedral_dict = get_dihedral_groups(
    >>>     topology, atoms_needed=["CA", "CA", "CA", "CA"], offset=[-3.,-2.,-1.,0.]
    >>> )
    """
    res_per_chain = [[res for res in chain.residues] for chain in top.chains]
    atom_groups = {}
    for chain_idx, chain in enumerate(res_per_chain):
        for res in chain:
            res_idx = chain.index(res)
            if any(res_idx + ofs < 0 or res_idx + ofs >= len(chain) for ofs in offset):
                continue
            if any(atom not in [a.name for a in res.atoms] for atom in atoms_needed):
                continue
            label = f"{res.name}{tag}"
            if label not in atom_groups:
                atom_groups[label] = []
            dihedral = []
            for i, atom in enumerate(atoms_needed):
                atom_idx = top.select(
                    f"(chainid {chain_idx}) and (resid {res.index+offset[i]}) and (name {atom})"
                )
                dihedral.append(atom_idx)
            atom_groups[label].append(np.concatenate(dihedral))

    return atom_groups
