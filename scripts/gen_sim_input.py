import os.path as osp
import sys

import os.path as osp
import sys
import numpy as np
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))



from input_generator.raw_dataset import SampleCollection, RawDataset, SimInput
from input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from input_generator.raw_data_loader import DatasetLoader, SimInput_loader
from input_generator.prior_gen import Bonds, PriorBuilder
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Optional
from jsonargparse import CLI
import pickle as pck

import numpy as np

from mlcg.data import AtomicData
import torch
from copy import deepcopy

def process_sim_input(
    dataset_name: str,
    names: List[str],
    raw_data_dir: str,
    save_dir: str,
    tag: str,
    sample_loader: DatasetLoader,
    pdb_fns: List[str],
    cg_atoms: List[str],
    embedding_map: str, #CGEmbeddingMap,
    embedding_func: str,# Callable,
    skip_residues: List[str],
    use_terminal_embeddings: bool,
    copies: int,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    cg_mapping_strategy: str,
    pdb_path_aa: str,
    map_path_aa: str,
    itp_path_cg: str,


    mass_scale: Optional[float] = 418.4,


):

    """
    Generates input AtomicData objects for coarse-grained simulations

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    raw_data_dir : str
        Path to location of input structures
    pdb_fns : str
        List of pdb filenames from which input will be generated
    save_dir : str
        Path to directory in which output will be saved
    tag : str
        Label given to all output files produced from dataset
    cg_atoms : List[str]
        List of atom names to preserve in coarse-grained resolution
    embedding_map : CGEmbeddingMap
        Mapping object
    embedding_func : Callable
        Function which will be used to apply CG mapping
    skip_residues : List[str]
        List of residues to skip, can be None
    use_terminal_embeddings : bool
        Whether separate embedding types should be assigned to terminal atoms
    cg_mapping_strategy : str
        Strategy to use for coordinate and force mappings;
        currently only "slice_aggregate" and "slice_optimize" are implemented
    """
    print('ENTRA: ¡')
    cg_coord_list = []
    cg_type_list = []
    cg_mass_list = []
    cg_nls_list = []

    dataset = SimInput(dataset_name, tag, pdb_fns)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        sample_loader = SimInput_loader()
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            name=samples.name, raw_data_dir=raw_data_dir
        )

        samples.apply_cg_mapping(
            cg_atoms=cg_atoms,
            embedding_function=embedding_func,
            embedding_dict=embedding_map,
            skip_residues=skip_residues,

            pdb_file = pdb_path_aa,
            map_file = map_path_aa,
            itp_file_cg = itp_path_cg,
        )

        if use_terminal_embeddings:
            # TODO: fix usage add_terminal_embeddings wrt inputs
            samples.add_terminal_embeddings(
                N_term=sub_data_dict["N_term"], C_term=sub_data_dict["C_term"]
            )
        if cg_mapping_strategy == 'cg_mapping':
            cg_trajs = samples.cg_traj
            print('CG_TRAJ_ ',cg_trajs)
        else:
            cg_trajs = samples.input_traj.atom_slice(samples.cg_atom_indices)

        cg_masses = (
            np.array([int(atom.element.mass) for atom in cg_trajs[0].topology.atoms])
            / mass_scale
        )
        prior_nls = samples.get_prior_nls(
            prior_builders, save_nls=False, save_dir=save_dir, prior_tag=prior_tag, name = cg_mapping_strategy, pdb_file = pdb_path_aa, map_file = map_path_aa, itp_file_cg = itp_path_cg,
        )
        
        cg_types = samples.cg_dataframe["type"].to_list()
        for i in range(cg_trajs.n_frames):
            cg_traj = cg_trajs[i]
            cg_coords = cg_traj.xyz * 10
            for i in range(copies):
                cg_coord_list.append(cg_coords)
                cg_type_list.append(cg_types)
                cg_mass_list.append(cg_masses)
                cg_nls_list.append(prior_nls)

    data_list = []
    print('MASAS :',cg_mass_list)
    for coords, types, masses, nls in zip(
        cg_coord_list, cg_type_list, cg_mass_list, cg_nls_list
    ):
        data = AtomicData.from_points(
            pos=torch.tensor(coords[0]),
            atom_types=torch.tensor(types),
            masses=torch.tensor(masses),
        )
        data.neighbor_list = deepcopy(nls)
        data_list.append(data)

    torch.save(data_list, f"{save_dir}{dataset_name}_configurations.pt")
if __name__ == "__main__":
    print("Start gen_sim_input.py: {}".format(ctime()))

    CLI([process_sim_input])

    print("Finish gen_sim_input.py: {}".format(ctime()))
