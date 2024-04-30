import os.path as osp
import sys
import numpy as np
SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from input_generator.raw_dataset import SampleCollection, RawDataset
from input_generator.embedding_maps import (
    CGEmbeddingMap,
)
from input_generator.raw_data_loader import DatasetLoader
from input_generator.prior_gen import Bonds, PriorBuilder
from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable, Any, List, Optional, Union
from jsonargparse import CLI
import pickle as pck


def process_raw_dataset(
    dataset_name: str,
    names: List[str],
    sample_loader: DatasetLoader,
    raw_data_dir: str,
    tag: str,
    pdb_template_fn: str,
    save_dir: str,
    cg_atoms: List[str],
    embedding_map:  Union[CGEmbeddingMap,Optional[Any]],
    embedding_func: Union[Callable,Optional[Any]],
    skip_residues: List[str],
    use_terminal_embeddings: bool,
    cg_mapping_strategy: str,
    
    pdb_path_aa: str,
    map_path_aa: str,
    itp_path_cg: str,
):
    """
    Applies coarse-grained mapping to coordinates and forces using input sample
    topology and specified mapping strategies

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    sample_loader : DatasetLoader
        Loader object defined for specific dataset
    raw_data_dir : str
        Path to coordinate and force files
    tag : str
        Label given to all output files produced from dataset
    pdb_template_fn : str
        Template file location of atomistic structure to be used for topology
    save_dir : str
        Path to directory in which output will be saved
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
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_builders : List[PriorBuilder]
        List of PriorBuilder objects and their corresponding parameters

    """
    dataset = RawDataset(dataset_name, names, tag)
    print('Datos: ',dataset)
    for samples in tqdm(dataset, f"Processing CG data for {dataset_name} dataset..."):
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            samples.name, pdb_template_fn
        )
        print('Sample: ',samples)
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


        print('Cargando coordenadas en scripts input.py')
        
        '''         
        #Version vieja
        aa_coords, aa_forces = sample_loader.load_coords_forces(
            raw_data_dir, samples.name)
        print('Coordenadas cargadas en script input.py')
        cg_coords, cg_forces, new_top = samples.process_coords_forces(
            aa_coords, aa_forces, mapping=cg_mapping_strategy, force_stride = 10000, pdb_file = pdb_path_aa, map_file = map_path_aa, itp_file_cg = itp_path_cg #añadidas
         )
        print('SHAPE_TOTAL: ',cg_coords.shape)
        #Version vieja fin
            
        '''
        #CHAT GPT VERSION
        cg_coords = []; cg_forces = []
        for aa_coords, aa_forces in sample_loader.load_coords_forces_a(raw_data_dir, samples.name):
            cg_coords_1, cg_forces_1, new_top = samples.process_coords_forces(aa_coords, aa_forces, mapping=cg_mapping_strategy, force_stride = 10000, pdb_file = pdb_path_aa, map_file = map_path_aa, itp_file_cg = itp_path_cg)
            cg_coords.append(cg_coords_1);cg_forces.append(cg_forces_1)
            print('TAMAÑO AUMENTA?: ',len(cg_coords))
        cg_coords = np.concatenate(cg_coords,axis=0);cg_forces = np.concatenate(cg_forces,axis=0)
        print('SHAPE_TOTAL: ',cg_coords.shape)
        #END GPT VERSION
        print('ESTRATEGIA: ',cg_mapping_strategy)
        samples.save_cg_output(save_dir, True,cg_coords,cg_forces)


def build_neighborlists(
    dataset_name: str,
    names: List[str],
    sample_loader: DatasetLoader,
    tag: str,
    pdb_template_fn: str,
    save_dir: str,
    cg_atoms: List[str],
    embedding_map:  Union[CGEmbeddingMap,Optional[Any]],
    embedding_func: Union[Callable,Optional[Any]],

    skip_residues: List[str],
    use_terminal_embeddings: bool,
    prior_tag: str,
    prior_builders: List[PriorBuilder],
    
    raw_data_dir: str,
    cg_mapping_strategy: str,
    pdb_path_aa: str,
    map_path_aa: str,
    itp_path_cg: str,
):
    """
    Generates neighbour lists for all samples in dataset using prior term information

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    sample_loader : DatasetLoader
        Loader object defined for specific dataset
    tag : str
        Label given to all output files produced from dataset
    pdb_template_fn : str
        Template file location of atomistic structure to be used for topology
    save_dir : str
        Path to directory in which output will be saved
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
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_builders : List[PriorBuilder]
        List of PriorBuilder objects and their corresponding parameters
    """
    print('DONDE GUARDAR: ',save_dir)
    dataset = RawDataset(dataset_name, names, tag)
    for samples in tqdm(dataset, f"Building NL for {dataset_name} dataset..."):
        print('Save dir:' ,save_dir)
        samples.input_traj, samples.top_dataframe = sample_loader.get_traj_top(
            samples.name, pdb_template_fn
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
        print('En scripts estpy mandando :', pdb_path_aa, cg_mapping_strategy)
        prior_nls = samples.get_prior_nls(
            prior_builders, save_nls=True, save_dir=save_dir, prior_tag=prior_tag, name = cg_mapping_strategy, pdb_file = pdb_path_aa, map_file = map_path_aa, itp_file_cg = itp_path_cg,
        )


if __name__ == "__main__":
    print("Start gen_input_data.py: {}".format(ctime()))

    CLI([process_raw_dataset, build_neighborlists])

    print("Finish gen_input_data.py: {}".format(ctime()))
