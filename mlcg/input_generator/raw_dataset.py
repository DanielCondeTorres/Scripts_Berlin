import mdtraj as md
import pickle

from typing import List, Dict, Tuple, Optional, Union, Any
from copy import deepcopy
import numpy as np
import torch
import warnings
import os
from importlib import import_module

from torch_geometric.data.collate import collate

import sys
sys.path.append('/mnt/lustre/hsm/nlsas/notape/home/usc/qo/supepmem/DANIEL/75_per_leaflet/simulacion_openmm/NVT_TODAS/PLAY_GROUND/mlcg-playground-feat-raw_data_processing_and_prior_fit/input_generator')

from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.data.atomic_data import AtomicData

from .utils import (
    map_cg_topology,
    slice_coord_forces,
    get_terminal_atoms,
    get_edges_and_orders,
    create_martini_cg,
)
from .prior_gen import PriorBuilder


def get_strides(n_structure: int, batch_size: int):
    """
    Helper function to stride batched data
    """
    n_elem, remain = np.divmod(n_structure, batch_size)
    assert remain > -1, f"remain: {remain}"
    if remain == 0:
        batches = np.zeros(n_elem + 1)
        batches[1:] = batch_size
    else:
        batches = np.zeros(n_elem + 2)
        batches[1:-1] = batch_size
        batches[-1] = remain
    strides = np.cumsum(batches, dtype=int)
    strides = np.vstack([strides[:-1], strides[1:]]).T
    return strides


class CGDataBatch:
    """
    Splits input CG data into batches for further memory-efficient processing

    Attributes
    ----------
    batch_size:
        Number of frames to use in each batch
    stride:
        Integer by which to stride frames
    concat_forces:
        Boolean indicating whether forces should be added to batch
    cg_coords:
        Coarse grained coordinates
    cg_forces:
        Coarse grained forces
    cg_embeds:
        Atom embeddings
    cg_prior_nls:
        Dictionary of prior neighbour list
    """

    def __init__(
        self,
        cg_coords: np.ndarray,
        cg_forces: np.ndarray,
        cg_embeds: np.ndarray,
        cg_prior_nls: Dict,
        batch_size: int,
        stride: int,
        concat_forces: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.stride = stride
        self.concat_forces = concat_forces
        self.cg_coords = torch.from_numpy(cg_coords[::stride])
        self.cg_forces = torch.from_numpy(cg_forces[::stride])
        print('Embedings: ', cg_embeds)
        self.cg_embeds = torch.from_numpy(cg_embeds)
        self.cg_prior_nls = cg_prior_nls

        self.n_structure = self.cg_coords.shape[0]
        if batch_size > self.n_structure:
            self.batch_size = self.n_structure

        self.strides = get_strides(self.n_structure, self.batch_size)
        self.n_elem = self.strides.shape[0]

    def __len__(self):
        return self.n_elem

    def __getitem__(self, idx):
        """
        Returns list of AtomicData objects for indexed batch
        """
        st, nd = self.strides[idx]
        data_list = []
        # TODO: build the collated AtomicData by hand to avoid copy/concat ops
        for ii in range(st, nd):
            dd = dict(
                pos=self.cg_coords[ii],
                atom_types=self.cg_embeds,
                masses=None,
                neighborlist=self.cg_prior_nls,
            )
            if self.concat_forces:
                dd["forces"] = self.cg_forces[ii]

            data = AtomicData.from_points(**dd)
            data_list.append(data)
        datas, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )
        return datas


class SampleCollection:
    """
    Input generation object for loading, manupulating, and saving training data samples.

    Attributes
    ----------
    name:
        String associated with atomistic trajectory output.
    tag:
        String to identify dataset in output files.
    pdb_fn:
        File location of atomistic structure to be used for topology.
    """

    def __init__(
        self,
        name: str,
        tag: str,
    ) -> None:
        self.name = name
        self.tag = tag

    def apply_cg_mapping(
        self,
        cg_atoms: List[str],
        embedding_function: Union[str,Optional[Any]],
        embedding_dict: Union[str,Optional[Any]],
        skip_residues: Optional[List[str]] = None,

        pdb_file: str = None,
        map_file: str = None,
        itp_file_cg: str = None,
    ):
        """
        Applies mapping function to atomistic topology to obtain CG representation.

        Parameters
        ----------
        cg_atoms:
            List of atom names to preserve in CG representation.
        embedding_function:
            Name of function (should be defined in embedding_maps) to apply CG mapping.
        embedding_dict:
            Name of dictionary (should be defined in embedding_maps) to define embeddings of CG beads.
        skip_residues: (Optional)
            List of residue names to skip (can be used to skip terminal caps, for example).
            Currently, can only be used to skip all residues with given name.
        """
        if embedding_function == 'cg_mapping':
            config_map_matrix,force_map_matrix, new_coords, cg_top, system = create_martini_cg(pdb_file, map_file, itp_file_cg)
            print('TOPOLOGIA: ',cg_top)
            cg_traj = md.Trajectory(xyz = new_coords, topology = cg_top)
            self.cg_traj = cg_traj
            cg_df,bonds = cg_top.to_dataframe()
            self.top_dataframe = cg_df
            print('DATA FRAME_ ',cg_df)
            print('BONDS: ',bonds)
            for bond in bonds: 
                print('Enlace entre: ', bond)
            all_atom_indices = cg_top.select('all')
            cg_atom_idx = all_atom_indices
            #cg_atom_idx = cg_df.index.tolist()
            self.cg_map = config_map_matrix
            self.N_term = None
            self.C_term = None
            cg_traj.save("/mnt/lustre/hsm/nlsas/notape/home/usc/qo/supepmem/DANIEL/75_per_leaflet/simulacion_openmm/NVT_TODAS/PLAY_GROUND/mlcg-playground-feat-raw_data_processing_and_prior_fit/examples/martinini.gro")
            self.cg_atom_indices = cg_atom_idx
            element_map = {
                    "N": 1,
                    "P": 2,
                    "VS": 3,
                    "C": 4,
                    "B": 5,
                    }
            #element_map =  {"NC31":1,"PO42":2,"GL13":3,"GL24":4,"C1A5":5,"D2A6":6,"C3A7":7,"C4A8":8,"C1B9":9,"C2B10":10,"C3B11":11,"C4B12":12}
            print('ESTO ES: ',cg_df["element"].unique())
                #poner element abajo
            cg_df["type"] = cg_df["element"].map(element_map)#;cg_df["type"] = cg_df["type"].astype(int)    #ojo name o element depende
            print('VALORES UNICOAS: ',cg_df["type"].unique())
            print('NEW_DF_ _:',cg_df["type"])
            cg_df["type"] = cg_df["type"].astype(int)
            #cg_df["type"]  =1
            self.cg_dataframe = cg_df
            self.terminal_embedings = False
            #Added to a.sh, a ver si va bien si no borrar:
            #cg_atom_idx = cg_df.index.values.tolist()
            self.cg_atom_indices = cg_atom_idx
            cg_df.index = [i for i in range(len(cg_df.index))]
            cg_df.serial = [i + 1 for i in range(len(cg_df.index))]
            self.cg_dataframe = cg_df
            self.system = system
            print('OUT SIN IMPRIMIR')
            print('OUT indexs',system.xyz)
        else:
            print('EM BD:',embedding_dict, embedding_function)
            if isinstance(embedding_dict, str):
                self.embedding_dict = eval(embedding_dict)
            print('NO SALE EM BD')
            self.top_dataframe = self.top_dataframe.apply(
                                        map_cg_topology,
                                        axis=1,
                                        cg_atoms=cg_atoms,
                                        embedding_function=embedding_function,
                                        skip_residues=skip_residues,
                                        )
            cg_df = deepcopy(self.top_dataframe.loc[self.top_dataframe["mapped"] == True])
            print('CG DFFFFFFFFFFFFFFFFFFFFFFFFf:', cg_df)
            cg_atom_idx = cg_df.index.values.tolist()
            print('CG INDEX: ',cg_atom_idx)
            self.cg_atom_indices = cg_atom_idx
            cg_df.index = [i for i in range(len(cg_df.index))]
            cg_df.serial = [i + 1 for i in range(len(cg_df.index))]
            self.cg_dataframe = cg_df
            cg_map = np.zeros((len(cg_atom_idx), self.input_traj.n_atoms))
            cg_map[[i for i in range(len(cg_atom_idx))], cg_atom_idx] = 1
            if not all([sum(row) == 1 for row in cg_map]):
                warnings.warn("WARNING: Slice mapping matrix is not unique.")
            if not all([row.tolist().count(1) == 1 for row in cg_map]):
                warnings.warn("WARNING: Slice mapping matrix is not linear.")

            self.cg_map = cg_map
            # save N_term and C_term as None, to be overwritten if terminal embeddings used
            self.N_term = None
            self.C_term = None
            
            #Added
            cg_xyz = self.input_traj.atom_slice(self.cg_atom_indices).xyz
            self.cg_traj = md.Trajectory(cg_xyz, md.Topology.from_dataframe(self.cg_dataframe))
            self.terminal_embedings = True

    def add_terminal_embeddings(
        self, N_term: Union[str, None] = "N", C_term: Union[str, None] = "C", terminal_embedings: bool = True):
        """
        Adds separate embedding to terminals (do not need to be defined in original embedding_dict).

        Parameters
        ----------
        N_term:
            Atom of N-terminus to which N_term embedding will be assigned.
        C_term:
            Atom of C-terminus to which C_term embedding will be assigned.

        Either of N_term and/or C_term can be None; in this case only one (or no) terminal embedding(s) will be assigned.
        """
        terminal_embedings =  self.terminal_embedings
        print('EMBEDINGS : ',terminal_embedings)
        if terminal_embedings == True:
            df_cg = self.cg_dataframe
            # proteins with multiple chains will have multiple N- and C-termini
            self.N_term = N_term
            self.C_term = C_term
            if N_term != None:
                if "N_term" not in self.embedding_dict:
                    self.embedding_dict["N_term"] = max(self.embedding_dict.values()) + 1
                N_term_atom = df_cg.loc[
                                (df_cg["resSeq"] == df_cg["resSeq"].min()) & (df_cg["name"] == N_term)
                                ].index
                for idx in N_term_atom:
                    self.cg_dataframe.at[idx, "type"] = self.embedding_dict["N_term"]

            if C_term != None:
                if "C_term" not in self.embedding_dict:
                    self.embedding_dict["C_term"] = max(self.embedding_dict.values()) + 1
                C_term_atom = df_cg.loc[
                                (df_cg["resSeq"] == df_cg["resSeq"].max()) & (df_cg["name"] == C_term)
                                    ].index
                for idx in C_term_atom:
                    self.cg_dataframe.at[idx, "type"] = self.embedding_dict["C_term"]
        else:
            print('No embedings')

    def process_coords_forces(
        self,
        coords: np.ndarray,
        forces: np.ndarray,
        mapping: str = "slice_aggregate",
        force_stride: int = 100,
        
        pdb_file = None,
        map_file = None,
        itp_file_cg = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps coordinates and forces to CG resolution

        Parameters
        ----------
        coords: [n_frames, n_atoms, 3]
            Atomistic coordinates
        forces: [n_frames, n_atoms, 3]
            Atomistic forces
        mapping:
            Mapping scheme to be used, must be either 'slice_aggregate' or 'slice_optimize' ;) or the new one (;.
        force_stride:
            Striding to use for force projection results

        Returns
        -------
        Tuple of np.ndarray's for coarse grained coordinates and forces
        """
        print('Process Forces')
        if coords.shape != forces.shape:
            warnings.warn(
                "Cannot process coordinates and forces: mismatch between array shapes."
            )
            return
        else:
            cg_coords, cg_forces, new_top = slice_coord_forces(
                coords, forces, self.cg_map, mapping, force_stride,  pdb_file, map_file, itp_file_cg
            )

            self.cg_coords = cg_coords
            self.cg_forces = cg_forces
            self.cg_top = new_top
            print('Mi topologia: ', self.cg_top)
            return cg_coords, cg_forces, new_top

    def save_cg_output(
        self,
        save_dir: str,
        save_coord_force: bool = True,
        cg_coords: Union[np.ndarray, None] = None,
        cg_forces: Union[np.ndarray, None] = None,
        
        mapping_str:str = 'cg_mapping'
    ):
        """
        Saves processed CG data.

        Parameters
        ----------
        save_dir:
            Path of directory to which output will be saved.
        save_coord_force:
            Whether coordinates and forces should also be saved.
        cg_coords:
            CG coordinates; if None, will check whether these are saved as attribute.
        cg_forces:
            CG forces; if None, will check whether these are saved as an object attribute.
        """
        print('Guardando archivos bip bip bip...')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        if not hasattr(self, "cg_atom_indices"):
            print("CG mapping must be applied before outputs can be saved.")
            return
        save_templ = os.path.join(save_dir, f"{self.tag}{self.name}")

        #if mapping_str == 'sa':
        #    config_map_matrix,force_map_matrix, new_coords, cg_top, cg_traj = create_martini_cg(pdb_file, map_file, itp_file_cg)
        #else:
        #cg_xyz = self.input_traj.atom_slice(self.cg_atom_indices).xyz
        #cg_traj = md.Trajectory(cg_xyz, md.Topology.from_dataframe(self.cg_dataframe))
        print('CG TRAJ: ',self.cg_traj)
        cg_traj = self.cg_traj


        cg_traj.save_pdb(f"{save_templ}_cg_structure.pdb")

        embeds = np.array(self.cg_dataframe["type"].to_list())
        np.save(f"{save_templ}_cg_embeds.npy", embeds)
        print('Guardado? bip bip bip...')
        if save_coord_force:
            if cg_coords is None:
                if not hasattr(self, "cg_coords"):
                    print(
                        "No coordinates found; only CG structure, embeddings and loaded forces will be saved."
                    )
                else:
                    np.save(f"{save_templ}_cg_coords.npy", self.cg_coords)
            else:
                np.save(f"{save_templ}_cg_coords.npy", cg_coords)

            if cg_forces is None:
                if not hasattr(self, "cg_forces"):
                    print(
                        "No forces found;  only CG structure, embeddings, and loaded coordinates will be saved."
                    )
                else:
                    np.save(f"{save_templ}_cg_forces.npy", self.cg_forces)
            else:
                np.save(f"{save_templ}_cg_forces.npy", cg_forces)

    def get_prior_nls(
        self, prior_builders: List[PriorBuilder], save_nls: bool = True, name: str = None, pdb_file:str = None, map_file:str = None, itp_file_cg:str = None, **kwargs
    ) -> Dict:
        """
        Creates neighbourlists for all prior terms specified in the prior_dict.

        Parameters
        ----------
        prior_builders:
            List of PriorBuilder objects and their corresponding parameters.
            Input config file must minimally contain the following information for
            each builder:
                class_path: class specifying PriorBuilder object implemented in `prior_gen.py`
                init_args:
                    name: string specifying type as one of 'bonds', 'angles', 'dihedrals', 'non_bonded'
                    nl_builder: name of class implemented in `prior_nls.py` which will be used to collect
                                atom groups associated with the prior term.
        save_nls:
            If true, will save an output of the molecule's neighbourlist.
        kwargs:
            save_dir:
                If save_nls = True, the neighbourlist will be saved to this directory.
            prior_tag:
                String identifying the specific combination of prior terms.

        Returns
        -------
        Dictionary of prior terms with specific index mapping for the given molecule.

        Example
        -------
        To build neighbour lists for a system with priors for bonds, angles, nonbonded pairs, and phi and
        psi dihedral angles:

            - class_path: input_generator.Bonds
              init_args:
                name: bonds
                separate_termini: true
                nl_builder: input_generator.StandardBonds
            - class_path: input_generator.Angles
              init_args:
                name: angles
                separate_termini: true
                nl_builder: input_generator.StandardAngles
            - class_path: input_generator.NonBonded
              init_args:
                name: non_bonded
                min_pair: 6
                res_exclusion: 1
                separate_termini: false
                nl_builder: input_generator.Non_Bonded
            - class_path: input_generator.Dihedrals
              init_args:
                name: phi
                nl_builder: input_generator.Phi
            - class_path: input_generator.Dihedrals
              init_args:
                name: psi
                nl_builder: input_generator.Psi
        """
        print('PREVIO CREATE MARTINI CG de raw_dataset',name)
        for prior_builder in prior_builders:
            if getattr(prior_builder, "separate_termini", False):
                prior_builder = get_terminal_atoms(
                    prior_builder,
                    cg_dataframe=self.cg_dataframe,
                    N_term=self.N_term,
                    C_term=self.C_term,
                )

        # get atom groups for edges and orders for all prior terms
        print('Nome: ',name)
        if name == 'cg_mapping':
            print('HOSTIA PUTA: ')
            print(pdb_file)
            config_map_matrix,force_map_matrix, new_coords, cg_top, system = create_martini_cg(pdb_file, map_file, itp_file_cg)
        else:
            cg_top = self.input_traj.atom_slice(self.cg_atom_indices).topology
        

        all_edges_and_orders = get_edges_and_orders(
            prior_builders,
            topology=cg_top,
        )
        tags = [x[0] for x in all_edges_and_orders]
        orders = [x[1] for x in all_edges_and_orders]
        edges = [
            (
                torch.tensor(x[2]).type(torch.LongTensor)
                if isinstance(x[2], np.ndarray)
                else x[2].type(torch.LongTensor)
            )
            for x in all_edges_and_orders
        ]
        prior_nls = {}
        for tag, order, edge in zip(tags, orders, edges):
            nl = make_neighbor_list(tag, order, edge)
            prior_nls[tag] = nl

        if save_nls:
            ofile = os.path.join(
                kwargs["save_dir"],
                f"{self.tag}{self.name}_prior_nls_{kwargs['prior_tag']}.pkl",
            )
            with open(ofile, "wb") as pfile:
                pickle.dump(prior_nls, pfile)

        return prior_nls

    def load_cg_output(self, save_dir: str, prior_tag: str = "") -> Tuple:
        """
        Loads all cg data produced by `save_cg_output` and `get_prior_nls`

        Parameters
        ----------
        save_dir:
            Location of saved cg data
        prior_tag:
            String identifying the specific combination of prior terms

        Returns
        -------
        Tuple of np.ndarrays containing coarse grained coordinates, forces, embeddings,
        structure, and prior neighbour list
        """
        save_templ = os.path.join(save_dir, f"{self.tag}{self.name}")
        cg_coords = np.load(f"{save_templ}_cg_coords.npy")
        cg_forces = np.load(f"{save_templ}_cg_forces.npy")
        cg_embeds = np.load(f"{save_templ}_cg_embeds.npy")
        cg_pdb = md.load(f"{save_templ}_cg_structure.pdb")
        # load NLs
        ofile = os.path.join(
            save_dir, f"{self.tag}{self.name}_prior_nls_{prior_tag}.pkl"
        )
        with open(ofile, "rb") as f:
            cg_prior_nls = pickle.load(f)
        return cg_coords, cg_forces, cg_embeds, cg_pdb, cg_prior_nls

    def load_cg_output_into_batches(
        self,
        save_dir: str,
        prior_tag: str,
        batch_size: int,
        stride: int,
    ):
        """
        Loads saved CG data nad splits these into batches for further processing

        Parameters
        ----------
        save_dir:
            Location of saved cg data
        prior_tag:
            String identifying the specific combination of prior terms
        batch_size:
            Number of frames to use in each batch
        stride:
            Integer by which to stride frames

        Returns
        -------
        Loaded CG data split into list of batches
        """
        cg_coords, cg_forces, cg_embeds, cg_pdb, cg_prior_nls = self.load_cg_output(
            save_dir, prior_tag
        )
        batch_list = CGDataBatch(
            cg_coords, cg_forces, cg_embeds, cg_prior_nls, batch_size, stride
        )
        return batch_list


class RawDataset:
    """
    Generates a list of data samples for a specified dataset

    Attributes
    ----------
    dataset_name:
        Name given to dataset
    names:
        List of sample names
    tag:
        Label given to all output files produced from dataset
    dataset:
        List of SampleCollection objects for all samples in dataset
    """

    def __init__(self, dataset_name: str, names: List[str], tag: str) -> None:
        self.dataset_name = dataset_name
        self.names = names
        self.tag = tag
        self.dataset = []

        for name in names:
            data_samples = SampleCollection(
                name=name,
                tag=tag,
            )
            self.dataset.append(data_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class SimInput:
    """
    Generates a list of samples from pdb structures to be used in simulation

    Attributes
    ----------
    dataset_name:
        Name given to dataset
    tag:
        Label given to all output files produced from dataset
    pdb_fns:
        List of pdb filenames from which samples will be generated
    dataset:
        List of SampleCollection objects for all structures
    """

    def __init__(self, dataset_name: str, tag: str, pdb_fns: List[str]) -> None:
        self.dataset_name = dataset_name
        self.names = [fn[:-4] for fn in pdb_fns]
        self.dataset = []

        for name in self.names:
            data_samples = SampleCollection(
                name=name,
                tag=tag,
            )
            self.dataset.append(data_samples)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
