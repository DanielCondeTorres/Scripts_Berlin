import os.path as osp
import sys

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

import mdtraj as md
import numpy as np
import torch
sys.path.append('/mnt/lustre/hsm/nlsas/notape/home/usc/qo/supepmem/DANIEL/75_per_leaflet/simulacion_openmm/NVT_TODAS/PLAY_GROUND/mlcg-playground-feat-raw_data_processing_and_prior_fit/input_generator')

from mlcg.data.atomic_data import AtomicData
from mlcg.datasets.utils import remove_baseline_forces

from input_generator.raw_dataset import *

from tqdm import tqdm

from time import ctime

from typing import Dict, List, Union, Callable
from jsonargparse import CLI


def produce_delta_forces(
    dataset_name: str,
    names: List[str],
    tag: str,
    save_dir: str,
    prior_tag: str,
    prior_fn: str,
    device: str,
    batch_size: int,
):
    """
    Removes prior energy terms from input forces to produce delta force input
    for training

    Parameters
    ----------
    dataset_name : str
        Name given to specific dataset
    names : List[str]
        List of sample names
    tag : str
        Label given to all output files produced from dataset
    save_dir : str
        Path to directory from which input will be loaded and to which output will be saved
    prior_tag : str
        String identifying the specific combination of prior terms
    prior_fn : str
        Path to filename in which prior model is saved
    device: str
        Device on which to run delta force calculations
    batch_size : int
        Number of frames to take per batch
    """

    prior_model = torch.load(open(prior_fn, "rb")).models.to(device)
    dataset = RawDataset(dataset_name, names, tag)
    for samples in tqdm(dataset, f"Processing delta forces for {dataset_name} dataset..."):
        coords, forces, embeds, pdb, prior_nls = samples.load_cg_output(save_dir=save_dir, prior_tag=prior_tag)

        num_frames = coords.shape[0]
        delta_forces = []
        for i in range(0, num_frames, batch_size):
            sub_data_list = []
            for j in range(batch_size):
                if i + j < len(coords):
                    data = AtomicData.from_points(
                                                pos=torch.tensor(coords[i + j]),
                                                forces=torch.tensor(forces[i + j]),
                                                atom_types=torch.tensor(embeds),
                                                masses=None,
                                                neighborlist=prior_nls,
                                                )
                    sub_data_list.append(data.to(device))
            sub_data_list = tuple(sub_data_list)
            _ = remove_baseline_forces(sub_data_list,prior_model)
            for j in range(batch_size):
                if j < len(sub_data_list):
                    delta_force = sub_data_list[j].forces.detach().cpu()
                    delta_forces.append(delta_force.numpy())

        np.save(os.path.join(save_dir, f"{tag}{samples.name}_{prior_tag}_delta_forces.npy"),np.concatenate(delta_forces, axis=0).reshape(*coords.shape))


if __name__ == "__main__":
    print("Start produce_delta_forces.py: {}".format(ctime()))

    CLI([produce_delta_forces])

    print("Finish produce_delta_forces.py: {}".format(ctime()))
