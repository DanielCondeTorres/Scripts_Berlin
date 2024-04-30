import torch

import mdtraj as md
from typing import Any, List, Union, Tuple, Optional
import numpy as np

from mlcg.geometry._symmetrize import _symmetrise_distance_interaction
from networkx.algorithms.shortest_paths.unweighted import (
    bidirectional_shortest_path,
)
import networkx as nx
from mlcg.geometry.topology import (
    Topology,
    get_connectivity_matrix,
    get_n_paths,
)

from .utils import get_dihedral_groups, split_bulk_termini
from .embedding_maps import all_residues


class StandardBonds:
    """
    Pairwise interactions corresponding to physically bonded atoms

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        If `separate_termini` is False, only bonds are returned, otherwise
        atom groups are split based on interactions between only bulk atoms or
        interactions with atoms in terminal residues.
    """

    nl_names = ["n_term_bonds", "bulk_bonds", "c_term_bonds", "bonds"]

    def __call__(
        self, topology: md.Topology, separate_termini: bool = False, **kwargs
    ) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        """
        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created.
        separate_termini:
            Whether atom groups should be split between bulk interactions and those involving atoms
            in terminal residues
        """
        mlcg_top = Topology.from_mdtraj(topology)
        conn_mat = get_connectivity_matrix(mlcg_top).numpy()
        bond_edges = get_n_paths(conn_mat, n=2).numpy()
        if separate_termini:
            n_term_atoms, c_term_atoms = kwargs["n_term_atoms"], kwargs["c_term_atoms"]
            n_term_bonds, c_term_bonds, bulk_bonds = split_bulk_termini(
                n_term_atoms, c_term_atoms, bond_edges
            )

            if len(bulk_bonds) == 0:
                bonds = [
                    ("n_term_bonds", 2, n_term_bonds),
                    ("bulk_bonds", 2, torch.tensor([]).reshape(2, 0)),
                    ("c_term_bonds", 2, c_term_bonds),
                ]

            elif len(n_term_bonds) == 0 or len(c_term_bonds) == 0:
                bonds = [
                    ("n_term_bonds", 2, torch.tensor([]).reshape(2, 0)),
                    ("bulk_bonds", 2, bulk_bonds),
                    ("c_term_bonds", 2, torch.tensor([]).reshape(2, 0)),
                ]
            else:
                bonds = [
                    ("n_term_bonds", 2, n_term_bonds),
                    ("bulk_bonds", 2, bulk_bonds),
                    ("c_term_bonds", 2, c_term_bonds),
                ]

        else:
            bonds = ("bonds", 2, bond_edges)
        print('ENlazaditos: ', bonds)
        return bonds

    def get_fit_kwargs(self, nl_name):
        return {}


class StandardAngles:
    """
    Interactions corresponding to angles formed between three physically bonded atoms

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        If `separate_termini` is False, only bonds are returned, otherwise
        atom groups are split based on interactions between only bulk atoms or
        interactions with atoms in terminal residues.
    """

    nl_names = ["n_term_angles", "bulk_angles", "c_term_angles", "angles"]

    def __call__(
        self, topology: md.Topology, separate_termini: bool = False, **kwargs
    ) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        """
        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created.
        separate_termini:
            Whether atom groups should be split between bulk interactions and those involving atoms
            in terminal residues
        """
        mlcg_top = Topology.from_mdtraj(topology)
        conn_mat = get_connectivity_matrix(mlcg_top).numpy()
        angle_edges = get_n_paths(conn_mat, n=3).numpy()
        print('MLCG TOP:',mlcg_top)
        print('ANGLES EDGES: ',angle_edges)
        if separate_termini:
            n_term_atoms, c_term_atoms = kwargs["n_term_atoms"], kwargs["c_term_atoms"]
            n_term_angles, c_term_angles, bulk_angles = split_bulk_termini(
                n_term_atoms, c_term_atoms, angle_edges
            )
            if len(bulk_angles) == 0:
                angles = [
                    ("n_term_angles", 3, n_term_angles),
                    ("bulk_angles", 3, torch.tensor([]).reshape(3, 0)),
                    ("c_term_angles", 3, c_term_angles),
                ]

            elif len(n_term_angles) == 0 or len(c_term_angles) == 0:
                angles = [
                    ("n_term_angles", 3, torch.tensor([]).reshape(3, 0)),
                    ("bulk_angles", 3, bulk_angles),
                    ("c_term_angles", 3, torch.tensor([]).reshape(3, 0)),
                ]
            else:
                angles = [
                    ("n_term_angles", 3, n_term_angles),
                    ("bulk_angles", 3, bulk_angles),
                    ("c_term_angles", 3, c_term_angles),
                ]
        else:
            angles = ("angles", 3, angle_edges)
        print('ANGULOS : ',angle_edges)
        return angles

    def get_fit_kwargs(self, nl_name):
        return {}


class Non_Bonded:
    """
    Pairwise interactions corresponding to nonbonded atoms

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        If `separate_termini` is False, only bonds are returned, otherwise
        atom groups are split based on interactions between only bulk atoms or
        interactions with atoms in terminal residues.
    """

    nl_names = ["n_term_nonbonded", "bulk_nonbonded", "c_term_nonbonded", "non_bonded"]

    def __call__(
        self,
        topology: md.Topology,
        bond_edges: Union[np.array, List, None] = None,
        angle_edges: Union[np.array, List, None] = None,
        min_pair: int = 6,
        res_exclusion: int = 1,
        separate_termini: bool = False,
        **kwargs,
    ) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        """
        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created.
        bond_edges:
            All edges associated with bond atom groups already defined
        angle_edges:
            All edges associated with angle atom groups already defined
        min_pair:
            Minimum number of bond edges between two atoms in order to be considered
            a member of the non-bonded set
        res_exclusion:
            If supplied, pairs within res_exclusion residues of each other are removed
            from the non-bonded set
        separate_termini:
            Whether atom groups should be split between bulk interactions and those involving atoms
            in terminal residues
        """
        mlcg_top = Topology.from_mdtraj(topology)
        print('MLCG TOP:',mlcg_top)
        fully_connected_edges = _symmetrise_distance_interaction(
            mlcg_top.fully_connected2torch()
        ).numpy()
        conn_mat = get_connectivity_matrix(mlcg_top).numpy()
        graph = nx.Graph(conn_mat)

        pairs_parsed = np.array(
            [
                p
                for p in fully_connected_edges.T
                if (
                    abs(
                        topology.atom(p[0]).residue.index
                        - topology.atom(p[1]).residue.index
                    )
                    >= res_exclusion
                )
                and (
                    graph.has_edge(p[0], p[1]) == False
                    or len(bidirectional_shortest_path(graph, p[0], p[1])) >= min_pair
                )
                and not np.all(bond_edges == p[:, None], axis=0).any()
                and not np.all(angle_edges[[0, 2], :] == p[:, None], axis=0).any()
            ]
        )

        non_bonded_edges = torch.tensor(pairs_parsed.T)
        non_bonded_edges = torch.unique(
            _symmetrise_distance_interaction(non_bonded_edges), dim=1
        ).numpy()

        if separate_termini:
            if "use_terminal_res" in kwargs and kwargs["use_terminal_res"] == True:
                n_atoms = kwargs["n_term_atoms"]
                c_atoms = kwargs["c_term_atoms"]
            else:
                n_atoms = kwargs["n_atoms"]
                c_atoms = kwargs["c_atoms"]
            n_term_nonbonded, c_term_nonbonded, bulk_nonbonded = split_bulk_termini(
                n_atoms, c_atoms, non_bonded_edges
            )
            return [
                ("n_term_nonbonded", 2, n_term_nonbonded),
                ("bulk_nonbonded", 2, bulk_nonbonded),
                ("c_term_nonbonded", 2, c_term_nonbonded),
            ]
        else:
            return ("non_bonded", 2, non_bonded_edges)

    def get_fit_kwargs(self, nl_name):
        return {}


class Phi:
    """
    Phi (proper) dihedral angle formed by the following atoms:
    C_{n-1} - N_{n} - CA_{n} - C_{n}
    where n represents the amino acid for which the angle is defined

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        Atom groups of phi angles of each amino acid are recorded separately
    """

    nl_names = [f"{res}_phi" for res in all_residues]

    def __call__(
        self, topology: md.Topology, **kwargs
    ) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        dihedral_dict = get_dihedral_groups(
            topology,
            atoms_needed=["C", "N", "CA", "C"],
            offset=[-1.0, 0.0, 0.0, 0.0],
            tag="_phi",
        )
        dihedrals = []
        for res in all_residues:
            dihedral_tag = f"{res}_phi"
            if dihedral_tag in dihedral_dict:
                atom_groups = np.array(dihedral_dict[dihedral_tag])
                dihedrals.append((dihedral_tag, 4, torch.tensor(atom_groups).T))
            else:
                dihedrals.append((dihedral_tag, 4, torch.tensor([]).reshape(4, 0)))
        return dihedrals

    def get_fit_kwargs(self, nl_name):
        if nl_name == "PRO_phi":
            return {"n_degs": 1, "constrain_deg": 1}
        else:
            return {"n_degs": 3, "constrain_deg": 3}


class Psi:
    """
    Psi (proper) dihedral angle formed by the following atoms:
    N_{n} - CA_{n} - C_{n} - N_{n+1}
    where n represents the amino acid for which the angle is defined

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        Atom groups of psi angles of each amino acid are recorded separately
    """

    nl_names = [f"{res}_psi" for res in all_residues]

    def __call__(
        self, topology: md.Topology, **kwargs
    ) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        dihedral_dict = get_dihedral_groups(
            topology,
            atoms_needed=["N", "CA", "C", "N"],
            offset=[0.0, 0.0, 0.0, 1.0],
            tag="_psi",
        )
        dihedrals = []
        for res in all_residues:
            dihedral_tag = f"{res}_psi"
            if dihedral_tag in dihedral_dict:
                atom_groups = np.array(dihedral_dict[dihedral_tag])
                dihedrals.append((dihedral_tag, 4, torch.tensor(atom_groups).T))
            else:
                dihedrals.append((dihedral_tag, 4, torch.tensor([]).reshape(4, 0)))
        return dihedrals

    def get_fit_kwargs(self, nl_name):
        return {"n_degs": 3, "constrain_deg": 3}


class Omega:
    """
    Omega (proper) dihedral angle formed by the following atoms:
    CA_{n-1} - C_{n-1} - N_{n} - C_{n}
    where n represents the amino acid for which the angle is defined

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        Atom groups of omega angles are recorded separately only for proline
    """

    nl_names = ["pro_omega", "non_pro_omega"]
    replace_gly_ca_stats = True

    def __call__(
        self, topology: md.Topology, **kwargs
    ) -> List[Tuple[str, int, torch.Tensor]]:
        dihedral_dict = get_dihedral_groups(
            topology,
            atoms_needed=["CA", "C", "N", "CA"],
            offset=[-1, -1, 0, 0],
            tag="_omega",
        )
        pro_omega = []
        non_pro_omega = []
        for dihedral_tag in dihedral_dict.keys():
            atom_groups = np.array(dihedral_dict[dihedral_tag])
            if dihedral_tag == "PRO_omega":
                pro_omega.extend(atom_groups)
            else:
                non_pro_omega.extend(atom_groups)
        dihedrals = []
        for dihedral in ["pro_omega", "non_pro_omega"]:
            if len(eval(dihedral)) == 0:
                dihedrals.append((dihedral, 4, torch.tensor([]).reshape(4, 0)))
            else:
                dihedrals.append(
                    (dihedral, 4, torch.tensor(np.array(eval(dihedral))).T)
                )
        return dihedrals

    def get_fit_kwargs(self, nl_name):
        if nl_name == "pro_omega":
            return {"n_degs": 2, "constrain_deg": 2}
        else:
            return {"n_degs": 1, "constrain_deg": 1}


class Gamma1:
    """
    Improper dihedral angle formed by the following atoms:
    N_{n} - CB_{n} - C_{n} - CA_{n}
    where n represents the amino acid for which the angle is defined;
    gamma_1 angle is measured between the plane formed by the first, third, and
    fourth atom and the vector from the first to second atom.

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        Atom groups of gamma_1 angles are not separaeted by amino acid type
    """

    nl_names = ["gamma_1"]

    def __call__(
        self, topology: md.Topology, **kwargs
    ) -> Tuple[str, int, torch.Tensor]:
        dihedral_dict = get_dihedral_groups(
            topology,
            atoms_needed=["N", "CB", "C", "CA"],
            offset=[0, 0, 0, 0],
            tag="_gamma_1",
        )
        atom_groups = []
        for res in dihedral_dict:
            atom_groups.extend(dihedral_dict[res])
        if len(atom_groups) == 0:
            dihedrals = ("gamma_1", 4, torch.tensor([]).reshape(4, 0))
        else:
            dihedrals = ("gamma_1", 4, torch.tensor(np.array(atom_groups)).T)
        return dihedrals

    def get_fit_kwargs(self, nl_name):
        return {"n_degs": 1, "constrain_deg": 1}


class Gamma2:
    """
    Improper dihedral angle formed by the following atoms:
    CA_{n} - O_{n} - N_{n+1} - C_{n}
    where n represents the amino acid for which the angle is defined;
    gamma_2 angle is measured between the plane formed by the first, third, and
    fourth atom and the vector from the first to second atom.

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        Atom groups of gamma_2 angles are not separaeted by amino acid type
    """

    nl_names = ["gamma_2"]

    def __call__(
        self, topology: md.Topology, **kwargs
    ) -> Tuple[str, int, torch.Tensor]:
        dihedral_dict = get_dihedral_groups(
            topology,
            atoms_needed=["CA", "O", "N", "C"],
            offset=[0, 0, 1, 0],
            tag="_gamma_2",
        )
        atom_groups = []
        for res in dihedral_dict:
            atom_groups.extend(dihedral_dict[res])
        if len(atom_groups) == 0:
            dihedrals = ("gamma_2", 4, torch.tensor([]).reshape(4, 0))
        else:
            dihedrals = ("gamma_2", 4, torch.tensor(np.array(atom_groups)).T)
        return dihedrals

    def get_fit_kwargs(self, nl_name):
        return {"n_degs": 1, "constrain_deg": 1}



class StandardDihedrals:
    """
    Interactions corresponding to dihedrals formed between four physically bonded atoms

    Attributes
    ----------
    nl_names
        All possible outputs of bonded neighbourlist;
        If `separate_termini` is False, only bonds are returned, otherwise
        atom groups are split based on interactions between only bulk atoms or
        interactions with atoms in terminal residues.
    """

    nl_names = ["n_term_dihedrals", "bulk_dihedrals", "c_term_dihedrals", "dihedrals"]

    def __call__(
        self, topology: md.Topology, separate_termini: bool = False, **kwargs
    ) -> Union[List[Tuple[str, int, torch.Tensor]], Tuple[str, int, torch.Tensor]]:
        """
        Parameters
        ----------
        topology:
            MDTraj topology object from which atom groups defining each prior term will be created.
        separate_termini:
            Whether atom groups should be split between bulk interactions and those involving atoms
            in terminal residues
        """
        mlcg_top = Topology.from_mdtraj(topology)
        conn_mat = get_connectivity_matrix(mlcg_top).numpy()
        dihedral_edges = get_n_paths(conn_mat, n=4).numpy()
        print('MLCG TOP:',mlcg_top)
        print('DIHEDRAL EDGES: ', dihedral_edges)
        if separate_termini:
            n_term_atoms, c_term_atoms = kwargs["n_term_atoms"], kwargs["c_term_atoms"]
            n_term_dihedrals, c_term_dihedrals, bulk_dihedrals = split_bulk_termini(
                n_term_atoms, c_term_atoms, dihedral_edges
            )
            if len(bulk_dihedrals) == 0:
                dihedrals = [
                    ("n_term_dihedrals", 4, n_term_dihedrals),
                    ("bulk_dihedrals", 4, torch.tensor([]).reshape(4, 0)),
                    ("c_term_dihedrals", 4, c_term_dihedrals),
                ]

            elif len(n_term_dihedrals) == 0 or len(c_term_dihedrals) == 0:
                dihedrals = [
                    ("n_term_dihedrals", 4, torch.tensor([]).reshape(4, 0)),
                    ("bulk_dihedrals", 4, bulk_dihedrals),
                    ("c_term_dihedrals", 4, torch.tensor([]).reshape(4, 0)),
                ]
            else:
                dihedrals = [
                    ("n_term_dihedrals", 4, n_term_dihedrals),
                    ("bulk_dihedrals", 4, bulk_dihedrals),
                    ("c_term_dihedrals", 4, c_term_dihedrals),
                ]
        else:
            dihedrals = ("dihedrals", 4, dihedral_edges)
        print('DIHEDRALES : ',dihedrals)
        return dihedrals

