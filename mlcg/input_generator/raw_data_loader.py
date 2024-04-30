import numpy as np
import os
from natsort import natsorted
from glob import glob
import h5py
from typing import Tuple
import mdtraj as md
import re

class DatasetLoader:
    pass


class CATH_loader(DatasetLoader):
    """
    Loader object for original 50 CATH domain proteins
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given CATH domain name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given CATH domain name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        # return sorted(name, key=alphanum_key)
        outputs_fns = natsorted(glob(os.path.join(base_dir, f"output/{name}/*_part_*")))
        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for fn in outputs_fns:
            output = np.load(fn)
            coord = output["coords"]
            coord = 10.0 * coord  # convert nm to angstroms
            force = output["Fs"]
            force = force / 41.84  # convert to from kJ/mol/nm to kcal/mol/ang
            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        print('No da: ',aa_coord_list)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class CATH_ext_loader(DatasetLoader):
    """
    Loader object for extended dataset of CATH domain proteins
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given CATH domain name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb_fns = glob(pdb_fn.format(name))
        pdb = md.load(pdb_fns[0])
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given CATH domain name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        traj_dirs = glob(os.path.join(base_dir, f"group_*/{name}_*/"))
        all_coords = []
        all_forces = []
        for traj_dir in traj_dirs:
            traj_coords = []
            traj_forces = []
            fns = glob(os.path.join(traj_dir, "prod_out_full_output/*.npz"))
            fns.sort(key=lambda file: int(file.split("_")[-2]))
            last_parent_id = None
            for fn in fns:
                np_dict = np.load(fn, allow_pickle=True)
                current_id = np_dict["id"]
                parent_id = np_dict["parent_id"]
                if parent_id is not None:
                    assert parent_id == last_parent_id
                traj_coords.append(np_dict["coords"])
                traj_forces.append(np_dict["Fs"])
                last_parent_id = current_id
            traj_full_coords = np.concatenate(traj_coords)
            traj_full_forces = np.concatenate(traj_forces)
            if traj_full_coords.shape[0] != 25000:
                continue
            else:
                all_coords.append(traj_full_coords)
                all_forces.append(traj_full_forces)
        full_coords = np.concatenate(all_coords)
        full_forces = np.concatenate(all_forces)
        return full_coords, full_forces


class DIMER_loader(DatasetLoader):
    """
    Loader object for original dataset of mono- and dipeptide pairwise umbrella sampling simulations
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given DIMER pair name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given DIMER pair name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        with h5py.File(os.path.join(base_dir, "allatom.h5"), "r") as data:
            coord = data["MINI"][name]["aa_coords"][:]
            force = data["MINI"][name]["aa_forces"][:]

        # convert to kcal/mol/angstrom and angstrom
        # from kJ/mol/nm and nm
        coord = coord * 10
        force = force / 41.84

        return coord, force


class DIMER_ext_loader(DatasetLoader):
    """
    Loader object for extended dataset of mono- and dipeptide pairwise umbrella sampling simulations
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given DIMER pair name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given DIMER pair name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        coord = np.load(
            glob(os.path.join(base_dir, f"dip_dimers_*/data/{name}_coord.npy"))[0],
            allow_pickle=True,
        )
        force = np.load(
            glob(os.path.join(base_dir, f"dip_dimers_*/data/{name}_force.npy"))[0],
            allow_pickle=True,
        )

        # convert to kcal/mol/angstrom and angstrom
        # from kJ/mol/nm and nm
        coord = coord * 10
        force = force / 41.84

        return coord, force


class Trpcage_loader(DatasetLoader):
    """
    Loader object for Trpcage simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        print('ALDO: ',aa_traj,'TOPOL: ',top_dataframe)
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        print('HACE ALGO')
        coords_fns = natsorted(
            glob(
                os.path.join(base_dir, f"coords_nowater/chig_coor*.npy")
            )
        )

        forces_fns = [
            fn.replace(
                "coords_nowater/trp_coor_folding", "forces_nowater/chig_force*"
            )
            for fn in coords_fns
        ]

        aa_coord_list = []
        aa_force_list = []
        
        # load the files, checking against the mol dictionary
        for cfn, ffn in zip(coords_fns, forces_fns):
            force = np.load(ffn)
            coord = np.load(cfn)
            coord = coord  # * 10
            force = force / 4.184  # convert to from kJ/mol/ang to kcal/mol/ang
            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class Cln_loader(DatasetLoader):
    def get_traj_top(self, name: str, pdb_fn: str):
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self, base_dir: str, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        coords_fns = natsorted(
            glob(os.path.join(base_dir, f"coords_nowater/chig_coor_*.npy"))
        )

        forces_fns = [
            fn.replace("coords_nowater/chig_coor_", "forces_nowater/chig_force_")
            for fn in coords_fns
        ]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in zip(coords_fns, forces_fns):
            force = np.load(ffn)
            coord = np.load(cfn)
            coord = coord  # * 10
            force = force / 4.184  # convert to from kJ/mol/ang to kcal/mol/ang
            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class SimInput_loader(DatasetLoader):
    """
    Loader for protein structures to be used in CG simulations
    """

    def get_traj_top(self, name: str, raw_data_dir: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        raw_data_dir:
            Path to pdb structure file
        """
        pdb = md.load(f"{raw_data_dir}{name}.pdb")
        input_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = input_traj.topology.to_dataframe()[0]
        return input_traj, top_dataframe







##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
class Membrane_loader(DatasetLoader):
    """
    Loader object for extended dataset of POPC membranes it will be increase to future models
    """
    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given CATH domain name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb_fns = glob(pdb_fn.format(name))
        pdb = md.load(pdb_fns[0])
        pdb = pdb.remove_solvent()
        print('Sistema: ',pdb)
        aa_traj = pdb.atom_slice([a.index for a in pdb.topology.atoms]) #if a.residue.is_protein])
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe
    # Función para dividir el nombre del directorio en partes numéricas
    def obtener_partes_numericas(self, nombre_directorio):
        partes = []
        for parte in nombre_directorio.split():
            if parte.isdigit():
                partes.append(int(parte))
            else:
                partes.append(parte)
        return partes 
    def load_coords_forces(self, base_dir: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given CATH domain name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        print('Previo: ',base_dir) #solo de 0,1 y 2 [0-9]
        #traj_dirs = glob(os.path.join(base_dir,f"[1-2][0-9]/"))#"[1][0-9]/"))   #f"1/")) #f"[0-1]?[0-9]|20/"))
        traj_dirs = [os.path.join(base_dir, f"{i}/") for i in range(10, 20)] 
        print('Directorio: ',traj_dirs)
        traj_dirs = sorted(traj_dirs)#, key=lambda x: int(x.split('_')[0]))
        all_coords = []
        all_forces = []
        contador = 0
        #print('DIR: ',traj_dirs)
        for traj_dir in traj_dirs:
            traj_coords = []
            traj_forces = []
            fns = glob(os.path.join(traj_dir, "production_full_output/*[024].npz"))
            for fn in fns:
                contador = contador +1
                np_dict = np.load(fn, allow_pickle=True)
                coords =  np_dict["coords"].astype(np.float16)
                print('SHAPE: ',coords.shape)
                forces = np_dict["Fs"].astype(np.float16)
                assert coords.shape == forces.shape
                traj_coords.append(coords)
                traj_forces.append(forces)
                #last_parent_id = current_id
            #print('COORDS: ', traj_forces)
            #print('Llega hasta este punto: ')
            traj_full_coords = np.concatenate(traj_coords)
            traj_full_forces = np.concatenate(traj_forces)
            #if traj_full_coords.shape[0] != 25000:
            #    continue
            #else:
            all_coords.append(traj_full_coords)
            all_forces.append(traj_full_forces)
        print('PREVIO FUERZAS CONCATENADO')
        print('NUMERO DE CARPETAS: ',contador)
        full_coords = np.concatenate(all_coords)
        full_forces = np.concatenate(all_forces)
        print('Fuerzas: ',full_forces.shape)
        print('Hasta aqui bien')
        return full_coords, full_forces




    #ESTE DEBE SER EL MEJOR

    #Cogiendo 0,2,4 en .npz y 10 a 29 en carpetas
    def load_coords_forces_n(self, base_dir: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
        import numpy as np
        import os
        from glob import glob
        def generate_data(traj_dirs):
            for traj_dir in traj_dirs: # 024 bien
                fns = glob(os.path.join(traj_dir, f"production_full_output/*[024].npz"))#[02468]
                for fn in fns:
                    np_dict = np.load(fn, allow_pickle=True)
                    coords = np_dict["coords"].astype(np.float16)
                    forces = np_dict["Fs"].astype(np.float16)
                    assert coords.shape == forces.shape
                    yield coords, forces
        all_coords = []
        all_forces = []
        traj_dirs = glob(os.path.join(base_dir, f"[1-2][0-9]/"))  # Suponiendo "1" como carpeta de trayectorias
        print('DIRS: ',traj_dirs)
        #traj_dirs.extend(glob(os.path.join(base_dir, f"[1-5]/")))
        #traj_dirs = sorted(traj_dirs)
        print('DIRS: ',traj_dirs)
        #pattern = re.compile(r'^[0-2]?[0-9]/$')

        # Listar todos los archivos y directorios en el directorio base
        #traj_dirs = [nombre for nombre in os.listdir(base_dir) if pattern.match(nombre)]
        #print('DIRS: ',traj_dirs)


        for coords, forces in generate_data(traj_dirs):
            all_coords.append(coords)
            all_forces.append(forces)

        if not all_coords or not all_forces:
            raise ValueError("No se encontraron archivos de datos")

        # Concatenar todas las matrices de coordenadas y fuerzas
        full_coords = np.concatenate(all_coords, axis=0)
        full_forces = np.concatenate(all_forces, axis=0)

        return full_coords, full_forces



        #METODO CHAT GPT
    import numpy as np
    import os
    from glob import glob

    def generate_data(self,traj_dirs):
        for traj_dir in traj_dirs:
            fns = glob(os.path.join(traj_dir, "production_full_output/*.npz"))#[02468]
            for fn in fns:
                np_dict = np.load(fn, allow_pickle=True)
                coords = np_dict["coords"].astype(np.float16)
                forces = np_dict["Fs"].astype(np.float16)
                if coords.shape != forces.shape:
                    raise ValueError("Shapes of coordinates and forces do not match")
                yield coords, forces

    def load_coords_forces_a(self, base_dir: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
        traj_dirs = [os.path.join(base_dir, f"{i}/") for i in range(1, 30)]  # Carpeta de trayectorias de 10 a 29
        #traj_dirs.extend([os.path.join(base_dir, f"{i}/") for i in range(1, 9)])  # Carpeta de trayectorias de 1 a 5

        batch_size = 10  # Reducir el tamaño del lote

        data_generator = self.generate_data(traj_dirs)

        all_coords = []
        all_forces = []
        for coords, forces in data_generator:
            all_coords.append(coords)
            all_forces.append(forces)

            if len(all_coords) >= batch_size:
                full_coords = np.concatenate(all_coords, axis=0)
                full_forces = np.concatenate(all_forces, axis=0)
                print("Tamaño de datos restantes:", full_coords.shape, full_forces.shape)
                yield full_coords, full_forces
                # Limpiar memoria
                all_coords.clear()
                all_forces.clear()

        # Procesar los datos restantes si quedan
        if all_coords:
            full_coords = np.concatenate(all_coords, axis=0)
            full_forces = np.concatenate(all_forces, axis=0)
            print("Tamaño de datos restantes:", full_coords.shape, full_forces.shape)
            yield full_coords, full_forces

