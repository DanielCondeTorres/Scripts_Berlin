import argparse
from pathlib import Path
from copy import deepcopy
import mdtraj as md
import numpy as np

'''
This script can map any membrane, if you have the .map and the .itp CG files in two directories, and one pdb structure in atomistic
'''


parser = argparse.ArgumentParser(description='Create CG matrix from martini .map file')
parser.add_argument('--pdb', required=True, help='.pdb file with the system')
parser.add_argument('--map', required=True, help='path to .map directory')
parser.add_argument('--itp', required=True, help='path to .itp directory')
args = parser.parse_args()


#This function creates the maping matrix, that convert atomistic coordintes to forces
def create_martini_cg(pdb_file:str, map_file:str, system:dict, new_top:dict, path_to_itp:str, lipid_name:str, lip_idx:int):
    #Calculate the number of lipids in the system for each specific lipid
    num_lipid = len(set([atom.residue.resSeq for atom in system.top.atoms if atom.residue.name == lipid_name]))
    #Open the map files, and obtain the relation between beads and real atoms
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
    output_dict = {item[0]: item[1:] for item in atoms_per_bead if item}
    atoms_per_lip = list(output_dict.keys())
    masses = {'H': 1,'C': 12,'N': 14,'O': 16,'S': 32,'P': 31,'M': 0, 'B': 32}
    cg_map = np.zeros((num_lipid*len(martini_bead_list),system.n_atoms))
    #Initial value to take account the position
    c_inicial=-1
    for c,res in enumerate(system.top.residues):
        if res.name == lipid_name:
            if c_inicial == -1:
                c_inicial = c
            else:
                pass
            for atom in res.atoms:
                if c_inicial== -1:
                aa_idx = atom.index
                m = masses[atom.name[0]]
                contributions = output_dict[atom.name]
                cg_indexes = [martini_bead_list.index(bead) for bead in contributions]
                cg_indexes = [idx+(c-c_inicial)*len(martini_bead_list) for idx in cg_indexes]
                cg_map[cg_indexes,aa_idx] = m
        else: continue
    col_sum = cg_map.sum(axis=1)
    cg_map = (cg_map.T*(1/col_sum)).T
    # As the CG change is not a simple slicing of the original AA topology
    # we need to create the new topology from scratch.
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
    with open(path_to_itp , 'r') as file:
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
        new_top.add_residue(f"{lipid_name}",new_top.chain(lip_idx))
        for j,elem in enumerate(martini_bead_list,start=1):
            md_elem = element_map[elem[0]]

            #Duda en el c_inicial
            new_atom = new_top.add_atom(f"{elem}{j}", md_elem, new_top.residue(i+c_inicial))
            
        for enlace in enlaces_definidos:
            bead1_index, bead2_index = enlace
            
            #Duda en el c_inicial
            new_top.add_bond(new_top.atom(bead1_index - 1 +c_inicial), new_top.atom(bead2_index - 1 + c_inicial))
    config_map_matrix = cg_map
    force_map_matrix = cg_map*4.184
    return config_map_matrix, force_map_matrix, new_top

import os
# apply the cg map to the system coordinates
system = md.load_pdb(args.pdb)
#Solvent is not needed
system = system.remove_solvent()
# Obtain topology of the system
topologia = system.topology

# List with the residue names, we need to iterate and search in the directories
nombres_residuos = [res.name for res in topologia.residues]
composition = list(set(nombres_residuos))#['CHOL','POPE','POPG','POPS','DOPS','POR']
new_top = md.Topology()
print('Membrane composition: ',composition)
boolean = False
lip_idx = 0
for lipidos in composition:
    print('Lipidos: ',lipidos)
    path_to_itp = f'{args.itp}/{lipidos}.itp'
    path_to_map = f'{args.map}/{lipidos}.map'
    new_top.add_chain()    
    indices_residuo_deseado = system.top.select(f'resname {lipidos}')
# Crear un nuevo objeto Trajectory que contiene solo el residuo deseado
    system_con_residuo_deseado = system.atom_slice(indices_residuo_deseado)
    if os.path.isfile(path_to_map): 
        config_map_matrix,force_map_matrix, new_top = create_martini_cg(args.pdb, path_to_map, system,  new_top,path_to_itp, f'{lipidos}',lip_idx)
        lip_idx = lip_idx+1
        if boolean == False:
            matriz_total_coordenadas = config_map_matrix
            matriz_total_fuerzas = force_map_matrix
            boolean = True
        else:
            matriz_total_coordenadas= np.vstack((matriz_total_coordenadas, config_map_matrix))
            matriz_total_fuerzas = np.vstack((matriz_total_fuerzas, force_map_matrix))
    else:
        continue
new_coords = matriz_total_coordenadas @ system.xyz
traj = md.Trajectory(xyz = new_coords, topology = new_top)
path = Path(args.pdb)
traj.save(f"{path.stem}_martinini.gro")

























'''
import MDAnalysis as mda
def lectura_archivos_npz_y_matching_gro(file_npz, file_pdb):
    archivo_npz = np.load(file_npz)
    # Muestra las claves (nombres) de los elementos almacenados en el archivo
    claves = archivo_npz.files
    coordenadas = archivo_npz['coords']
    fuerzas = archivo_npz['Fs']
    #Leemos archivo pdb
    system=mda.Universe(file_pdb)
    seleccion_popc = system.select_atoms("resname POP")
    numero_lipidos_popc = len(set(seleccion_popc.resids))
    print('Tamaño : ',numero_lipidos_popc,len(seleccion_popc))
# Cierra el archivo después de obtener las claves
    archivo_npz.close()
    return claves, fuerzas, coordenadas, numero_lipidos_popc
claves, fuerzas, coordenadas, numero_lipidos_popc = lectura_archivos_npz_y_matching_gro('../FORCES_MATCHING/archivos_npz/production_116.npz','input_openmm.pdb')

coordenadas_cg = config_map_matrix @ coordenadas
print('NEW TOP:' ,new_top)
traj = md.Trajectory(xyz=coordenadas_cg,topology=new_top)
path = Path(args.pdb)
traj.save_pdb(f"{path.stem}_martinini2{path.suffix}")

fuerzas_cg = force_map_matrix @ fuerzas
print('Fuerzas:')
print('Max at: ',np.max(fuerzas),np.argmax(fuerzas),'Min at: ',np.min(fuerzas),np.argmin(fuerzas))
print('Max cg: ', np.max(fuerzas_cg),np.argmax(fuerzas_cg),'Min cg: ',np.min(fuerzas_cg),np.argmin(fuerzas_cg))
print('#####################################################')
print(' ')
print('COordenadas: ')
print('Max at: ',np.max(coordenadas),np.argmax(coordenadas),'Min at: ',np.min(coordenadas),np.argmin(coordenadas))
print('Max cg: ', np.max(coordenadas_cg),np.argmax(coordenadas_cg),'Min cg: ',np.min(coordenadas_cg),np.argmin(coordenadas_cg))
import matplotlib.pyplot as plt





# Extraer las coordenadas para cada dimensión
import matplotlib as mpl
mpl.use('Agg')
def function_to_plot(matrix: np.array, name:str):
    x_coords = matrix[:, :, 0].flatten()
    y_coords = matrix[:, :, 1].flatten()
    z_coords = matrix[:, :, 2].flatten()
    # Crear histogramas para cada dimensión
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(x_coords, bins=50, color='red', alpha=0.7, edgecolor='black')
    plt.title(f'Histograma de {name} X')

    plt.subplot(1, 3, 2)
    plt.hist(y_coords, bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.title(f'Histograma de {name} Y')

    plt.subplot(1, 3, 3)
    plt.hist(z_coords, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f'Histograma de {name} Z')

    plt.tight_layout()
    plt.savefig(name,dpi=300)


function_to_plot(fuerzas, 'Fuerzas CG')'''
