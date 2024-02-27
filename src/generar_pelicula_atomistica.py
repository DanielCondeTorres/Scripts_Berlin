import MDAnalysis as mda
import mdtraj as md
import numpy as np
def lectura_archivos_npz_y_matching_gro(file_npz, file_pdb):
# Carga el archivo .npz
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

claves, fuerzas, coordenadas, numero_lipidos_popc = lectura_archivos_npz_y_matching_gro('archivos_npz/production_116.npz','input_openmm.pdb')
# Supongamos que tienes tus coordenadas en un array tridimensional
coords =  coordenadas  # Reemplazar con tus datos reales

# Nombre del archivo .xyz
nombre_archivo = "pelicula.xyz"

# Número total de átomos por frame
num_atomos = coords.shape[1]

# Abrir el archivo .xyz para escribir
with open(nombre_archivo, 'w') as f:
    # Iterar sobre los frames
    for frame_idx in range(coords.shape[0]):
        # Escribir el encabezado con el número total de átomos
        f.write(f"{num_atomos}\n")
        f.write(f"Frame {frame_idx + 1}\n")

        # Iterar sobre los átomos en el frame y escribir las coordenadas
        for i in range(num_atomos):
            x, y, z = coords[frame_idx, i, :]
            f.write(f"A {x} {y} {z}\n")
