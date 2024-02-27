import numpy as np
import MDAnalysis as mda
import mdtraj as md
import csv
import re
def escribir_xyz(coordenadas, nombres, nombre_archivo="nuevas_coordenadas_ordenadas.xyz"):
    with open(nombre_archivo, "a") as archivo:
        n_atomos = len(coordenadas)
        archivo.write(f"{n_atomos}\n")
        archivo.write("Coordenadas generadas con Python\n")
        for i, (x, y, z) in enumerate(coordenadas):
            nombre_atomo = nombres[i] if i < len(nombres) else "W"
            archivo.write(f"{nombre_atomo} {x:12.6f} {y:12.6f} {z:12.6f}\n")

'''def escribir_csv(matriz, nombres):
    with open('fuerzas.csv', 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)

        # Escribir encabezados
        escritor_csv.writerow(nombres)

        # Escribir datos
        for fila in matriz:
            escritor_csv.writerow(fila)'''
def escribir_csv(matriz, nombres):
    with open('fuerzas.csv', 'w', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)

        # Escribir encabezados
        escritor_csv.writerow(['', ''] + nombres)  # Agregar dos columnas vacías antes de los nombres

        # Escribir datos
        for fila in matriz:
            escritor_csv.writerow([''] + [str(valor) for valor in fila])  # Convertir valores a cadenas antes de la concatenación


def indices_segun_pdb(archivo_pdb):
    # Lista para almacenar los nombres de átomos
    atomos_con_valor_1 = []

    # Abrir el archivo PDB y leer línea por línea
    with open(archivo_pdb, 'r') as pdb_file:
        for linea in pdb_file:
            # Verificar si la línea es un registro de átomo (ATOM o HETATM)
            if linea.startswith('ATOM') or linea.startswith('HETATM'):
                # Obtener el nombre del átomo y el valor en la sexta columna
                nombre_atomo = linea[12:16].strip()  # Columnas 13 a 16 contienen el nombre del átomo
                valor_columna_seis = float(linea[24:29].strip())  # Columnas 55 a 60 contienen el valor
                tipo_residuo = linea[17:20].strip()  # Columnas 18 a 20 contienen el tipo de residuo 
                # Verificar si el valor en la sexta columna es 1
                if valor_columna_seis == 1.0 and tipo_residuo == 'POP':
                    atomos_con_valor_1.append(nombre_atomo)
    return atomos_con_valor_1






def creacion_diccionario(archivo_map, file_pdb):
    traj = md.load(file_pdb,top=file_pdb)
    topology = traj.topology

    with open(archivo_map,'r') as file:
        lines = file.readlines()
    # Inicializa listas vacías para almacenar los resultados
    martini_list = []
    atoms_list_2columns = []
    atoms_list_3columns = []
    # Itera sobre las líneas del archivo
    in_martini_section = False
    in_atoms_section = False
    # Itera sobre las líneas del archivo
    for i, line in enumerate(lines):
        # Verifica si estamos dentro de la sección [martini]
        if line.strip() == '[ martini ]':
            in_martini_section = True
            continue
        elif in_martini_section and line.startswith('['):
            in_martini_section = False
        # Verifica si estamos dentro de la sección [atoms]
        if line.strip() == '[ atoms ]':
            in_atoms_section = True
            continue
        elif in_atoms_section and line.startswith('['):
            in_atoms_section = False
        # Procesa las líneas según la sección actual
        if in_martini_section and line.strip():
            martini_list.extend(line.strip().split())
        elif in_atoms_section and line.strip():
            columns = line.split()
            atoms_list_3columns.append(columns[1:])
    # Crear el diccionario
    output_dict = {item[0]: item[1:] for item in atoms_list_3columns if item}
    # Imprimir el diccionario
    mass = {'H': 1,'C': 12,'N': 14,'O': 16,'S': 32,'P': 31,'M': 0, 'B': 32}
    # Nuevo diccionario con los valores numéricos añadidos
    nuevo_diccionario = {}
    for clave, valor in output_dict.items():
        for letra, masa in mass.items():
            if clave.startswith(letra):
                nuevo_diccionario[clave] = valor + [masa]
                break  # Rompemos el bucle una vez que encontramos la letra correspondiente
    # Display all keys in the dictionary
    all_keys = output_dict.keys()
    # Convert the view object to a list if needed
    
    elements = list(all_keys)
    elements = indices_segun_pdb(file_pdb)
    
    # Create an empty dictionary to store the results
    result_dict = {}
    # Loop through each element and apply the topology.select() method
    for element in elements:
        result_dict[f'{element}_lipid'] = topology.select(f'name {element}')
    nuevo_diccionario = dict(sorted(nuevo_diccionario.items(), key=lambda x: elements.index(x[0])))
    return result_dict, martini_list, nuevo_diccionario, mass, elements



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

def asignar_atomos_a_las_matrices(matrix, file_pdb, elements, numero_lipidos_popc
):
    traj = md.load(file_pdb, top=file_pdb)
    topology = traj.topology
    membrane_selection = topology.select("resname POP or resname POPC or resname POPE or resname POPS or resname CHOL")
    # Obtener los nombres de los átomos en la selección
    atom_names = [topology.atom(i).name for i in membrane_selection]#*numero_lipidos_popc

    #atom_names = elements*numero_lipidos_popc

    # Crear un diccionario para mapear los elementos a las matrices correspondientes
    mapeo_elementos = {}
    for i, elemento in enumerate(atom_names):
        if elemento not in mapeo_elementos:
            mapeo_elementos[elemento] = []

        # Añadir la matriz correspondiente al elemento en el diccionario
        mapeo_elementos[elemento].append(matrix[i, :])
    mapeo_ordenado = {key: mapeo_elementos[key] for key in elements if key in mapeo_elementos}
    return mapeo_ordenado


claves, fuerzas, coordenadas, numero_lipidos_popc = lectura_archivos_npz_y_matching_gro('archivos_npz/production_116.npz','input_openmm.pdb')
result_dict, martini_list, nuevo_diccionario, mass ,elements= creacion_diccionario('popc.amber.map','input_openmm.pdb')
archivo_numero = re.search(r'\d+', 'archivo_npz/production_116.npz').group()
print('NUMERO: ', archivo_numero)
print(claves)
print('Fuerzas: ', fuerzas)


def matching(matriz_a_estudiar,marini_list, nuevo_diccionario, archivo_numero):
    datos_acumulados = {'bead': [], 'Fuerzas': []}
    contador = 0
    for lipid_coordinates in coordenadas: #coordenadas are frames
        mapeo_elementos = asignar_atomos_a_las_matrices(lipid_coordinates,'input_openmm.pdb',elements,numero_lipidos_popc)
        matrices = [] 
        for elemento in mapeo_elementos:
            matrices.append(np.vstack(mapeo_elementos[elemento]))
        lipid_coordinates_n = np.vstack(np.column_stack(matrices).reshape(-1, matrices[0].shape[1]))
        R_split = np.split(lipid_coordinates_n,numero_lipidos_popc)
        R_stack = np.block(R_split)
        nombres_coordenadas = []#listas de almacenaje
        cg_coordinates = []
        matriz_a_guardar = []
        for elemento in martini_list:
            sumatorio = []
        #    seleccionados = {clave: valor for clave, valor in nuevo_diccionario.items() if elemento in valor}
            for clave, valor in nuevo_diccionario.items():
                letra_inicial = clave[0]
                valor_numerico = mass.get(letra_inicial, 0)
                count_nc3 = valor.count(elemento)
                size_lista = len(valor) - 1
                resultados= count_nc3 * valor_numerico/size_lista
                sumatorio.append(resultados)

            mapping_matrix_lipid = np.array(sumatorio)/sum(sumatorio)
            print('Matriz de fuerzas !',mapping_matrix_lipid)
            matriz_a_guardar.append(mapping_matrix_lipid)
            print('F atom: ',R_stack)
            cg_coordinates_lipids = np.matmul(mapping_matrix_lipid,R_stack).reshape((numero_lipidos_popc,3))
            #print('Finaal: ', cg_coordinates)
            nombres_coordenadas_elemento = [elemento]
            nombres_coordenadas_lipido = nombres_coordenadas_elemento*len(cg_coordinates_lipids)
            nombres_coordenadas.extend(nombres_coordenadas_lipido)
            cg_coordinates.append(cg_coordinates_lipids)   

        cg_coordinates = np.vstack(cg_coordinates)
        if matriz_a_estudiar is coordenadas: 
            escribir_xyz(cg_coordinates,nombres_coordenadas)

            #####################
            try:
                # Cargar el archivo .npz existente
                loaded_data = np.load('archivo_guardado.npz')
                new_data = {key: loaded_data[key] for key in loaded_data.files}
                new_data['Coords']= cg_coordinates
                # Guardar el archivo actualizado
                np.savez(f'archivo_actualizado_CG_{archivo_numero}.npz', **new_data)
            except:
                pass
            #####################
        if matriz_a_estudiar is fuerzas:
            #escribir_csv(cg_coordinates,nombres_coordenadas)
            # Crear un diccionario con las columnas especificadas
            data = {
                    'bead': nombres_coordenadas,
                    'Fuerzas': cg_coordinates
                    }

            datos_acumulados['bead'].extend(nombres_coordenadas)
            datos_acumulados['Fuerzas'].extend(cg_coordinates)
            # Guardar el archivo .npz
            #np.savez(f'archivo_CG2_{archivo_numero}.npz', **data)
        if contador == 0:
            matriz_a_guardar = np.vstack(tuple(matriz_a_guardar))
            #print('Matriz',matriz_a_guardar.shape)
            np.save('matriz_para_NOE.npy', matriz_a_guardar)
        contador = contador + 1
    return cg_coordinates,datos_acumulados

#Fuerzas o coordenadas
a,lista_data = matching(fuerzas, martini_list, nuevo_diccionario, archivo_numero)
np.savez(f'archivo_CG2_{archivo_numero}.npz', **lista_data)
