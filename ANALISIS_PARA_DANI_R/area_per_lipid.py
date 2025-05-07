import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder
from MDAnalysis.analysis.rdf import InterRDF
import matplotlib.pyplot as plt
import numpy as np
pdb_file = 'input_openmm.pdb'
dcd_file = 'production.dcd'

# Cargar los archivos de trayectoria y estructura
u = mda.Universe(pdb_file, dcd_file)

#Area per lipid
# Seleccionar los lípidos (asumiendo que los lípidos están etiquetados como 'resname POPC' o similar)
lipids = u.select_atoms('resname POPC')

# Crear un objeto LeafletFinder para identificar las monocapas de la bicapa lipídica
L = LeafletFinder(u, 'name P*')

# Asumimos que los lípidos están en la primera monocapa
lipid_leaflet = L.groups(0)

# Calcular el área de la bicapa en cada frame
areas = []
areas = []
for ts in u.trajectory:
    x_positions = lipid_leaflet.positions[:, 0]
    y_positions = lipid_leaflet.positions[:, 1]
    x_max, y_max = np.max(x_positions), np.max(y_positions)
    x_min, y_min = np.min(x_positions), np.min(y_positions)
    area = (x_max - x_min) * (y_max - y_min)
    areas.append(area / len(lipid_leaflet))

# Convertir a un array de numpy para facilidad de uso
areas = np.array(areas)

# Guardar los resultados en un archivo de texto
np.savetxt('area_per_lipid.txt', areas)

# Imprimir el área promedio por lípido
print(f'Área promedio por lípido: {np.mean(areas)}+-{np.std(areas)} Å^2')





#RDF
# Seleccionar las cabezas de los fosfolípidos (asumiendo que las cabezas están etiquetadas como 'name P')
phosphate_heads = u.select_atoms('name P')

# Seleccionar el resto de los átomos de los lípidos excluyendo las cabezas de fosfolípidos
# Asumimos que los lípidos están etiquetados como 'resname POPC' o similar
lipid_tails = u.select_atoms('resname POP and not name P')

# Asegurarse de que hay suficientes átomos seleccionados
print(f"Number of phosphate head atoms selected: {len(phosphate_heads)}")
print(f"Number of lipid tail atoms selected: {len(lipid_tails)}")

# Definir el rango y el número de bins para el cálculo de la RDF
rdf_range = (1.0, 20.0)  # Distancia máxima de 15 Å
nbins = 200  # Número de bins para el histograma RDF

# Calcular la RDF entre las cabezas de fosfato y las colas de los lípidos a lo largo de toda la trayectoria
rdf = InterRDF(phosphate_heads, phosphate_heads,nbins=nbins,range=rdf_range,norm = 'density')#, range=rdf_range, nbins=nbins)
rdf.run()
#cdf = rdf.get_cdf()
# Verificar los resultados
print(f"RDF bins: {rdf.bins}")
print(f"RDF values: {rdf.rdf}")

# Graficar los resultados
plt.figure(figsize=(8, 6))
plt.plot(rdf.bins, rdf.rdf, label='RDF P-Tails')
plt.xlabel('Distance (Å)')
plt.ylabel('RDF')
plt.title('Radial Distribution Function between Phosphate Heads and Lipid Tails')
plt.legend()
plt.grid(True)
plt.savefig('rdf_p_tails.png')




#Grosor membrana:
# Seleccionar los átomos de las cabezas de fosfolípidos (asumiendo que las cabezas están etiquetadas como 'name P')
phosphate_heads = u.select_atoms('name P')

# Crear un objeto LeafletFinder para identificar las monocapas de la bicapa lipídica
L = LeafletFinder(u, 'name P')

# Asumimos que los lípidos están en las dos monocapas identificadas
leaflet1 = L.groups(0)
leaflet2 = L.groups(1)

# Calcular el grosor de la membrana en cada frame
thicknesses = []
for ts in u.trajectory:
    z_positions_leaflet1 = leaflet1.positions[:, 2]
    z_positions_leaflet2 = leaflet2.positions[:, 2]
    mean_z_leaflet1 = np.mean(z_positions_leaflet1)
    mean_z_leaflet2 = np.mean(z_positions_leaflet2)
    thickness = np.abs(mean_z_leaflet1 - mean_z_leaflet2)
    thicknesses.append(thickness)

# Convertir a un array de numpy para facilidad de uso
thicknesses = np.array(thicknesses)

# Guardar los resultados en un archivo de texto
np.savetxt('membrane_thickness.txt', thicknesses)

# Graficar el grosor de la membrana a lo largo del tiempo
plt.figure(figsize=(8, 6))
plt.plot(thicknesses, label='Membrane Thickness')
plt.xlabel('Frame')
plt.ylabel('Thickness (Å)')
plt.title('Membrane Thickness over Time')
plt.legend()
plt.grid(True)
plt.savefig('membrane_thickness.png')
plt.show()

# Imprimir el grosor promedio de la membrana
print(f'Grosor promedio de la membrana: {np.mean(thicknesses):.2f} Å')
