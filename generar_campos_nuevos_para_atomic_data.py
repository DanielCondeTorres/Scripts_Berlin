import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import torch
import warnings
import os
from importlib import import_module

from torch_geometric.data.collate import collate
import sys
import os.path as osp

# Añadir la ruta del directorio que contiene el paquete mlcg
sys.path.append('/home/daniel/Escritorio/BERLIN_PROJECT/SCRIPT_CONVERSION/NUEVO_MLCG_CON_PBC')
print("sys.path:", sys.path)

# Verificar el contenido del directorio mlcg
directory_path = '/home/daniel/Escritorio/BERLIN_PROJECT/SCRIPT_CONVERSION/NUEVO_MLCG_CON_PBC/mlcg'
print("Contenido del directorio mlcg:", os.listdir(directory_path))

# Intentar importar el módulo mlcg.utils
try:
    import mlcg.utils as utils
    print("Módulo mlcg.utils importado exitosamente")
except ModuleNotFoundError as e:
    print("Error al importar mlcg.utils:", e)

# Cargar datos usando torch
datos = torch.load("input_configurations_for_simulation.pt")
print('Positions: ',datos[0].pos)
# Iterar sobre cada objeto AtomicData en la lista y asignar el nuevo campo
pbc = torch.tensor([[True, True, True]])
print('Ver valores de la caja en atomisitca en Amstrongs y ponerlas en la siguiente matriz')
values_of_the_box =torch.tensor([[[35.,  0.,  0.],
         [ 0., 35.,  0.],
         [ 0.,  0., 35.]]])
for atomic_data in datos:
    atomic_data.pbc = pbc
    atomic_data.cell = values_of_the_box
print('A_ ',datos[0].nuevo_campo)
print(datos[0].nuevo_campo)
