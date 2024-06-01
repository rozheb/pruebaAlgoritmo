import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# URL del archivo CSV
url = "https://raw.githubusercontent.com/rozheb/pruebaAlgoritmo/main/datasetrutas3.csv"
# Cargar datos desde la URL
datos = pd.read_csv(url,on_bad_lines='skip')

# Verificar las primeras filas y las columnas disponibles
print(datos.head())
print(datos.columns)

# Extraer las coordenadas de latitud y longitud
locations = datos[['Latitud', 'Longitud']].values

# Plot para visualizar las ubicaciones
plt.figure(figsize=(10, 8))
plt.scatter(locations[:, 0], locations[:, 1], marker='o', c='red', edgecolor='b')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ubicaciones')
plt.grid(True)
plt.show()

# Calcular matriz de distancias euclidianas
distance_matrix = cdist(locations, locations, metric='euclidean')

# Crear el modelo de vecinos más cercanos
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(locations)
distances, indices = nbrs.kneighbors(locations)

# Imprimir los índices de los vecinos más cercanos
print(indices)

def nearest_neighbor_route(start_index, distance_matrix):
    n = distance_matrix.shape[0]
    visited = [False] * n
    route = [start_index]
    visited[start_index] = True
    
    for _ in range(n - 1):
        last_index = route[-1]
        nearest_index = None
        nearest_distance = float('inf')
        
        for i in range(n):
            if not visited[i] and distance_matrix[last_index, i] < nearest_distance:
                nearest_index = i
                nearest_distance = distance_matrix[last_index, i]
        
        route.append(nearest_index)
        visited[nearest_index] = True
    
    return route

# Crear una ruta comenzando desde el primer punto
start_index = 0
route = nearest_neighbor_route(start_index, distance_matrix)
print("Ruta utilizando vecino más cercano:", route)

# Plot para visualizar la ruta
route_locations = locations[route]
plt.figure(figsize=(10, 8))
plt.plot(route_locations[:, 0], route_locations[:, 1], 'o-')
plt.scatter(locations[:, 0], locations[:, 1], marker='o', c='red', edgecolor='b')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ruta optimizada usando vecino más cercano')
plt.grid(True)
plt.show()

def two_opt(route, distance_matrix):
    best_route = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if total_distance(new_route, distance_matrix) < total_distance(best_route, distance_matrix):
                    best_route = new_route
                    improved = True
        route = best_route
    return best_route

def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

# Aplicar optimización de 2-opt
optimized_route = two_opt(route, distance_matrix)
print("Ruta optimizada usando 2-opt:", optimized_route)

# Plot para visualizar la ruta optimizada
optimized_route_locations = locations[optimized_route]
plt.figure(figsize=(10, 8))
plt.plot(optimized_route_locations[:, 0], optimized_route_locations[:, 1], 'o-')
plt.scatter(locations[:, 0], locations[:, 1], marker='o', c='red', edgecolor='b')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ruta optimizada usando 2-opt')
plt.grid(True)
plt.show()
