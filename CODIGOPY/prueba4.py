import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Puntos de inicio y fin común para todos los camiones
start_common = np.array([-11.859431, -77.081518])
end_common = np.array([-11.822902, -77.059019])

# URL del archivo CSV
url = "https://raw.githubusercontent.com/rozheb/pruebaAlgoritmo/main/datasetrutas4.csv"

# Cargar datos desde la URL
datos = pd.read_csv(url,on_bad_lines='skip')

# Extraer las coordenadas de latitud y longitud de inicio y fin de la ruta
start_locations = datos[['LatitudInicial', 'LongitudInicial']].values
end_locations = datos[['LatitudFinal', 'LongitudFinal']].values

# Concatenar coordenadas de inicio y fin para cada ruta y agregar los puntos comunes
locations = np.concatenate((start_common.reshape(1, 2), start_locations, end_locations, end_common.reshape(1, 2)), axis=0)

# Calcular matriz de distancias euclidianas
distance_matrix = cdist(locations, locations, metric='euclidean')

def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

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

def generate_initial_solution(distance_matrix):
    start_index = 0  # Comienza en el punto de inicio común
    return nearest_neighbor_route(start_index, distance_matrix)

def variable_neighborhood_search(distance_matrix, k_max, max_iterations):
    # Generar una solución inicial usando el vecino más cercano
    current_solution = generate_initial_solution(distance_matrix)
    best_solution = current_solution[:]
    best_distance = total_distance(current_solution, distance_matrix)
    
    k = 1
    iterations = 0
    while k <= k_max and iterations < max_iterations:
        new_solution = perturb(current_solution, k)
        new_solution = two_opt(new_solution, distance_matrix)
        new_distance = total_distance(new_solution, distance_matrix)
        
        if new_distance < best_distance:
            best_solution = new_solution[:]
            best_distance = new_distance
            current_solution = new_solution[:]
            k = 1  # Reiniciar el contador de vecindarios cuando se encuentra una mejor solución
        else:
            k += 1
        
        iterations += 1
    
    return best_solution, best_distance

def perturb(solution, k):
    n = len(solution)
    i = np.random.randint(1, n - 1)  # Excluye el primer y último punto (inicio y fin común)
    j = np.random.randint(1, n - 1)  # Excluye el primer y último punto (inicio y fin común)
    
    if i > j:
        i, j = j, i
    
    if j - i >= 2:
        return solution[:i] + solution[i:j][::-1] + solution[j:]
    else:
        return solution

# Parámetros del VNS
k_max = 3  # Máximo número de vecindarios
max_iterations = 100  # Máximo número de iteraciones

# Ejecutar Variable Neighborhood Search
optimal_route, optimal_distance = variable_neighborhood_search(distance_matrix, k_max, max_iterations)

# Mostrar la ruta encontrada y su distancia
print("Ruta óptima encontrada:", optimal_route)
print("Distancia óptima:", optimal_distance)

# Visualización de la ruta encontrada
optimal_route_locations = locations[optimal_route]
plt.figure(figsize=(10, 8))

# Dibujar los puntos de inicio y fin de todos los caminones
plt.plot(start_common[0], start_common[1], 'o', markersize=10, label='Inicio de todos los caminones', color='green')
plt.plot(end_common[0], end_common[1], 'o', markersize=10, label='fin de todos los caminones', color='blue')

# Dibujar la ruta óptima encontrada
plt.plot(optimal_route_locations[:, 0], optimal_route_locations[:, 1], 'o-')
plt.scatter(locations[:, 0], locations[:, 1], marker='o', c='red', edgecolor='b')

plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ruta óptima encontrada usando VNS')
plt.legend()
plt.grid(True)
plt.show()
