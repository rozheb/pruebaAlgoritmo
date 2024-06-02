import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Coordenadas del inicio y fin de la ruta de todos los camiones
inicio_ruta = (-11.859431, -77.081518)
fin_ruta = (-11.822902, -77.059019)

# URL del archivo CSV
url = "https://raw.githubusercontent.com/rozheb/pruebaAlgoritmo/main/DATASETS/datasetrutas4.csv"

# Cargar datos desde la URL
datos = pd.read_csv(url, on_bad_lines='skip')

# Verificar las primeras filas y las columnas disponibles
print(datos.head())
print(datos.columns)

# Extraer las coordenadas de latitud y longitud de inicio y fin de la ruta
start_locations = datos[['LatitudInicial', 'LongitudInicial']].values
end_locations = datos[['LatitudFinal', 'LongitudFinal']].values

# Concatenar coordenadas de inicio y fin para cada ruta, incluyendo el inicio y fin de todos los camiones
locations = np.concatenate(([inicio_ruta], start_locations, end_locations, [fin_ruta]), axis=0)

# Plot para visualizar las ubicaciones de inicio y fin de las rutas
plt.figure(figsize=(10, 8))

# Graficar puntos de inicio y fin con colores distintos
plt.scatter(start_locations[:, 0], start_locations[:, 1], marker='o', c='blue', label='Inicio de ruta')
plt.scatter(end_locations[:, 0], end_locations[:, 1], marker='o', c='green', label='Fin de ruta')

# Graficar marcadores para el inicio y fin de la ruta de todos los camiones
plt.scatter(inicio_ruta[0], inicio_ruta[1], marker='*', s=200, c='purple', label='Inicio de ruta de todos los camiones')
plt.scatter(fin_ruta[0], fin_ruta[1], marker='*', s=200, c='orange', label='Fin de ruta de todos los camiones')

plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ubicaciones de inicio y fin de las rutas')
plt.legend()  # Mostrar la leyenda
plt.grid(True)
plt.show()

# Calcular matriz de distancias euclidianas
distance_matrix = cdist(locations, locations, metric='euclidean')

# Función para calcular la distancia total de un camino en el grafo
def total_distance(route, distance_matrix):
    n = len(route)
    total_dist = 0
    for i in range(n - 1):  # No cerramos el ciclo
        total_dist += distance_matrix[route[i], route[i + 1]]
    return total_dist

# Función para la optimización 2-opt
def two_opt(route, distance_matrix):
    n = len(route)
    best_route = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                if total_distance(new_route, distance_matrix) < total_distance(best_route, distance_matrix):
                    best_route = new_route
                    improved = True
        route = best_route
    return best_route

# Algoritmo de Búsqueda de Vecindario Variable (VNS)
def vns(distance_matrix, max_iter=100):
    n = distance_matrix.shape[0]
    # Inicializar ruta asegurando que comience en 0 y termine en n-1
    mejor_camino = [0] + list(np.random.permutation(range(1, n-1))) + [n-1]
    mejor_distancia = total_distance(mejor_camino, distance_matrix)
    
    iteracion = 0
    while iteracion < max_iter:
        k = 1
        while k <= 3:  # Número máximo de estructuras de vecindario
            if k == 1:
                # Intercambiar dos nodos aleatorios, excluyendo el primero y último
                i, j = np.random.randint(1, n-1, size=2)
                mejor_camino[i], mejor_camino[j] = mejor_camino[j], mejor_camino[i]
            elif k == 2:
                # Realizar una búsqueda local 2-opt en el vecindario
                mejor_camino = two_opt(mejor_camino, distance_matrix)
            elif k == 3:
                # Permutar subrutas aleatorias, excluyendo el primero y último
                i, j = np.random.randint(1, n-1, size=2)
                if i > j:
                    i, j = j, i
                mejor_camino[i:j] = mejor_camino[i:j][::-1]  # Invertir subruta
            
            # Calcular la distancia del camino actual
            distancia_actual = total_distance(mejor_camino, distance_matrix)
            
            # Actualizar el mejor camino si encontramos uno mejor
            if distancia_actual < mejor_distancia:
                mejor_distancia = distancia_actual
                k = 1  # Reiniciar el contador de estructuras de vecindario
            else:
                k += 1  # Probar la siguiente estructura de vecindario
        
        iteracion += 1
        # Imprimir la ruta en cada iteración
        ruta_actual_locations = locations[mejor_camino]
        print(f"Iteración {iteracion}, Ruta actual: {ruta_actual_locations}")
    
    return mejor_camino, mejor_distancia

# Ejecutar el algoritmo VNS para encontrar la ruta óptima
ruta_optima, distancia_optima = vns(distance_matrix)

# Mostrar la ruta óptima encontrada
print("Ruta óptima encontrada:", locations[ruta_optima])
print("Distancia óptima:", distancia_optima)

# Plot para visualizar la ruta óptima encontrada
ruta_optima_locations = locations[ruta_optima]
plt.figure(figsize=(10, 8))
plt.plot(ruta_optima_locations[:, 0], ruta_optima_locations[:, 1], 'o-')
plt.scatter(locations[:, 0], locations[:, 1], marker='o', c='red', edgecolor='b')
plt.scatter(start_locations[:, 0], start_locations[:, 1], marker='o', c='blue', label='Inicio de ruta')
plt.scatter(end_locations[:, 0], end_locations[:, 1], marker='o', c='green', label='Fin de ruta')
plt.scatter(inicio_ruta[0], inicio_ruta[1], marker='*', s=200, c='purple', label='Inicio de ruta de todos los camiones')
plt.scatter(fin_ruta[0], fin_ruta[1], marker='*', s=200, c='orange', label='Fin de ruta de todos los camiones')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ruta óptima encontrada usando Variable Neighborhood Search (VNS)')
plt.legend()  # Mostrar la leyenda
plt.grid(True)
plt.show()
