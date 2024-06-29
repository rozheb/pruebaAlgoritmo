import pandas as pd
import numpy as np
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

# Extraer las coordenadas y otros atributos relevantes
locations = datos[['LAT', 'LONG']].values
turnos = datos['TURNO'].values
distancias = datos['DIST'].values
capacidades = datos['CAPACIDAD'].values

# Concatenar coordenadas de inicio y fin para cada ruta, incluyendo el inicio y fin de todos los camiones
locations = np.concatenate(([inicio_ruta], locations, [fin_ruta]), axis=0)

# Plot para visualizar las ubicaciones de inicio y fin de las rutas
plt.figure(figsize=(10, 8))

# Graficar puntos de inicio y fin con colores distintos
plt.scatter(locations[1:-1, 0], locations[1:-1, 1], marker='o', c='blue', label='Puntos de recolección')

# Graficar marcadores para el inicio y fin de la ruta de todos los camiones
plt.scatter(inicio_ruta[0], inicio_ruta[1], marker='*', s=200, c='purple', label='Inicio de ruta de todos los camiones')
plt.scatter(fin_ruta[0], fin_ruta[1], marker='*', s=200, c='orange', label='Fin de ruta de todos los camiones')

plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ubicaciones de recolección')
plt.legend()  # Mostrar la leyenda
plt.grid(True)
plt.show()

# Calcular matriz de distancias euclidianas
distance_matrix = cdist(locations, locations, metric='euclidean')

# Función para calcular la distancia total de un camino en el grafo
def total_distance(route, distance_matrix, demandas, capacidad):
    n = len(route)
    total_dist = 0
    total_demand = 0
    for i in range(n - 1):
        if total_demand + demandas[route[i]] > capacidad:
            return float('inf')  # Penalizar si excede la capacidad
        total_dist += distance_matrix[route[i], route[i + 1]]
        total_demand += demandas[route[i]]
    return total_dist

# Función para la optimización 2-opt
def two_opt(route, distance_matrix, demandas, capacidad):
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
                if total_distance(new_route, distance_matrix, demandas, capacidad) < total_distance(best_route, distance_matrix, demandas, capacidad):
                    best_route = new_route
                    improved = True
        route = best_route
    return best_route

# Algoritmo de Búsqueda de Vecindario Variable (VNS)
def vns(distance_matrix, demandas, capacidad, max_iter=100):
    n = distance_matrix.shape[0]
    mejor_camino = [0] + list(np.random.permutation(range(1, n-1))) + [n-1]
    mejor_distancia = total_distance(mejor_camino, distance_matrix, demandas, capacidad)
    
    iteracion = 0
    while iteracion < max_iter:
        k = 1
        while k <= 3:
            if k == 1:
                i, j = np.random.randint(1, n-1, size=2)
                mejor_camino[i], mejor_camino[j] = mejor_camino[j], mejor_camino[i]
            elif k == 2:
                mejor_camino = two_opt(mejor_camino, distance_matrix, demandas, capacidad)
            elif k == 3:
                i, j = np.random.randint(1, n-1, size=2)
                if i > j:
                    i, j = j, i
                mejor_camino[i:j] = mejor_camino[i:j][::-1]
            
            distancia_actual = total_distance(mejor_camino, distance_matrix, demandas, capacidad)
            
            if distancia_actual < mejor_distancia:
                mejor_distancia = distancia_actual
                k = 1
            else:
                k += 1
        
        iteracion += 1
        ruta_actual_locations = locations[mejor_camino]
        print(f"Iteración {iteracion}, Ruta actual: {ruta_actual_locations}")
    
    return mejor_camino, mejor_distancia

# Asumimos que la capacidad del camión es un valor constante (ajusta según sea necesario)
capacidad_camion = 10  # Capacidad en toneladas

# Demanda de recolección en cada punto (ejemplo)
demandas = np.concatenate(([0], capacidades, [0]))

# Ejecutar el algoritmo VNS para encontrar la ruta óptima
ruta_optima, distancia_optima = vns(distance_matrix, demandas, capacidad_camion)

# Mostrar la ruta óptima encontrada
print("Ruta óptima encontrada:", locations[ruta_optima])
print("Distancia óptima:", distancia_optima)

# Plot para visualizar la ruta óptima encontrada
ruta_optima_locations = locations[ruta_optima]
plt.figure(figsize=(10, 8))
plt.plot(ruta_optima_locations[:, 0], ruta_optima_locations[:, 1], 'o-')
plt.scatter(locations[:, 0], locations[:, 1], marker='o', c='red', edgecolor='b')
plt.scatter(locations[1:-1, 0], locations[1:-1, 1], marker='o', c='blue', label='Puntos de recolección')
plt.scatter(inicio_ruta[0], inicio_ruta[1], marker='*', s=200, c='purple', label='Inicio de ruta de todos los camiones')
plt.scatter(fin_ruta[0], fin_ruta[1], marker='*', s=200, c='orange', label='Fin de ruta de todos los camiones')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.title('Ruta óptima encontrada usando Variable Neighborhood Search (VNS)')
plt.legend()  # Mostrar la leyenda
plt.grid(True)
plt.show()
