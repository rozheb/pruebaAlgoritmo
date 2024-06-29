import pandas as pd
import numpy as np
import random

# Cargar los datos
url = "https://raw.githubusercontent.com/rozheb/pruebaAlgoritmo/main/datasetrutas3.csv"
datos = pd.read_csv(url,on_bad_lines='skip')

# Definir los atributos de interés
turnos = datos['turno'].unique()
nombres = datos['Nombre'].unique()
latitudes = datos['Latitud'].unique()
longitudes = datos['Longitud'].unique()
frecuencias = datos['frecuencia'].unique()

# Función para evaluar la solución
def evaluar_solucion(solucion):
    """
    Evalúa la solución asignada.
    Retorna un valor de costo (por ejemplo, número de rutas asignadas).
    """
    # Ejemplo: Calcular algún tipo de métrica o costo asociado a la solución
    return sum(solucion.values())

# Función para generar una solución inicial aleatoria
def generar_solucion_inicial():
    """
    Genera una solución inicial aleatoria.
    Retorna un diccionario con las rutas asignadas a cada combinación de atributos.
    """
    solucion = {}

    for turno in turnos:
        for nombre in nombres:
            for latitud in latitudes:
                for longitud in longitudes:
                    for frecuencia in frecuencias:
                        # Generar un número aleatorio de rutas asignadas (entre 0 y 5, por ejemplo)
                        solucion[(turno, nombre, latitud, longitud, frecuencia)] = random.randint(0, 5)

    return solucion

# Algoritmo Variable Neighborhood Search (VNS)
def VNS(max_iter):
    """
    Implementación del algoritmo Variable Neighborhood Search (VNS).
    max_iter: número máximo de iteraciones.
    """
    mejor_solucion = generar_solucion_inicial()
    mejor_valor = evaluar_solucion(mejor_solucion)

    iteracion = 0
    while iteracion < max_iter:
        # Generar una nueva solución vecina modificando aleatoriamente la mejor solución actual
        vecindad = generar_solucion_inicial()

        # Evaluar la nueva solución vecina
        valor_vecino = evaluar_solucion(vecindad)

        # Aceptar la nueva solución si mejora el valor
        if valor_vecino < mejor_valor:
            mejor_solucion = vecindad
            mejor_valor = valor_vecino
            iteracion = 0  # Reiniciar contador de iteraciones si encontramos una mejor solución
        else:
            iteracion += 1

    return mejor_solucion, mejor_valor

# Ejecutar el algoritmo VNS
mejor_solucion, mejor_valor = VNS(max_iter=1000)

# Mostrar resultados
print(f"Mejor solución encontrada:\n{mejor_solucion}")
print(f"Valor de la mejor solución: {mejor_valor}")
