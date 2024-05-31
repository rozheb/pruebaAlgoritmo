# Tratamiento de datos
import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices

# Estadísticas
import scipy 
from scipy import stats

# Para partir datos entrenamiento y validación
from sklearn.model_selection import train_test_split

# Modelo de Clasificación 
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sb

## cargar datos
datos = pd.read_csv("https://raw.githubusercontent.com/rozheb/pruebaAlgoritmo/main/datasetrutas.csv")
datos

print("Observaciones y variables: ", datos.shape)
print("Columnas y tipo de dato")
datos.columns
datos.dtypes

