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
datos = pd.read_csv("https://raw.githubusercontent.com/rozheb/pruebaAlgoritmo/main/datasetrutas.csv", on_bad_lines='skip')
datos

print("Observaciones y variables: ", datos.shape)

## Columnas y tipo de dato
datos.dtypes

#carga de datos 
X = datos.drop(columns='xi')
y = datos['xi']
# Identificar las columnas categóricas
columnas_categoricas = X.select_dtypes(include=['object']).columns

# Crear el transformador para las variables categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), columnas_categoricas)
    ],
    remainder='passthrough'  # Mantener las columnas numéricas tal como están
)

# Dividir los datos en entrenamiento y validación
X_entrena, X_valida, Y_entrena, Y_valida = train_test_split(X, y, train_size=0.80, random_state=1280)

# Aplicar el transformador a los datos de entrenamiento y validación
X_entrena = preprocessor.fit_transform(X_entrena)
X_valida = preprocessor.transform(X_valida)

# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_entrena, Y_entrena)

# Evaluar el modelo
y_pred = knn.predict(X_valida)
print("Accuracy:", accuracy_score(Y_valida, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_valida, y_pred))
print("Classification Report:\n", classification_report(Y_valida, y_pred))
