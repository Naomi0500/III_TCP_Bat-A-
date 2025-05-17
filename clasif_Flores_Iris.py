'''
Ejercicio 1: Clasificación Multiclase - Flores Iris
•  Objetivo: Clasificar las flores Iris en tres especies diferentes (Setosa, Versicolour, Virginica) basándose en la longitud y anchura de sus sépalos y pétalos.
•  Conjunto de Datos: Iris (disponible en Scikit-learn y Keras).
•  Tipo de Red: Red neuronal densa (MLP).

README

    El código carga el conjunto de datos Iris y divide los datos en entrenamiento/prueba.

    Las etiquetas se convierten a one-hot encoding para la clasificación multiclase.

    Se escala los datos para normalizar las características.

    La red neuronal tiene una capa oculta con 10 neuronas y función de activación ReLU, y una capa de salida con 3 neuronas (softmax).

    Se entrena el modelo durante 100 épocas y se evalúa su precisión.

    Finalmente, se guarda el modelo y se muestra una predicción de ejemplo.

'''

# Importar bibliotecas
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Cargar datos
datos = load_iris()
X = datos.data  # Características: longitud y anchura de sépalos y pétalos
y = datos.target  # Etiquetas: 0=Setosa, 1=Versicolour, 2=Virginica

# Preprocesamiento
y = to_categorical(y)  # Convertir a one-hot encoding
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3)

# Escalar características para mejorar el entrenamiento
escalador = StandardScaler()
X_entrenamiento = escalador.fit_transform(X_entrenamiento)
X_prueba = escalador.transform(X_prueba)

# Construir la red neuronal
modelo = Sequential()
modelo.add(Dense(10, activation='relu', input_shape=(4,)))  # Capa oculta con 10 neuronas
modelo.add(Dense(3, activation='softmax'))  # Capa de salida para 3 clases

# Compilar el modelo
modelo.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Entrenar la red
historial = modelo.fit(
    X_entrenamiento,
    y_entrenamiento,
    epochs=100,
    batch_size=16,
    validation_split=0.2
)

# Evaluar el modelo
pérdida, precisión = modelo.evaluate(X_prueba, y_prueba)
print(f'Precisión en datos de prueba: {precisión * 100:.2f}%')

# Guardar el modelo entrenado
modelo.save('modelo_iris.h5')

# Ejemplo de predicción
muestra = X_prueba[0].reshape(1, -1)  # Tomar una muestra de prueba
predicción = modelo.predict(muestra)
clase_predicha = np.argmax(predicción)
print(f'Clase predicha: {datos.target_names[clase_predicha]}')