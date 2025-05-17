'''
Ejercicio 2:
Identifica un problema de la localidad y argumente, las características del problema,    
porque el uso de las redes neuronales puede resolverlo e ilustrar con seudo-código u otra forma como se usaría en esta solución. 
Pueden ser problemas asociados a: 
     * La Educación.
     * La agricultura.
     * La salud.
     * La gestión de los servicios públicos y el gobierno popular.
     * La cultura.
     * Otro tema de la localidad debidamente argumentado.o en el rendimiento. A diferencia de modelos lineales, detectan patrones no evidentes en datos históricos.


Problema:
En zonas agrícolas, los cultivos sufren pérdidas por condiciones climáticas impredecibles y falta de planificación. 
Los agricultores necesitan predecir el rendimiento de sus cultivos para optimizar recursos.
    Variables de entrada: Lluvia mensual (mm), temperatura promedio, tipo de suelo, uso de fertilizantes.
    Salida: Rendimiento estimado (kg/hectárea).

¿Por qué usar redes neuronales?
Las redes pueden capturar relaciones no lineales entre variables climáticas y rendimiento, 
además de adaptarse a cambios estacionales y múltiples factores simultáneos.
'''

#Pseudocódigo:
# 1. Cargar datos históricos (ejemplo)
datos = pd.read_csv('datos_cosechas.csv')
X = datos[['lluvia', 'temperatura', 'fertilizante', 'tipo_suelo']]
y = datos['rendimiento']

# 2. Preprocesamiento
escalador = StandardScaler()
X = escalador.fit_transform(X)
X_entrenar, X_validar, y_entrenar, y_validar = train_test_split(X, y, test_size=0.2)

# 3. Construir red neuronal
modelo = Sequential()
modelo.add(Dense(32, activation='relu', input_dim=4))  # Capa oculta
modelo.add(Dense(1))  # Salida para regresión

modelo.compile(optimizer='adam', loss='mse')  # Error cuadrático medio

# 4. Entrenar
modelo.fit(X_entrenar, y_entrenar, epochs=200, batch_size=32)

# 5. Predicción
nuevos_datos = escalador.transform([[150, 28, 50, 2]])  # Ejemplo: Lluvia=150mm, Temp=28°C, etc.
rendimiento_predicho = modelo.predict(nuevos_datos)
print(f'Rendimiento estimado: {rendimiento_predicho[0][0]:.2f} kg/ha')