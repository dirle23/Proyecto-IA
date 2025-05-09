# Sistema de Predicción de Riesgo Cardiovascular

Este proyecto implementa un sistema de predicción de riesgo cardiovascular utilizando machine learning y un bot de Telegram como interfaz de usuario.

## Estructura del Proyecto

El proyecto está compuesto por tres archivos principales:

1. `config.py`: Contiene la configuración del bot y los mensajes predefinidos
2. `modelos_prediccion.py`: Implementa los modelos de machine learning y las funciones de procesamiento de datos
3. `bot_telegram.py`: Implementa la interfaz del bot de Telegram

## Funcionalidad Principal

### Configuración del Bot (`config.py`)

- Contiene el token de autenticación de Telegram
- Define los mensajes de bienvenida y ayuda para el usuario
- Especifica el formato de entrada de datos requeridos

### Modelos de Predicción (`modelos_prediccion.py`)

#### Funciones Principales:

1. `cargar_y_preprocesar_datos(ruta_archivo)`
   - Carga y preprocesa los datos desde un archivo CSV
   - Maneja la imputación de valores faltantes
   - Normaliza las características numéricas

2. `seleccionar_modelos(tipo_archivo)`
   - Selecciona los modelos de machine learning apropiados según el tipo de análisis
   - Implementa modelos como Regresión Logística, Random Forest, Gradient Boosting, AdaBoost y SVM

3. `entrenar_modelos(X, y, tipo_archivo)`
   - Entrena múltiples modelos de machine learning
   - Calcula métricas de rendimiento (accuracy, F1-score, precision, recall)
   - Implementa técnicas de balanceo de clases

4. `hacer_prediccion(modelo, scaler, datos_usuario, columnas_pred)`
   - Realiza predicciones usando el modelo entrenado
   - Escala los datos de entrada según el mismo proceso de entrenamiento

### Bot de Telegram (`bot_telegram.py`)

#### Funcionalidades:

1. `/start` - Muestra el mensaje de bienvenida y ayuda
2. `/diabetes` - Inicia el análisis de riesgo de diabetes
3. `/hipertension` - Inicia el análisis de riesgo de hipertensión
4. `/infarto` - Inicia el análisis de riesgo de infarto

#### Proceso de Predicción:

1. El usuario proporciona 9 valores separados por comas:
   - Diabetes (0/1)
   - Sexo (0/1)
   - Edad
   - Fumador actual (0/1)
   - Colesterol total
   - Presión arterial sistólica
   - Presión arterial diastólica
   - IMC
   - Frecuencia cardíaca

2. El bot procesa los datos para tres tipos de análisis:
   - Diabetes
   - Hipertensión
   - Infarto

3. Para cada análisis, el bot:
   - Entrena múltiples modelos
   - Selecciona el mejor modelo basado en accuracy
   - Realiza la predicción
   - Muestra los resultados al usuario

## Requisitos Técnicos

- Python 3.x
- Bibliotecas necesarias:
  - telegram
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - imbalanced-learn

## Uso del Sistema

1. Iniciar el bot usando el comando `/start`
2. Proporcionar los 9 valores requeridos separados por comas
3. Esperar los resultados de los tres análisis
4. Revisar el riesgo para cada condición (Alto/Bajo)

## Consideraciones Importantes

- Los datos deben ser proporcionados en el orden especificado
- Los valores deben ser numéricos
- La precisión de las predicciones depende de la calidad de los datos de entrada
- El sistema utiliza técnicas de balanceo de clases para manejar datos desbalanceados

## Mantenimiento y Actualizaciones

- Los modelos se entrenan automáticamente con cada predicción
- Las métricas de rendimiento se calculan para cada modelo
- El sistema selecciona automáticamente el mejor modelo basado en accuracy
