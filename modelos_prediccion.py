import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline, Pipeline

def cargar_y_preprocesar_datos(ruta_archivo):
    # Cargar datos
    df = pd.read_csv(ruta_archivo)
    
    # Identificar las columnas predictoras según el tipo de archivo
    tipo_archivo = os.path.basename(ruta_archivo).split('_')[0]
    
    # Orden consistente de columnas para todos los archivos
    columnas_base = ['Sex', 'Age', 'Current_Smoker', 'Cholesterol_Total', 'BP_Sist', 'BP_Diast', 'BMI', 'Heart_Rate']
    
    if tipo_archivo == 'diabetes':
        # Para diabetes: 8 variables (sin Diabetes)
        columnas_pred = columnas_base
    else:
        # Para hipertensión e infarto: 9 variables (con Diabetes)
        columnas_pred = ['Diabetes'] + columnas_base
    
    # Separar variables predictoras (X) y variable objetivo (y)
    X = df[columnas_pred].copy()  # Usar las columnas predictoras y crear una copia
    y = df['Risk'].copy()     # Columna Risk como variable objetivo
    
    # Convertir todas las columnas a numérico
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            continue
    
    # Manejar valores faltantes
    X = X.fillna(X.mean())
    
    return X, y

def seleccionar_modelos(tipo_archivo):
    modelos = {}
    
    # Seleccionar modelos según el tipo de archivo
    if tipo_archivo == 'hypertension':
        # Para hipertensión: usar modelos que manejen bien datos numéricos y relaciones lineales
        modelos = {
            # Modelos base
            'Regresión Logística': LogisticRegression(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            
            # Modelos optimizados
            'Random Forest Optimizado': RandomForestClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                class_weight='balanced_subsample'
            ),
            
            'Gradient Boosting Optimizado': GradientBoostingClassifier(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.5
            )
        }
    elif tipo_archivo == 'diabetes':
        # Para diabetes: usar modelos que manejen bien datos con alta variabilidad
        modelos = {
            # Modelos base
            'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=200),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced'),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True),
            
            # Modelos optimizados
            'AdaBoost Optimizado': AdaBoostClassifier(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05
            ),
            
            'Random Forest Optimizado': RandomForestClassifier(
                random_state=42,
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced_subsample'
            ),
            
            # Modelos con técnicas de re-muestreo
            'Random Forest SMOTE': Pipeline([
                ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=200))
            ]),
            
            'AdaBoost SMOTE': Pipeline([
                ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
                ('adaboost', AdaBoostClassifier(random_state=42, n_estimators=200))
            ])
        }
    elif tipo_archivo == 'heartattack':
        # Para infarto: usar modelos robustos para datos con patrones complejos y relaciones no lineales
        modelos = {
            # Modelos base
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=20,
                min_samples_split=3,
                class_weight='balanced_subsample'
            ),
            
            'Gradient Boosting Optimizado': GradientBoostingClassifier(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8
            )
        }
    
    # Verificar si se encontraron modelos
    if not modelos:
        raise ValueError(f"No se encontraron modelos para el tipo de archivo: {tipo_archivo}")
    
    return modelos

def entrenar_modelos(X, y, tipo_archivo):
    # Seleccionar modelos según el tipo de archivo
    modelos = seleccionar_modelos(tipo_archivo)
    
    # Dividir datos en entrenamiento y prueba (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Escalar características
    scaler = StandardScaler()
    
    # Escalar datos manteniendo los nombres de las columnas
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        try:
            # Entrenar modelo
            if isinstance(modelo, Pipeline):
                # Para pipelines que incluyen SMOTE
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
            else:
                # Para modelos regulares
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=1)
            recall = recall_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Guardar resultados
            resultados[nombre] = {
                'modelo': modelo,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': cm,
                'report': classification_report(y_test, y_pred)
            }
        except Exception as e:
            print(f"Error al entrenar el modelo {nombre}: {str(e)}")
            continue
    
    return resultados, scaler, X.columns.tolist()

def hacer_prediccion(modelo, scaler, datos_usuario, columnas_pred):
    try:
        # Convertir datos del usuario a DataFrame para mantener consistencia con el entrenamiento
        datos_df = pd.DataFrame([datos_usuario], columns=columnas_pred)
        
        # Verificar que los datos tengan la misma dimensión que las columnas esperadas
        if len(datos_df.columns) != len(columnas_pred):
            raise ValueError(f"Número incorrecto de valores. Se esperaban {len(columnas_pred)} valores.")
        
        # Escalar datos
        datos_scaled = scaler.transform(datos_df)
        
        # Hacer predicción
        prediccion = modelo.predict(datos_scaled)
        return prediccion[0]
    except ValueError as ve:
        print(f"Error de validación: {str(ve)}")
        raise
    except Exception as e:
        print(f"Error al hacer la predicción: {str(e)}")
        raise
        return None

def comparar_modelos(ruta_archivo):
    # Cargar y preprocesar datos
    X, y = cargar_y_preprocesar_datos(ruta_archivo)
    
    # Determinar el tipo de archivo
    tipo_archivo = os.path.basename(ruta_archivo).split('_')[0]
    
    # Guardar las columnas predictoras
    columnas_pred = X.columns
    
    # Entrenar modelos y obtener resultados
    resultados, scaler, feature_names = entrenar_modelos(X, y, tipo_archivo)
    
    # Mostrar resultados
    print("\nResultados para el archivo:", os.path.basename(ruta_archivo))
    print("-" * 50)
    
    for nombre, resultado in resultados.items():
        print(f"\n{nombre}:")
        print(f"Accuracy: {resultado['accuracy']:.4f}")
        print("Report de clasificación:")
        print(resultado['report'])
    
    # Ejemplo de predicción
    print("\nEjemplo de predicción:")
    print("-" * 50)
    
    # Usar el modelo con mejor accuracy
    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['accuracy'])[1]['modelo']
    
    # Ejemplo de datos de usuario (usando las mismas columnas que el modelo)
    ejemplo_datos = [1, 45, 0, 0, 200, 120, 80, 25.5, 75]  # Ejemplo de datos
    ejemplo_datos = ejemplo_datos[:len(columnas_pred)]  # Ajustar al número de columnas
    
    prediccion = hacer_prediccion(mejor_modelo, scaler, ejemplo_datos, columnas_pred)
    
    print("\nPredicción para los datos de ejemplo:", "Positivo" if prediccion == 1 else "Negativo")
    print("-" * 50)
    print("\n")
    
    return resultados, scaler, feature_names

if __name__ == "__main__":
    import os
    
    # Obtener la ruta del directorio actual
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de archivos
    archivos = [
        os.path.join(directorio_actual, "Archivos", "hypertension_v1.csv"),
        os.path.join(directorio_actual, "Archivos", "diabetes_v1.csv"),
        os.path.join(directorio_actual, "Archivos", "heartattack_v1.csv")
    ]
    
    # Procesar cada archivo
    for archivo in archivos:
        resultados, scaler, columnas_pred = comparar_modelos(archivo)
        
        # Ejemplo de cómo hacer una predicción
        print("\nEjemplo de predicción:")
        print("-" * 50)
        
        # Usar el modelo con mejor accuracy
        mejor_modelo = max(resultados.items(), key=lambda x: x[1]['accuracy'])[1]['modelo']
        
        # Ejemplo de datos de usuario (usando las mismas columnas que el modelo)
        ejemplo_datos = [1, 45, 0, 0, 200, 120, 80, 25.5, 75]  # Ejemplo de datos
        ejemplo_datos = ejemplo_datos[:len(columnas_pred)]  # Ajustar al número de columnas
        
        prediccion = hacer_prediccion(mejor_modelo, scaler, ejemplo_datos, columnas_pred)
        
        print("\nPredicción para los datos de ejemplo:", "Positivo" if prediccion == 1 else "Negativo")
        print("-" * 50)
        print("\n")
