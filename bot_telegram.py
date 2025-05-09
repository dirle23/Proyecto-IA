import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from modelos_prediccion import cargar_y_preprocesar_datos, entrenar_modelos, hacer_prediccion
import config

def obtener_mejor_modelo(resultados):
    # Obtener el modelo con mejor accuracy
    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['accuracy'])[0]
    mejor_accuracy = max(resultados.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']
    # Devolver el objeto del modelo en lugar del nombre
    return resultados[mejor_modelo]['modelo'], mejor_accuracy

def obtener_variables(tipo_archivo):
    if tipo_archivo == 'diabetes':
        return ['Sex', 'Age', 'Current_Smoker', 'Cholesterol_Total', 'BMI', 'Heart_Rate', 'BP_Sist', 'BP_Diast']
    else:
        return ['Sex', 'Age', 'Current_Smoker', 'Diabetes', 'Cholesterol_Total', 'BP_Sist', 'BP_Diast', 'BMI', 'Heart_Rate']

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=config.MENSAJE_BIENVENIDA
    )

async def predecir(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Enviar mensaje de que se está analizando
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Analizando tus datos..."
        )
        
        # Obtener los datos del mensaje
        datos = update.message.text
        
        # Dividir los datos por comas
        valores = [float(x.strip()) for x in datos.split(',')]
        
        # Verificar que se hayan proporcionado 9 valores
        if len(valores) != 9:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Error: Debes proporcionar 9 valores.\nOrden: Diabetes, Sexo, Edad, Fumador, Colesterol, Presión Sistólica, Presión Diastólica, IMC, Frecuencia Cardíaca"
            )
            return
        
        # Crear mensaje de resultados
        mensaje = "Resultados de los análisis:\n\n"
        
        # Procesar los tres tipos de análisis
        for tipo_analisis in ['diabetes', 'hypertension', 'heartattack']:
            # Para diabetes, omitir el primer valor (Diabetes)
            valores_procesar = valores[1:] if tipo_analisis == 'diabetes' else valores
            
            # Cargar y preprocesar datos
            ruta_archivo = os.path.join(os.path.dirname(__file__), "Archivos", f"{tipo_analisis}_v1.csv")
            X, y = cargar_y_preprocesar_datos(ruta_archivo)
            
            # Entrenar modelos
            resultados, scaler, columnas_pred = entrenar_modelos(X, y, tipo_analisis)
            
            # Obtener el mejor modelo
            mejor_modelo, mejor_accuracy = obtener_mejor_modelo(resultados)
            
            # Hacer la predicción
            prediccion = hacer_prediccion(mejor_modelo, scaler, valores_procesar, columnas_pred)
            
            # Agregar resultados al mensaje
            mensaje += f"\n{tipo_analisis.capitalize()}:\n"
            mensaje += f"Modelo: {mejor_modelo}\n"
            mensaje += f"Precisión: {mejor_accuracy:.1%}\n"
            mensaje += f"Riesgo: {'Alto' if prediccion == 1 else 'Bajo'}\n"
            
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=mensaje
        )
        
    except Exception as e:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Error al procesar los datos: {str(e)}\nPor favor, verifica que has ingresado los valores correctos y en el formato adecuado."
        )

async def seleccionar_analisis(update: Update, context: ContextTypes.DEFAULT_TYPE, tipo: str):
    context.user_data['tipo_analisis'] = tipo
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Has seleccionado el análisis de {tipo}. Ahora puedes proporcionar los datos para la predicción."
    )

def main():
    # Crear la aplicación
    application = Application.builder().token(config.TELEGRAM_TOKEN).build()
    
    # Agregar los manejadores de comandos
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("diabetes", lambda update, context: seleccionar_analisis(update, context, 'diabetes')))
    application.add_handler(CommandHandler("hipertension", lambda update, context: seleccionar_analisis(update, context, 'hypertension')))
    application.add_handler(CommandHandler("infarto", lambda update, context: seleccionar_analisis(update, context, 'heartattack')))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predecir))
    
    # Agregar manejador de errores
    application.add_error_handler(error_handler)
    
    # Iniciar el bot
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        print(f"Error al iniciar el bot: {str(e)}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manejador de errores para el bot."""
    try:
        # Obtener el mensaje de error
        error = context.error
        
        # Registrar el error
        print(f"Error en el bot: {str(error)}")
        
        # Enviar mensaje de error al usuario
        if update and isinstance(update, Update):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Lo siento, ha ocurrido un error. Por favor, intenta nuevamente."
            )
    except Exception as e:
        print(f"Error en el manejador de errores: {str(e)}")

if __name__ == "__main__":
    main()
