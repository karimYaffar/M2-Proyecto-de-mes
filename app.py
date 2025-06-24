from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
import os

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Verificar si el modelo existe antes de cargarlo
model_path = 'modelo_random_forest_top6.pkl'
if not os.path.exists(model_path):
    app.logger.error(f'El archivo del modelo {model_path} no existe.')
    model = None
else:
    try:
        model = joblib.load(model_path)
        app.logger.debug('Modelo cargado correctamente.')
    except Exception as e:
        app.logger.error(f'Error al cargar el modelo: {str(e)}')
        model = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    try:
        # Obtener los valores del formulario y convertir a float
        hiv = float(request.form['HIV/AIDS'])
        income = float(request.form['Income composition of resources'])
        mortality = float(request.form['Adult Mortality'])
        schooling = float(request.form['Schooling'])
        under5 = float(request.form['under-five deaths'])
        bmi = float(request.form['BMI'])

        # Validar rangos de datos (opcional pero recomendado)
        if not (0 <= hiv <= 100):
            return jsonify({'error': 'HIV/AIDS debe estar entre 0 y 100'}), 400
        if not (0 <= income <= 1):
            return jsonify({'error': 'Income composition debe estar entre 0 y 1'}), 400
        if not (0 <= mortality <= 1000):
            return jsonify({'error': 'Adult Mortality debe ser un valor válido'}), 400
        if not (0 <= schooling <= 25):
            return jsonify({'error': 'Schooling debe estar entre 0 y 25 años'}), 400
        if not (0 <= bmi <= 100):
            return jsonify({'error': 'BMI debe estar entre 0 y 100'}), 400

        # Crear DataFrame con los datos de entrada
        # IMPORTANTE: El orden debe coincidir exactamente con el entrenamiento
        input_data = pd.DataFrame([[hiv, income, mortality, schooling, under5, bmi]],
                                  columns=[
                                      'HIV/AIDS',
                                      'Income composition of resources',
                                      'Adult Mortality',
                                      'Schooling',
                                      'under-five deaths',
                                      'BMI'
                                  ])
        
        app.logger.debug(f'DataFrame de entrada: {input_data}')
        app.logger.debug(f'Columnas del DataFrame: {input_data.columns.tolist()}')

        # Realizar la predicción
        prediction = model.predict(input_data)
        app.logger.debug(f'Predicción generada: {prediction[0]}')

        # Validar que la predicción sea razonable
        if prediction[0] < 0 or prediction[0] > 120:
            app.logger.warning(f'Predicción fuera de rango esperado: {prediction[0]}')

        # Retornar resultado
        return jsonify({'prediccion': float(prediction[0])})
    
    except ValueError as e:
        app.logger.error(f'Error de valor en la predicción: {str(e)}')
        return jsonify({'error': 'Valores de entrada inválidos'}), 400
    except KeyError as e:
        app.logger.error(f'Campo faltante en el formulario: {str(e)}')
        return jsonify({'error': f'Campo faltante: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f'Error inesperado en la predicción: {str(e)}')
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/health')
def health_check():
    """Endpoint para verificar el estado de la aplicación"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

""" if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
 """