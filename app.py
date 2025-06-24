from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelo y scaler
model = joblib.load('modelo_random_forest_top6.pkl')
scaler = joblib.load('scaler_top6.pkl')  # 👈 Nuevo
app.logger.debug('Modelo y scaler cargados correctamente.')

# Lista de features esperadas (orden correcto)
top_6_features = [
    'HIV/AIDS',
    'Income composition of resources',
    'Adult Mortality',
    'Schooling',
    'under-five deaths',
    'BMI'
]

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        input_vals = [float(request.form[feature]) for feature in top_6_features]
        
        # Crear DataFrame con nombres de columnas
        input_df = pd.DataFrame([input_vals], columns=top_6_features)
        app.logger.debug(f'Data cruda recibida: {input_df}')

        # Escalar los datos antes de predecir
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

        # Predicción
        prediction = model.predict(input_scaled_df)
        app.logger.debug(f'Predicción generada: {prediction[0]}')

        return jsonify({'prediccion': prediction[0]})
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400
