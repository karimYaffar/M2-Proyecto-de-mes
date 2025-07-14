from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos del Titanic
model = joblib.load('modelo_knn.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
app.logger.debug('Modelo KNN, scaler y PCA cargados correctamente.')

# Lista de features esperadas para el Titanic
titanic_features = ['Sex_female', 'Age', 'Fare', 'Pclass', 'Cabin']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        sex_female = 1 if request.form['sex'] == 'female' else 0
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        pclass = int(request.form['pclass'])
        cabin = 1 if request.form['cabin'] == 'yes' else 0
        
        # Crear array con los valores en el orden correcto
        input_vals = [sex_female, age, fare, pclass, cabin]
        
        # Crear DataFrame con nombres de columnas
        input_df = pd.DataFrame([input_vals], columns=titanic_features)
        app.logger.debug(f'Data cruda recibida: {input_df}')

        # Escalar los datos
        input_scaled = scaler.transform(input_df)
        app.logger.debug(f'Data escalada: {input_scaled}')
        
        # Aplicar PCA
        input_pca = pca.transform(input_scaled)
        app.logger.debug(f'Data después de PCA: {input_pca}')

        # Predicción
        prediction = model.predict(input_pca)
        prediction_proba = model.predict_proba(input_pca)
        
        app.logger.debug(f'Predicción generada: {prediction[0]}')
        app.logger.debug(f'Probabilidades: {prediction_proba[0]}')

        # Interpretar resultado
        survived = bool(prediction[0])
        probability = float(prediction_proba[0][1])  # Probabilidad de supervivencia
        
        return jsonify({
            'survived': survived,
            'probability': probability,
            'message': 'Sobrevive' if survived else 'No sobrevive'
        })
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
