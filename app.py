from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos y características del Titanic
model = joblib.load('modelo_knn.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
features = joblib.load('features.pkl')  # Cargar las características exactas

app.logger.debug('Modelo KNN, scaler, PCA y características cargados correctamente.')
app.logger.info(f'Características cargadas: {features}')
app.logger.info(f'Número de características: {len(features)}')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/features')
def get_features():
    """Endpoint para obtener las características del modelo"""
    return jsonify({
        'features': features,
        'count': len(features)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario y crear diccionario completo
        form_data = {
            'Sex_female': 1 if request.form['sex'] == 'female' else 0,
            'Age': float(request.form['age']),
            'Fare': float(request.form['fare']),
            'Pclass': int(request.form['pclass']),
            'Cabin': 1 if request.form['cabin'] == 'yes' else 0
        }
        
        app.logger.debug(f'Datos del formulario: {form_data}')
        
        # Crear DataFrame con TODAS las características posibles
        X_new = pd.DataFrame([form_data])
        app.logger.debug(f'DataFrame inicial: {X_new}')
        app.logger.debug(f'Columnas iniciales: {list(X_new.columns)}')
        
        # Reordenar/filtrar usando las características exactas del entrenamiento
        X_new = X_new[features]
        app.logger.debug(f'DataFrame después de reordenar: {X_new}')
        app.logger.debug(f'Columnas finales: {list(X_new.columns)}')
        app.logger.debug(f'Valores finales: {X_new.values[0]}')

        # Pipeline exacto: scaler → PCA → modelo
        # 1. Escalar los datos
        X_scaled = scaler.transform(X_new)
        app.logger.debug(f'Datos escalados - forma: {X_scaled.shape}')
        
        # 2. Aplicar PCA
        X_pca = pca.transform(X_scaled)
        app.logger.debug(f'Datos después de PCA - forma: {X_pca.shape}')

        # 3. Predicción con KNN
        prediction = model.predict(X_pca)
        prediction_proba = model.predict_proba(X_pca)
        
        app.logger.debug(f'Predicción: {prediction[0]}')
        app.logger.debug(f'Probabilidades: {prediction_proba[0]}')

        # Interpretar resultado
        survived = bool(prediction[0])
        probability = float(prediction_proba[0][1])  # Probabilidad de supervivencia
        
        return jsonify({
            'survived': survived,
            'probability': probability,
            'message': 'Sobrevive' if survived else 'No sobrevive',
            'features_used': features,
            'input_data': form_data
        })
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        import traceback
        app.logger.error(f'Traceback completo: {traceback.format_exc()}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
